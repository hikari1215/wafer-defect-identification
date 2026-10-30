import argparse
import csv
import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Các định dạng file được hỗ trợ
IMG_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
VID_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script suy luận YOLO hoàn chỉnh - Sạch Warning, hỗ trợ CUDA/CPU, lưu Output & CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Đường dẫn file weights (.pt, .engine, .onnx)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Đường dẫn ảnh, video, thư mục hoặc index camera (ví dụ: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Thiết bị chạy: '0' hoặc 'cuda' cho GPU, 'cpu' cho CPU. Để trống tự chọn.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Bật FP16 (Half Precision) cho GPU CUDA để tăng tốc",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output",
        help="Thư mục lưu ảnh/video kết quả và file CSV (Mặc định: output)",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="results.csv",
        help="Tên file CSV thống kê (Mặc định: results.csv)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng độ tin cậy (Confidence threshold)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Không hiển thị cửa sổ preview",
    )
    return parser.parse_args()


def log_to_csv(csv_path, data_rows):
    """Ghi hoặc nối dữ liệu nhận diện vào file CSV"""
    file_exists = os.path.exists(csv_path)
    header = [
        "Filename",
        "Class_ID",
        "Class_Name",
        "Confidence (%)",
        "Type",
        "Device",
        "Output_Path",
    ]

    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for row in data_rows:
            writer.writerow(row)


def process_single_image(model, img_path, args, out_dir, csv_path, device_str):
    """Xử lý 1 file ảnh duy nhất, xuất ảnh vào thư mục output và ghi CSV"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Lỗi] Không thể đọc file ảnh: {img_path}")
        return

    # Tắt autograd để tối ưu suy luận
    with torch.inference_mode():
        results = model.predict(
            img,
            conf=args.conf,
            device=args.device if args.device else None,
            verbose=False,
        )

    res = results[0]
    annotated_img = res.plot()

    p = Path(img_path)
    out_img_path = out_dir / f"out_{p.name}"
    cv2.imwrite(str(out_img_path), annotated_img)

    csv_rows = []
    filename = p.name

    # 1. TRƯỜNG HỢP OBJECT DETECTION
    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0]) * 100
            cls_name = model.names[cls_id]

            print(
                f"--> Detected Box: {cls_name} (Class {cls_id}) | Conf: {conf_val:.2f}% | Device: {device_str}"
            )
            csv_rows.append([
                filename,
                cls_id,
                cls_name,
                f"{conf_val:.2f}",
                "Detection",
                device_str,
                str(out_img_path.resolve()),
            ])

    # 2. TRƯỜNG HỢP IMAGE CLASSIFICATION
    elif res.probs is not None:
        top1_id = int(res.probs.top1)
        top1_conf = float(res.probs.top1conf) * 100
        cls_name = model.names[top1_id]

        print(
            f"--> Detected Class: {cls_name} (Class {top1_id}) | Conf: {top1_conf:.2f}% | Device: {device_str}"
        )
        csv_rows.append([
            filename,
            top1_id,
            cls_name,
            f"{top1_conf:.2f}",
            "Classification",
            device_str,
            str(out_img_path.resolve()),
        ])

    # 3. KHÔNG PHÁT HIỆN LỖI
    else:
        print("--> Không phát hiện defect nào vượt ngưỡng confidence.")
        csv_rows.append([
            filename,
            "N/A",
            "None",
            "0.00",
            "No Detection",
            device_str,
            str(out_img_path.resolve()),
        ])

    log_to_csv(csv_path, csv_rows)
    print(f"[THÀNH CÔNG] Ảnh kết quả: {out_img_path.resolve()}")
    print(f"[THÀNH CÔNG] Đã lưu log CSV: {csv_path.resolve()}")

    if not args.no_show:
        cv2.imshow("YOLO Result", annotated_img)
        print("[Info] Nhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_directory(model, dir_path, args, out_dir, csv_path, device_str):
    """Xử lý toàn bộ ảnh trong 1 thư mục"""
    img_files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(IMG_FORMATS)
    ]

    if not img_files:
        print(f"[Cảnh báo] Không tìm thấy file ảnh hợp lệ nào trong: {dir_path}")
        return

    print(f"--> Tìm thấy {len(img_files)} ảnh. Đang xử lý trên [{device_str}]...")
    for idx, img_path in enumerate(img_files, 1):
        print(f"\n[{idx}/{len(img_files)}] Xử lý: {os.path.basename(img_path)}")
        process_single_image(model, img_path, args, out_dir, csv_path, device_str)


def process_video_or_webcam(model, video_source, args, out_dir, csv_path, device_str):
    """Xử lý Video file hoặc Webcam"""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Lỗi] Không mở được nguồn Video/Camera: {video_source}")
        return

    if isinstance(video_source, str):
        p = Path(video_source)
        out_vid_path = out_dir / f"out_{p.stem}.mp4"
    else:
        out_vid_path = out_dir / "out_webcam.mp4"

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (w, h))

    print(f"--> Chạy stream trên [{device_str}], lưu video tại: {out_vid_path.resolve()}")

    frame_idx = 0
    source_name = "webcam" if isinstance(video_source, int) else Path(video_source).name

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        with torch.inference_mode():
            results = model.predict(
                frame,
                conf=args.conf,
                device=args.device if args.device else None,
                verbose=False,
            )

        res = results[0]
        annotated_frame = res.plot()
        writer.write(annotated_frame)

        csv_rows = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0]) * 100
                cls_name = model.names[cls_id]
                csv_rows.append([
                    f"{source_name}_f{frame_idx}",
                    cls_id,
                    cls_name,
                    f"{conf_val:.2f}",
                    "Video-Detection",
                    device_str,
                    str(out_vid_path.resolve()),
                ])
        elif res.probs is not None:
            top1_id = int(res.probs.top1)
            top1_conf = float(res.probs.top1conf) * 100
            cls_name = model.names[top1_id]
            csv_rows.append([
                f"{source_name}_f{frame_idx}",
                top1_id,
                cls_name,
                f"{top1_conf:.2f}",
                "Video-Classification",
                device_str,
                str(out_vid_path.resolve()),
            ])

        if csv_rows:
            log_to_csv(csv_path, csv_rows)

        if not args.no_show:
            cv2.imshow("YOLO Stream", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[THÀNH CÔNG] Đã lưu video: {out_vid_path.resolve()}")


def run_inference(args):
    # Xác định phần cứng
    if args.device == "cpu":
        device_str = "CPU"
    elif args.device in ["0", "cuda"] or torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA"
        device_str = f"GPU CUDA ({gpu_name})"
    else:
        device_str = "CPU"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name

    print(f"--> Load Model: {args.model}")
    model = YOLO(args.model)

    # Bật FP16 trực tiếp lên đối tượng model (tránh warning deprecation)
    if args.half and "CUDA" in device_str:
        print("--> Bật chế độ FP16 (Half Precision) trên GPU CUDA")
        model.to("cuda").half()

    source = args.source

    if source.isdigit():
        print(f"--> Chế độ: WEBCAM (Index: {source})")
        process_video_or_webcam(model, int(source), args, out_dir, csv_path, device_str)

    elif os.path.isfile(source) and source.lower().endswith(IMG_FORMATS):
        print(f"--> Chế độ: SINGLE IMAGE ({source})")
        process_single_image(model, source, args, out_dir, csv_path, device_str)

    elif os.path.isfile(source) and source.lower().endswith(VID_FORMATS):
        print(f"--> Chế độ: VIDEO FILE ({source})")
        process_video_or_webcam(model, source, args, out_dir, csv_path, device_str)

    elif os.path.isdir(source):
        print(f"--> Chế độ: DIRECTORY ({source})")
        process_directory(model, source, args, out_dir, csv_path, device_str)

    else:
        print(f"[Lỗi] Đường dẫn không tồn tại hoặc file không hợp lệ: '{source}'")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)