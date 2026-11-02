import argparse
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Supported files
IMG_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VID_FORMATS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")

CSV_HEADER = [
    "Filename",
    "Class_ID",
    "Class_Name",
    "Confidence (%)",
    "X1",
    "Y1",
    "X2",
    "Y2",
    "Inference_ms",
    "Type",
    "Device",
    "Output_Path",
]

COL_NAME, COL_CONF, COL_MS, COL_TYPE, COL_DEVICE = 2, 3, 8, 9, 10


MAX_WIN_W, MAX_WIN_H = 1280, 720



# Arguments

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO inference on image / folder / video / webcam")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to model weights")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image file, folder of images, video file, or webcam index (e.g. 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="'cpu', '0', '1', 'cuda:0'... Empty = auto (GPU if available)",
    )
    parser.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size when processing a folder")
    parser.add_argument("--out-dir", type=str, default="output", help="Root output folder")
    parser.add_argument("--csv-name", type=str, default="results.csv")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-show", action="store_true", help="Do not open preview windows")
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not log images/frames with no detection (applies to images AND video)",
    )
    return parser.parse_args()

# Helpers
def resolve_device(arg):
    """Return (device for YOLO, display string, is_cuda). Falls back to CPU safely."""
    arg = (arg or "").strip().lower()

    if arg == "cpu":
        return "cpu", "CPU", False

    if not torch.cuda.is_available():
        if arg:
            print(f"[Warning] CUDA is not available, falling back to CPU (requested: '{arg}')")
        return "cpu", "CPU", False

    idx_str = arg.replace("cuda", "").replace(":", "")
    idx = int(idx_str) if idx_str.isdigit() else 0
    if idx >= torch.cuda.device_count():
        print(f"[Warning] GPU {idx} not found, using GPU 0")
        idx = 0

    return str(idx), f"GPU CUDA ({torch.cuda.get_device_name(idx)})", True


def save_image(path, img):
    """cv2.imwrite replacement that works with Unicode paths on Windows."""
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    buf.tofile(str(path))


_open_windows = set()


def show_window(name, img, wait_ms):
    """Show `img` in a resizable window scaled to fit the screen. Returns the pressed key (0-255)."""
    if name not in _open_windows:
        h, w = img.shape[:2]
        scale = min(MAX_WIN_W / w, MAX_WIN_H / h, 1.0)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, max(1, int(w * scale)), max(1, int(h * scale)))
        _open_windows.add(name)
    cv2.imshow(name, img)
    return cv2.waitKey(wait_ms) & 0xFF


def close_windows():
    cv2.destroyAllWindows()
    _open_windows.clear()


#CSV
class CsvLogger:
    """Opens the CSV once per run and keeps it open."""

    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "w", newline="", encoding="utf-8-sig")
        self._w = csv.writer(self._f)
        self._w.writerow(CSV_HEADER)

    def write(self, rows):
        if rows:
            self._w.writerows(rows)

    def flush(self):
        self._f.flush()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_rows(res, label, device_str, out_path, prefix=""):
    """Convert one YOLO result into CSV rows.

    Handles detect / segment / pose (res.boxes), OBB (res.obb) and classify (res.probs).
    """
    out = str(Path(out_path).resolve())
    names = res.names
    ms = f"{float(res.speed.get('inference', 0.0)):.1f}"

    dets = None
    if res.boxes is not None and len(res.boxes) > 0:
        dets = res.boxes
    elif getattr(res, "obb", None) is not None and len(res.obb) > 0:
        dets = res.obb

    rows = []

    if dets is not None:
        for cls, conf, box in zip(dets.cls.tolist(), dets.conf.tolist(), dets.xyxy.tolist()):
            cid = int(cls)
            x1, y1, x2, y2 = (f"{v:.1f}" for v in box)
            rows.append(
                [label, cid, names[cid], f"{conf * 100:.2f}", x1, y1, x2, y2, ms,
                 f"{prefix}Detection", device_str, out]
            )

    elif res.probs is not None:
        cid = int(res.probs.top1)
        conf = float(res.probs.top1conf) * 100
        rows.append(
            [label, cid, names[cid], f"{conf:.2f}", "", "", "", "", ms,
             f"{prefix}Classification", device_str, out]
        )

    else:
        rows.append(
            [label, "N/A", "None", "0.00", "", "", "", "", ms,
             f"{prefix}No Detection", device_str, out]
        )

    return rows


def is_empty(rows):
    return rows[0][COL_TYPE].endswith("No Detection")


def print_rows(rows):
    for r in rows:
        if r[COL_TYPE].endswith("No Detection"):
            print("--> No detections.")
        else:
            print(f"--> {r[COL_NAME]} | Conf: {r[COL_CONF]}% | {r[COL_MS]} ms | Device: {r[COL_DEVICE]}")


#Stats
class Stats:
    """Collects numbers for the summary printed at the end of a run."""

    def __init__(self):
        self.n = 0
        self.with_det = 0
        self.objects = 0
        self.per_class = Counter()
        self.infer_ms = 0.0

    def update(self, res, rows):
        self.n += 1
        self.infer_ms += float(res.speed.get("inference", 0.0))
        if not is_empty(rows):
            self.with_det += 1
            self.objects += len(rows)
            for r in rows:
                self.per_class[r[COL_NAME]] += 1

    def report(self, unit):
        print("\n========== Summary ==========")
        if self.n == 0:
            print("Nothing was processed.")
            return
        avg = self.infer_ms / self.n
        print(f"Processed {self.n} {unit}(s), {self.with_det} with detections ({self.with_det / self.n * 100:.1f}%)")
        print(f"Total objects: {self.objects}")
        for name, count in self.per_class.most_common():
            print(f"  - {name}: {count}")
        if avg > 0:
            print(f"Avg inference: {avg:.1f} ms/{unit} (~{1000 / avg:.1f} FPS)")


# Image / folder
def process_result(res, rel_path, args, run_dir, logger, stats, device_str, block):
    """Annotate, save, log and (optionally) preview one image result.

    Returns False when the user pressed 'q' (stop), True otherwise.
    `rel_path` keeps the sub-folder structure so output names never collide.
    """
    out_img_path = run_dir / rel_path.parent / f"out_{rel_path.name}"
    out_img_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        annotated = res.plot()
        save_image(out_img_path, annotated)
    except Exception as e:
        print(f"[Error] Failed on {rel_path}: {e}")
        return True

    rows = extract_rows(res, rel_path.as_posix(), device_str, out_img_path)
    print_rows(rows)
    stats.update(res, rows)

    if not (args.skip_empty and is_empty(rows)):
        logger.write(rows)
    logger.flush()

    print(f"[Success] Output image: {out_img_path.resolve()}")

    if args.no_show:
        return True

    if block:
        print("[Info] Press any key in the image window to close")
    key = show_window("YOLO Result", annotated, 0 if block else 1)
    return key != ord("q")


def process_image(model, img_path, predict_kw, args, run_dir, logger, stats, device_str):
    try:
        # Passing the path (not a cv2-loaded array) lets Ultralytics read Unicode paths safely.
        res = model.predict(str(img_path), **predict_kw)[0]
    except Exception as e:  # corrupt image, unreadable file, ...
        print(f"[Error] Failed on {img_path}: {e}")
        return

    process_result(res, Path(img_path.name), args, run_dir, logger, stats, device_str, block=True)
    if not args.no_show:
        close_windows()


def process_directory(model, dir_path, predict_kw, args, run_dir, logger, stats, device_str):
    dir_path = Path(dir_path)
    run_root = run_dir.parent.resolve()

    img_files = sorted(
        p
        for p in dir_path.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMG_FORMATS
        and run_root not in p.resolve().parents  # don't re-process our own outputs
    )

    if not img_files:
        print(f"[Warning] No valid image files were found in: {dir_path}")
        return

    rel_map = {str(p.resolve()): p.relative_to(dir_path) for p in img_files}

    print(f"--> Found {len(img_files)} image(s). Running on [{device_str}], batch={args.batch}...")
    if not args.no_show:
        print("[Info] Press 'q' in the preview window to stop")

    try:
        results = model.predict([str(p) for p in img_files], stream=True, batch=args.batch, **predict_kw)
        for idx, res in enumerate(results, 1):
            res_path = Path(res.path)
            rel = rel_map.get(str(res_path.resolve()), Path(res_path.name))
            print(f"\n[{idx}/{len(img_files)}] Processing: {rel}")
            if not process_result(res, rel, args, run_dir, logger, stats, device_str, block=False):
                print("[Info] Stopped by user")
                break
    finally:
        if not args.no_show:
            close_windows()


# Video / webcam
def process_video_or_webcam(model, source, predict_kw, args, run_dir, logger, stats, device_str):
    """Process a video file or a webcam (source is an int index)."""
    is_cam = isinstance(source, int)

    cap = cv2.VideoCapture(source if is_cam else str(source))
    if not cap.isOpened():
        print(f"[Error] Unable to open Video/Camera: {source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    fps = fps if fps and fps > 0 else 30

    source_name = "webcam" if is_cam else Path(source).name
    out_vid_path = run_dir / ("out_webcam.mp4" if is_cam else f"out_{Path(source).stem}.mp4")

    print(f"--> Running stream on [{device_str}], saving video to: {out_vid_path.resolve()}")
    if not args.no_show:
        print("[Info] Press 'q' in the preview window to stop")

    writer = None
    frame_idx = 0

    try:
        # stream=True -> generator, frames are processed one by one (low memory)
        for res in model.predict(source=source if is_cam else str(source), stream=True, **predict_kw):
            frame_idx += 1
            annotated = res.plot()

            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out_vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)

            rows = extract_rows(
                res, f"{source_name}_f{frame_idx}", device_str, out_vid_path, prefix="Video-"
            )
            stats.update(res, rows)
            if not (args.skip_empty and is_empty(rows)):
                logger.write(rows)
            if frame_idx % 50 == 0:
                logger.flush()

            if not args.no_show:
                if show_window("YOLO Stream", annotated, 1) == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
        logger.flush()
        if not args.no_show:
            close_windows()

    print(f"[Success] Processed {frame_idx} frame(s). Saved video: {out_vid_path.resolve()}")



# Main#
def run_inference(args):
    # 1. Work out what kind of source we have BEFORE loading the model
    source = args.source
    src_path = Path(source)

    if source.isdigit():
        mode = "webcam"
    elif src_path.is_file() and src_path.suffix.lower() in IMG_FORMATS:
        mode = "image"
    elif src_path.is_file() and src_path.suffix.lower() in VID_FORMATS:
        mode = "video"
    elif src_path.is_dir():
        mode = "directory"
    else:
        print(f"[Error] The path does not exist or the file is invalid: '{source}'")
        sys.exit(1)

    # 2. Hardware
    device, device_str, use_cuda = resolve_device(args.device)
    half = args.half and use_cuda
    if args.half and not use_cuda:
        print("[Warning] --half is only supported on GPU, ignoring it")

    # 3. Output
    run_dir = Path(args.out_dir) / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / args.csv_name

    # 4. Model
    print(f"--> Load Model: {args.model}")
    model = YOLO(args.model)

    predict_kw = dict(conf=args.conf, device=device, quantize=16 if half else None, verbose=False)
    if half:
        print("--> FP16 (Half Precision) enabled")

    # 5. Run
    stats = Stats()
    unit = "image"

    with CsvLogger(csv_path) as logger:
        if mode == "webcam":
            unit = "frame"
            print(f"--> Mode: WEBCAM (Index: {source})")
            process_video_or_webcam(model, int(source), predict_kw, args, run_dir, logger, stats, device_str)

        elif mode == "image":
            print(f"--> Mode: SINGLE IMAGE ({source})")
            process_image(model, src_path, predict_kw, args, run_dir, logger, stats, device_str)

        elif mode == "video":
            unit = "frame"
            print(f"--> Mode: VIDEO FILE ({source})")
            process_video_or_webcam(model, source, predict_kw, args, run_dir, logger, stats, device_str)

        else:
            print(f"--> Mode: DIRECTORY ({source})")
            process_directory(model, src_path, predict_kw, args, run_dir, logger, stats, device_str)

    stats.report(unit)
    print(f"\n[Success] Saved log CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    run_inference(parse_args())
