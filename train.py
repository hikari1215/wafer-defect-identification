from ultralytics import YOLO

if __name__ == '__main__':
    # Load model base
    model = YOLO('yolov8n.pt')
    
    # Bắt đầu train
    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        project='runs/train',
        name='wafer_exp'
    )