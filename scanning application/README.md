### 🛠️ Prerequisites
Ensure you have Python 3.8+ installed along with the core computer vision stack:

```bash
pip install torch torchvision ultralytics opencv-python
```

---

###  Running Inference (`demo.py`)

Clone or download the repository, open your terminal in the project directory, and execute the pipeline based on your target input source:

####  1. Single Image Processing
```bash
python demo.py --model best.pt --source test.png --device 0
```

####  2. Video File Processing
```bash
python demo.py --model best.pt --source test.mp4 --device 0
```

#### 3. Real-Time Webcam / Stream
```bash
python demo.py --model best.pt --source 0 --device 0
```

---

###  CLI Arguments Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `str` | `best.pt` | Path to weights (`.pt`, `.engine`, `.onnx`) |
| `--source` | `str` | *Required* | Image/Video path, directory, or camera index (`0`) |
| `--device` | `str` | `0` | Hardware device (`0` for CUDA GPU, `cpu` for CPU) |
| `--conf` | `float` | `0.25` | Confidence threshold |
| `--half` | `flag` | `False` | Enable FP16 half-precision on CUDA |
| `--no-show` | `flag` | `False` | Disable GUI preview for pure FPS benchmarking |
