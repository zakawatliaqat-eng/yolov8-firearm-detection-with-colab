# YOLOv8 Firearm Detection (Real-Time)

[![Python CI](https://github.com/zakawatliaqat-eng/yolov8-firearm-detection/actions/workflows/python-ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/yolov8-firearm-detection/actions)

## Overview
This repository provides a **YOLOv8-based firearm detection system** for real-time surveillance applications.  
Built with **PyTorch + Ultralytics YOLOv8**, it supports:

- Training on custom datasets (COCO / YOLO format)
- Real-time inference (webcam, RTSP, video files)
- Evaluation (mAP, precision, recall)
- Export to **ONNX / TorchScript** for edge deployment
- Google Colab notebook for quick testing & prototyping

>  **Ethics Disclaimer**: This project is for research and educational purposes.  
> Use responsibly and comply with all applicable laws, privacy, and surveillance policies.



##  Features
- Easy-to-run training pipeline (`src/train.py`)
- Real-time detection with bounding boxes & FPS display
- Single-image inference and batch evaluation
- Pre-configured CI workflow for GitHub
- Colab notebook for hands-on experiments
- Modular structure for deployment in production



##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/yolov8-firearm-detection.git
   cd yolov8-firearm-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```



##  Repository Structure
```
yolov8-firearm-detection/
├─ src/
│  ├─ train.py          # Training script
│  ├─ infer_realtime.py # Real-time inference (webcam/RTSP)
│  ├─ infer_image.py    # Single image inference
│  ├─ export.py         # Export to ONNX / TorchScript
│  ├─ evaluate.py       # Evaluate on validation set
│  └─ utils.py          # Drawing & helper functions
├─ notebooks/
│  └─ Train_YOLOv8_Firearm.ipynb  # Colab training notebook
├─ data.yaml            # Dataset config
├─ requirements.txt     # Dependencies
├─ README.md            # Project documentation
├─ LICENSE              # MIT License
└─ .github/workflows/   # CI pipeline
```



##  Dataset Preparation

This project expects datasets in **YOLO format**:
- Each image has a `.txt` label file with:  
  `class x_center y_center width height` (all normalized 0–1).
- Example `data.yaml`:
  ```yaml
  train: ./data/train/images
  val: ./data/val/images
  nc: 1
  names: ['firearm']
  ```

### Public datasets
- [Kaggle Gun Detection datasets](https://www.kaggle.com/)
- [Open Images V6 subset (firearms)](https://storage.googleapis.com/openimages/web/index.html)

 **Do not** upload private/sensitive CCTV footage to public repos. Always anonymize faces if required.



##  Usage

### Training
```bash
python src/train.py --data data.yaml --model yolov8n.pt --epochs 50 --batch 16
```

### Real-Time Inference
```bash
# Webcam
python src/infer_realtime.py --source 0 --weights runs/train/yolov8-firearm/weights/best.pt

# RTSP Stream
python src/infer_realtime.py --source "rtsp://user:pass@ip:554/stream" --weights runs/train/yolov8-firearm/weights/best.pt
```

### Single Image Inference
```bash
python src/infer_image.py --img data/sample.jpg --weights runs/train/.../best.pt
```

### Export Model
```bash
python src/export.py --weights runs/train/.../best.pt --format onnx
```

### Evaluation
```bash
python src/evaluate.py --weights runs/train/.../best.pt --data data.yaml
```



##  Google Colab Quickstart
Try the ready-to-use Colab notebook for training & validation:  

 [Open in Colab](https://colab.research.google.com/github/YOUR_USERNAME/yolov8-firearm-detection/blob/main/notebooks/Train_YOLOv8_Firearm.ipynb)



##  Results
- Metrics reported: **mAP@0.5**, **mAP@0.5:0.95**, **Precision**, **Recall**
- Sample outputs will appear in `runs/` after training/inference.



##  Ethics & Legal Considerations
- This project detects **objects (firearms)**, not people.  
- Use responsibly as an **assistive tool**, not an autonomous decision-maker.
- **False positives** and **false negatives** can have serious consequences. Always keep a human-in-the-loop.
- Respect local laws on CCTV, privacy, and surveillance.



##  Contributing
Contributions are welcome! Please:
1. Fork the repo
2. Create a new branch (`feature/your-feature`)
3. Commit changes
4. Open a Pull Request



##  License
This project is licensed under the [MIT License](LICENSE).


