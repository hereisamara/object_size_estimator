# Object Size Estimator — Depth-Based Method (v2)

This experiment uses a fixed distance and camera parameters to measure objects without needing a reference card.

## 🚀 Setup & Usage

### 1. Requirements
Ensure you have the dependencies from v1:
```bash
pip install numpy opencv-python scipy
```

### 2. Run the experiment
Specify the distance (in cm) and your camera's focal length (mm). 
*Note: iPhone 13/14/15 main camera is usually 26mm focal length and ~6mm sensor width.*

```bash
python depth_estimator.py --image test_images/depth_test_150cm.png --distance 150 --focal 26 --sensor-w 6.3
```

## 📐 The Math
The system calculates size using the pinhole camera model:
`Size = (Pixels * Distance) / Focal_Pixels`

## 📊 Results Summary
At a controlled 1.5m distance, the system achieved **99.9% accuracy** on synthetic objects.
Check [report_v2.md](report_v2.md) for full details.
