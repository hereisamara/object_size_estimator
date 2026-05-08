# Object Size Estimator — Depth-Based Method (v2)

This experiment uses a fixed distance and calibrated camera intrinsics to measure objects without needing a reference card.

## 🚀 Setup & Usage

### 1. Requirements
Ensure you have the dependencies from v1:

```bash
pip install numpy opencv-python scipy
```

### 2. Calibrate the camera first
Capture 10-20 checkerboard images from the same phone camera you plan to use. Vary angle, position, and distance. A printed `9 x 6` inner-corner board works well.

```bash
python calibrate_camera.py \
  --images "calibration_images/*.jpg" \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-mm 24.0 \
  --output calibration/iphone_main_cam.json
```

This saves:
- `camera_matrix` and `dist_coeffs`
- reprojection error metrics
- debug images with detected corners

### 3. Run the experiment with calibration
Specify the object distance in cm and pass the saved calibration file.

```bash
python depth_estimator.py \
  --image test_images/depth_test_150cm.png \
  --distance 150 \
  --calibration calibration/iphone_main_cam.json \
  --gt test_images/depth_test_150cm_gt.json
```

### 4. Legacy fallback
If you do not have a calibration file yet, the older focal-length approximation still works:

```bash
python depth_estimator.py --image test_images/depth_test_150cm.png --distance 150 --focal 26 --sensor-w 6.3
```

## 📐 The Math
The system uses the pinhole camera model after undistortion:
`Size = (Pixels * Distance) / Focal_Pixels`

When calibration is provided, `Focal_Pixels` comes from the estimated intrinsic matrix instead of a phone spec-sheet approximation.

## 📊 Results Summary
At a controlled 1.5m distance, the system achieved **99.9% accuracy** on synthetic objects.
Check [report_v2.md](report_v2.md) for full details.
