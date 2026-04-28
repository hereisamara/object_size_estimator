# Real-Time Object Size Estimator v1 — Reference Object Method

## Overview

Estimate real-world object dimensions from a **single image** using a **known reference object** (e.g., credit card, coin).

**No ArUco markers. No fixed camera. No special setup.**

Take a photo with any phone or camera, include a reference object of known size in the frame, and the script measures everything else.

## Why This Approach?

| Problem with current methods | Our solution |
|---|---|
| Requires fixed camera setup | Works with any camera / phone |
| Needs ArUco markers | Uses everyday objects as reference |
| Fixed scenario only | Any flat surface, any angle |
| Poor detection accuracy | Adaptive preprocessing + contour filtering |
| High error rate | Pixel-per-metric calibration from reference |

## Pipeline

```
Image → Preprocess → Edge Detect → Find Contours → Identify Reference → Compute Scale → Measure All Objects
```

## Quick Start

```bash
# Activate virtualenv
source .venv/bin/activate

# Run on an image (credit card width = 85.6 mm)
python size_estimator.py --image test_images/synthetic_test.png --ref-width 85.6

# Run with a coin (US quarter diameter = 24.26 mm)
python size_estimator.py --image photo.jpg --ref-width 24.26

# Adjust preprocessing
python size_estimator.py --image photo.jpg --ref-width 85.6 --blur 7 --canny-low 30 --canny-high 120
```

## Output

- Annotated image saved to `results/` with bounding boxes and dimensions in mm
- Console printout of all detected object sizes
- Summary CSV with measurements

## Limitations (v1)

- Objects must be on a **contrasting background** (e.g., dark objects on white surface)
- Works best for **flat objects** photographed from above
- Reference object must be **fully visible** and **rectangular**
- Perspective distortion reduces accuracy for non-top-down shots
