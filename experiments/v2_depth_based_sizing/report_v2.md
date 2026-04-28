# Technical Report: POC Baseline v2 (Depth-Based)

## Overview
This document represents the **Phase 2 Proof of Concept (POC)** baseline. It evaluates the feasibility of sizing objects using purely camera parameters and distance, aiming to eventually eliminate the need for physical reference objects in the scene.

## Methodology
The core calculation follows the thin-lens projection model:
1. **Focal Length in Pixels ($f_p$):** 
   $$f_p = \frac{\text{focal\_length\_mm} \times \text{image\_width\_px}}{\text{sensor\_width\_mm}}$$
2. **Real World Size ($S$):**
   $$S = \frac{\text{pixel\_size} \times \text{distance\_from\_camera}}{f_p}$$

The v2 pipeline (implemented in `depth_estimator.py`) uses Otsu segmentation and oriented bounding boxes to measure any detected object in the frame.

## Results (Fixed Distance: 150 cm)
based on synthetic data
Testing was conducted using a simulated 26mm lens on a 6.3mm sensor (common smartphone specs) at a 1.5-meter distance (typical top-down shot height).

| Object | GT Size (mm) | Est Size (mm) | Error (%) |
| :--- | :--- | :--- | :--- |
| **Card** | 85.6 x 54.0 | 85.6 x 54.0 | **0.02%** |
| **Notebook** | 148.0 x 100.0| 147.8 x 100.0 | **0.07%** |
| **Phone** | 150.0 x 72.0 | 149.9 x 71.9 | **0.10%** |
| **SD Card** | 32.0 x 24.0 | 32.0 x 23.9 | **0.21%** |

**Mean Accuracy: 99.9% (Error: 0.10%)**

## Comparison with Reference-Based Method (v1)
| Feature | v1 (Reference Card) | v2 (Depth-Based) |
| :--- | :--- | :--- |
| **Ease of Use** | High (just find a card) | Low (must know distance/camera) |
| **Hardware Dependency**| None | High (intrinsic parameters required) |
| **Accuracy** | High (~2-5%) | Extremely High (<1% if distance is exact) |
| **Robustness** | Robust to distance change | Very sensitive to distance errors |

## Key Insights
- **Distance as the Bottleneck**: While the math is perfect, a 1cm error in distance estimation at 50cm would lead to a 2% error in size.
- **Intrinsic Precision**: Knowing the exact sensor width/focal length from EXIF data or calibration is critical for sub-1% accuracy.
- **Top-Down Assumption**: Both methods currently assume a planar, top-down view. Angular tilt will introduce cosine errors in both cases.


- need to work on calibration
