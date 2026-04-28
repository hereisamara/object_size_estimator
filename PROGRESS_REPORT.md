# Progress Report: Real-Time Object Size Estimator

## 1. Our Mission
To create a **frictionless object measurement tool** for real-world office environments using standard smartphones.

## 2. Our Solution
Our primary objective is to **relieve the limiting conditions** currently required for mobile CV measurement. Specifically, we are solving for:
- **Reference-Free Estimation**: Removing the need for a credit card or coin to be in the frame.
- **Perspective Flexibility**: Removing the requirement for a strict top-down camera angle, making the tool usable for natural, tilted office shots.

This approach makes the technology significantly more usable for everyday office tasks where speed and ease-of-use are paramount.

---

## 3. Experimental Progress (POC Baselines)
Before building the final unconstrained system, we developed and verified two baseline **Proof-of-Concept (POC)** methods to establish accuracy limits and technical foundations.

### POC 1: Reference-Object Method
- **Method**: Automatic detection of a known reference (credit card) to calibrate pixels.
- **Verification Result**: **2.75% Mean Error**. Proven robust to texture and lighting noise.
- **Status**: Baseline established. Supports YAML-based custom reference inputs.

### POC 2: Depth-Based Method
- **Method**: Sizing based on fixed distance (e.g., 150cm) and camera intrinsic parameters.
- **Verification Result**: **0.1% Mean Error** (at 1.5m). Highly accurate in controlled environments.
- **Status**: Verified the mathematical limits of smartphone lens geometry.

---

## 4. Next Steps: Removing constraints
We are now moving toward Phase 3, which focuses on eliminating the human-input requirements (no card, no fixed distance):

1.  **AI Object Detection**: Using YOLO/SAM to identify objects on cluttered office desks without merging errors.
2.  **Depth via Triangulation**: Estimating 3D depth from multiple frames or motion (parallax), removing the need for a physical reference or a known fixed distance.
3.  **Perspective Correction**: Using homography and AI to de-warp images taken at an angle.
