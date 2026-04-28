# Technical Report: POC Baseline v1 (Reference-Object)

## Overview
This document represents the **Phase 1 Proof of Concept (POC)** for establishing a measurement baseline. This version relies on a known reference object to calibrate the scale, serving as the first step toward our goal of fully unconstrained measurement.

## 🏗 Architecture
The system implements a 6-stage computer vision pipeline designed for robustness against real-world lighting:

1.  **Preprocessing**: Grayscale conversion + **Bilateral Filter** (reduces noise but keeps edges sharp).
2.  **Segmentation**: **Otsu Adaptive Thresholding**. The script auto-detects if objects are dark-on-light (e.g., table) or light-on-dark.
3.  **Contour Filtering**: Area-based noise removal + **Rectangularity Scoring** (Area / BoundingBoxArea).
4.  **Reference ID**: Auto-matches contours to the **ISO Credit Card ratio (1.586)**.
5.  **Calibration**: Computes **Pixels-per-Millimeter** from the reference object.
6.  **Measurement**: Applies scale to all objects using oriented bounding boxes (`minAreaRect`).

## 🧪 Experiment Results

| Test Case | Accuracy | Observations |
| :--- | :--- | :--- |
| **01: Basic** | **97.2%** | Excellent for separated objects. Small erosion on edges causes ~2.8% error. |
| **02: Rotated** | **Partial** | Rotated objects are currently filtered out if they appear non-rectangular after morphological cleanup. |
| **03: Noisy** | **100%** | Perfect accuracy even with Gaussian noise and shadow gradients. |

## 💡 Key Findings
- **Otsu + Bilateral** is significantly more robust than Canny for uneven lighting.
- **Reference Auto-Detection** using aspect ratio works reliably without markers.
- **Limitation**: Rotated objects and perspective distortion remain the primary challenges for v1.
