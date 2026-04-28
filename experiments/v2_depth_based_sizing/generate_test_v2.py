"""
Generate synthetic images for depth-based sizing experiments (v1.5 meter).
Fixed: Maximum separation to ensure no merges.
"""

import cv2
import numpy as np
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FOCAL_MM = 26.0
SENSOR_W_MM = 6.3
IMAGE_W = 1920 
IMAGE_H = 1080
DISTANCE_CM = 150.0 

# Calculations
F_PX = (FOCAL_MM * IMAGE_W) / SENSOR_W_MM
PX_PER_MM = F_PX / (DISTANCE_CM * 10.0)

def mm_to_px(mm_val):
    return int(mm_val * PX_PER_MM)

def generate_test_v2():
    img = np.ones((IMAGE_H, IMAGE_W, 3), dtype=np.uint8) * 235
    gt = {}

    # 1. Notebook (148 x 100 mm) - TOP LEFT
    w1, h1 = 148.0, 100.0
    x1, y1 = 100, 100
    cv2.rectangle(img, (x1, y1), (x1 + mm_to_px(w1), y1 + mm_to_px(h1)), (45, 80, 45), -1)
    gt["notebook"] = {"w_mm": w1, "h_mm": h1}

    # 2. Card (85.6 x 54 mm) - TOP RIGHT
    w2, h2 = 85.6, 53.98
    x2, y2 = 1400, 100
    cv2.rectangle(img, (x2, y2), (x2 + mm_to_px(w2), y2 + mm_to_px(h2)), (50, 50, 100), -1)
    gt["card"] = {"w_mm": w2, "h_mm": h2}

    # 3. Phone (150 x 72 mm) - BOTTOM LEFT
    w3, h3 = 150.0, 72.0
    x3, y3 = 100, 650
    cv2.rectangle(img, (x3, y3), (x3 + mm_to_px(w3), y3 + mm_to_px(h3)), (30, 30, 30), -1)
    gt["phone"] = {"w_mm": w3, "h_mm": h3}

    # 4. SD Card (32 x 24 mm) - BOTTOM RIGHT
    w4, h4 = 32.0, 24.0
    x4, y4 = 1600, 800
    cv2.rectangle(img, (x4, y4), (x4 + mm_to_px(w4), y4 + mm_to_px(h4)), (100, 60, 60), -1)
    gt["sd_card"] = {"w_mm": w4, "h_mm": h4}

    # Save
    img_path = os.path.join(OUTPUT_DIR, "depth_test_150cm.png")
    gt_path = os.path.join(OUTPUT_DIR, "depth_test_150cm_gt.json")
    cv2.imwrite(img_path, img)
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    
    print(f"[✓] Final depth test image (1.5m) saved.")

if __name__ == "__main__":
    generate_test_v2()
