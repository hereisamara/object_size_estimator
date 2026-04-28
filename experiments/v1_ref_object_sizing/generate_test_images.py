"""
Generate synthetic test images with known ground-truth dimensions.
Objects are well-separated on a contrasting background.
"""

import cv2
import numpy as np
import json
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PX_PER_MM = 5.0
CANVAS_W = int(400 * PX_PER_MM)   # 2000 px — wider to avoid overlaps
CANVAS_H = int(300 * PX_PER_MM)   # 1500 px

CARD_W_MM = 85.6
CARD_H_MM = 53.98

def mm(val):
    return int(val * PX_PER_MM)


def save(img, gt, name):
    img_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    gt_path  = os.path.join(OUTPUT_DIR, f"{name}_gt.json")
    cv2.imwrite(img_path, img)
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[✓] {name}: {img_path}")


def generate_basic_test():
    """Test 1: Well-separated axis-aligned rectangles on white bg."""
    img = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 240
    gt = {}

    # Reference card — top-left
    cx, cy = 150, 150
    cv2.rectangle(img, (cx, cy), (cx + mm(CARD_W_MM), cy + mm(CARD_H_MM)), (60, 30, 15), -1)
    gt["reference_card"] = {"w_mm": CARD_W_MM, "h_mm": CARD_H_MM}

    # Phone (150×72) — top-right, well separated
    w1, h1 = 150.0, 72.0
    x1, y1 = 1100, 150
    cv2.rectangle(img, (x1, y1), (x1 + mm(w1), y1 + mm(h1)), (35, 35, 35), -1)
    gt["phone_shape"] = {"w_mm": w1, "h_mm": h1}

    # Post-it (76×76) — middle-left
    s2 = 76.0
    x2, y2 = 200, 700
    cv2.rectangle(img, (x2, y2), (x2 + mm(s2), y2 + mm(s2)), (0, 150, 150), -1)
    gt["postit_note"] = {"w_mm": s2, "h_mm": s2}

    # Book (230×150) — bottom-right
    w3, h3 = 230.0, 150.0
    x3, y3 = 1000, 800
    cv2.rectangle(img, (x3, y3), (x3 + mm(w3), y3 + mm(h3)), (25, 70, 25), -1)
    gt["book_shape"] = {"w_mm": w3, "h_mm": h3}

    # SD card (32×24) — middle area
    w4, h4 = 32.0, 24.0
    x4, y4 = 700, 400
    cv2.rectangle(img, (x4, y4), (x4 + mm(w4), y4 + mm(h4)), (50, 50, 100), -1)
    gt["sd_card"] = {"w_mm": w4, "h_mm": h4}

    save(img, gt, "synthetic_test")


def generate_rotated_test():
    """Test 2: Reference is axis-aligned, other objects rotated."""
    img = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 235
    gt = {}

    # Reference card — axis-aligned, top-left
    cx, cy = 150, 150
    cv2.rectangle(img, (cx, cy), (cx + mm(CARD_W_MM), cy + mm(CARD_H_MM)), (60, 30, 15), -1)
    gt["reference_card"] = {"w_mm": CARD_W_MM, "h_mm": CARD_H_MM}

    # Rotated phone (150×72, 25°) — right side
    w1, h1 = 150.0, 72.0
    pts1 = cv2.boxPoints(((1300, 350), (mm(w1), mm(h1)), 25))
    cv2.fillPoly(img, [np.int32(pts1)], (35, 35, 35))
    gt["phone_rotated_25deg"] = {"w_mm": w1, "h_mm": h1}

    # Rotated post-it (76×76, 45°) — bottom-left
    s2 = 76.0
    pts2 = cv2.boxPoints(((400, 1000), (mm(s2), mm(s2)), 45))
    cv2.fillPoly(img, [np.int32(pts2)], (0, 150, 150))
    gt["postit_rotated_45deg"] = {"w_mm": s2, "h_mm": s2}

    # Rotated book (230×150, -15°) — bottom-right
    w3, h3 = 230.0, 150.0
    pts3 = cv2.boxPoints(((1300, 1000), (mm(w3), mm(h3)), -15))
    cv2.fillPoly(img, [np.int32(pts3)], (25, 70, 25))
    gt["book_rotated_neg15deg"] = {"w_mm": w3, "h_mm": h3}

    save(img, gt, "synthetic_rotated")


def generate_noisy_test():
    """Test 3: Noise + shadow gradient."""
    img = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 230
    gt = {}

    # Reference card
    cx, cy = 150, 150
    cv2.rectangle(img, (cx, cy), (cx + mm(CARD_W_MM), cy + mm(CARD_H_MM)), (60, 30, 15), -1)
    gt["reference_card"] = {"w_mm": CARD_W_MM, "h_mm": CARD_H_MM}

    objects = [
        ("usb_stick",  60.0, 18.0, 1100, 200, (40, 40, 40)),
        ("eraser",     55.0, 22.0,  250, 700, (130, 50, 50)),
        ("notebook",  210.0, 148.0, 900, 750, (35, 80, 35)),
    ]
    for name, w, h, x, y, color in objects:
        cv2.rectangle(img, (x, y), (x + mm(w), y + mm(h)), color, -1)
        gt[name] = {"w_mm": w, "h_mm": h}

    # Gaussian noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Shadow gradient
    shadow = np.linspace(1.0, 0.90, CANVAS_W).reshape(1, -1, 1)
    img = np.clip(img * np.broadcast_to(shadow, img.shape), 0, 255).astype(np.uint8)

    save(img, gt, "synthetic_noisy")


if __name__ == "__main__":
    print("=== Generating Synthetic Test Images ===\n")
    generate_basic_test()
    generate_rotated_test()
    generate_noisy_test()
    print("\nDone.")
