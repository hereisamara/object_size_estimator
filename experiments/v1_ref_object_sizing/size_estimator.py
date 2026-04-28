"""
Real-Time Object Size Estimator v1.1 — Modular Reference Object Estimator

Refactored into a class-based "plugin-able" system with YAML configuration support.
"""

import argparse
import os
import csv
import json
import yaml
import cv2
import numpy as np
from scipy.spatial import distance as dist


class ReferenceSizeEstimator:
    def __init__(self, blur_kernel=7, min_area_ratio=0.003):
        self.blur_kernel = blur_kernel
        self.min_area_ratio = min_area_ratio
        self.ppm = None  # Pixels per Millimeter

    @staticmethod
    def order_points(pts):
        """Order 4 corners: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    @staticmethod
    def contour_rectangularity(c):
        """Area(contour) / Area(minAreaRect)."""
        area = cv2.contourArea(c)
        if area < 1: return 0
        _, (rw, rh), _ = cv2.minAreaRect(c)
        return min(area / (rw * rh), 1.0) if rw * rh > 0 else 0

    @staticmethod
    def contour_aspect_ratio(c):
        """Returns (longer/shorter) aspect ratio of min-area rect."""
        _, (rw, rh), _ = cv2.minAreaRect(c)
        if rw < 1 or rh < 1: return 0
        return max(rw, rh) / min(rw, rh)

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        blurred = cv2.GaussianBlur(filtered, (self.blur_kernel, self.blur_kernel), 0)
        return blurred

    def segment(self, blurred):
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.sum(binary == 255) / binary.size > 0.5:
            binary = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def find_contours(self, binary):
        image_area = binary.shape[0] * binary.shape[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        for c in contours:
            if image_area * self.min_area_ratio <= cv2.contourArea(c) <= image_area * 0.5:
                if self.contour_rectangularity(c) > 0.4:
                    valid.append(c)
        valid.sort(key=lambda c: (cv2.boundingRect(c)[0], cv2.boundingRect(c)[1]))
        return valid

    def calibrate(self, contours, ref_w_mm, ref_h_mm=None, ref_index=None):
        if ref_index is not None:
            best_idx = ref_index
        else:
            expected_aspect = ref_w_mm / ref_h_mm if ref_h_mm else 1.586
            best_idx, best_score = 0, -1
            for i, c in enumerate(contours[:5]):
                score = self.contour_rectangularity(c) * max(0, 1 - abs(self.contour_aspect_ratio(c) - expected_aspect)/expected_aspect)
                if score > best_score:
                    best_score, best_idx = score, i

        rect = cv2.minAreaRect(contours[best_idx])
        box = self.order_points(cv2.boxPoints(rect))
        long_px = max(dist.euclidean(box[0], box[1]), dist.euclidean(box[1], box[2]))
        self.ppm = long_px / ref_w_mm
        return self.ppm, best_idx

    def measure(self, image, contours, ref_idx):
        output = image.copy()
        measurements = []
        for i, c in enumerate(contours):
            is_ref = (i == ref_idx)
            rect = cv2.minAreaRect(c)
            box = self.order_points(cv2.boxPoints(rect))
            w_px = dist.euclidean(box[0], box[1])
            h_px = dist.euclidean(box[1], box[2])
            w_mm, h_mm = w_px / self.ppm, h_px / self.ppm
            if w_mm < h_mm: w_mm, h_mm = h_mm, w_mm
            
            label = "REFERENCE" if is_ref else f"Obj_{i+1}"
            measurements.append({"label": label, "w_mm": round(w_mm, 1), "h_mm": round(h_mm, 1), "is_ref": is_ref})
            
            color = (0, 255, 0) if is_ref else (255, 150, 50)
            cv2.drawContours(output, [np.int32(box)], -1, color, 3)
            cv2.putText(output, f"{w_mm:.1f}x{h_mm:.1f}mm", (int(box[0][0]), int(box[0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return output, measurements


def load_config(path, ref_id):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    if ref_id not in config:
        raise ValueError(f"Reference ID '{ref_id}' not found in {path}")
    return config[ref_id]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--image", required=True)
    p.add_argument("-c", "--config", default="ref_objects.yaml")
    p.add_argument("-r", "--ref-id", default="credit_card")
    args = p.parse_args()

    ref_data = load_config(args.config, args.ref_id)
    img = cv2.imread(args.image)
    
    estimator = ReferenceSizeEstimator()
    mask = estimator.segment(estimator.preprocess(img))
    contours = estimator.find_contours(mask)
    
    if not contours:
        print("No objects found.")
        return

    ppm, ref_idx = estimator.calibrate(contours, ref_data['width_mm'], ref_data.get('height_mm'))
    output, results = estimator.measure(img, contours, ref_idx)
    
    cv2.imwrite("result_modular.png", output)
    print(f"Calibration: {ppm:.3f} px/mm")
    for r in results:
        print(f"{r['label']}: {r['w_mm']} x {r['h_mm']} mm")


if __name__ == "__main__":
    main()
