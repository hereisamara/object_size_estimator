"""
Object Size Estimation using Fixed Distance (Depth) and Camera Intrinsic Parameters.
v1.2 - Improved robustness and border handling.
"""

import argparse
import os
import cv2
import json
import numpy as np
from scipy.spatial import distance as dist


class DepthSizeEstimator:
    def __init__(self, focal_mm, sensor_width_mm, distance_cm):
        self.focal_mm = focal_mm
        self.sensor_w_mm = sensor_width_mm
        self.dist_mm = distance_cm * 10.0
        self.f_px = None

    def calculate_focal_pixels(self, image_width_px):
        self.f_px = (self.focal_mm * image_width_px) / self.sensor_w_mm
        return self.f_px

    def pixels_to_mm(self, px):
        if not self.f_px: return 0
        return (px * self.dist_mm) / self.f_px

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    def segment(self, blurred):
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Ensure objects are white
        white_pixels = np.sum(binary == 255)
        if white_pixels > binary.size * 0.5:
            binary = cv2.bitwise_not(binary)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def find_objects(self, binary):
        # Use RETR_LIST instead of EXTERNAL to be more robust
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        valid = []
        h, w = binary.shape[:2]
        image_area = h * w
        
        for c in contours:
            area = cv2.contourArea(c)
            # Filter by area (0.1% to 60%)
            if image_area * 0.001 <= area <= image_area * 0.6:
                # Filter noise by checking if it's likely a duplicate (parents/children)
                # For simplicity in this POC, we just keep all distinct areas
                valid.append(c)
        
        # Deduplicate overlapping contours (keep largest)
        valid.sort(key=cv2.contourArea, reverse=True)
        final = []
        for v in valid:
            is_inside = False
            for f in final:
                # If the center of V is inside F, it's likely a duplicate
                M = cv2.moments(v)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    if cv2.pointPolygonTest(f, (cx, cy), False) >= 0:
                        is_inside = True
                        break
            if not is_inside:
                final.append(v)
        
        return final

    def measure(self, image, contours):
        output = image.copy()
        img_w = image.shape[1]
        self.calculate_focal_pixels(img_w)
        
        measurements = []
        for i, c in enumerate(contours):
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box_ordered = np.int32(box)
            
            w_px = dist.euclidean(box[0], box[1])
            h_px = dist.euclidean(box[1], box[2])
            
            w_mm = self.pixels_to_mm(w_px)
            h_mm = self.pixels_to_mm(h_px)
            
            if w_mm < h_mm: w_mm, h_mm = h_mm, w_mm
            
            measurements.append({
                "label": f"Obj_{i+1}",
                "w_mm": round(w_mm, 1),
                "h_mm": round(h_mm, 1),
                "area_mm2": w_mm * h_mm
            })
            
            cv2.drawContours(output, [box_ordered], -1, (0, 255, 0), 2)
            label = f"{w_mm:.1f}x{h_mm:.1f}mm"
            cv2.putText(output, label, (int(box[0][0]), int(box[0][1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        return output, measurements


def evaluate(results, gt_path):
    if not os.path.exists(gt_path):
        print("\n[!] No ground truth found.")
        return
    
    with open(gt_path) as f:
        gt = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"  ACCURACY EVALUATION (Distance-Based at 1.5m)")
    print(f"{'='*70}")
    print(f"{'Object':<20} {'GT (mm)':>18} {'Est (mm)':>18} {'Err %':>8}")
    print("-" * 70)
    
    used = set()
    errors = []
    for name, val in gt.items():
        gt_d = sorted([val["w_mm"], val["h_mm"]], reverse=True)
        gt_area = gt_d[0] * gt_d[1]
        
        best_j, best_diff = -1, float('inf')
        for j, res in enumerate(results):
            if j in used: continue
            diff = abs(gt_area - res["area_mm2"])
            if diff < best_diff:
                best_diff, best_j = diff, j
        
        if best_j == -1:
            print(f"{name:<20} {'MISSED':>18}")
            continue
            
        used.add(best_j)
        m = results[best_j]
        est_d = [m["w_mm"], m["h_mm"]]
        
        err_w = abs(gt_d[0] - est_d[0]) / gt_d[0]
        err_h = abs(gt_d[1] - est_d[1]) / gt_d[1]
        avg_pct = (err_w + err_h) / 2 * 100
        errors.append(avg_pct)
        
        print(f"{name:<20} {gt_d[0]:>7.1f}x{gt_d[1]:<8.1f} {est_d[0]:>7.1f}x{est_d[1]:<8.1f} {avg_pct:>7.2f}%")
    
    if errors:
        print(f"\nMean Error: {np.mean(errors):.2f}%")
    print(f"{'='*70}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--image", required=True)
    p.add_argument("-d", "--distance", type=float, default=150.0)
    p.add_argument("-f", "--focal", type=float, default=26.0)
    p.add_argument("-s", "--sensor-w", type=float, default=6.3)
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None: return
    
    estimator = DepthSizeEstimator(args.focal, args.sensor_w, args.distance)
    # Save Intermediate Steps
    os.makedirs("results/debug_v2", exist_ok=True)
    cv2.imwrite("results/debug_v2/1_original.jpg", img)
    
    blurred = estimator.preprocess(img)
    cv2.imwrite("results/debug_v2/2_preprocessed.jpg", blurred)
    
    mask = estimator.segment(blurred)
    cv2.imwrite("results/debug_v2/3_binary_mask.jpg", mask)
    
    contours = estimator.find_objects(mask)
    output, results = estimator.measure(img, contours)
    cv2.imwrite("results/debug_v2/4_final_measurement.jpg", output)
    
    print(f"\n--- DEPTH ESTIMATION BREAKDOWN ---")
    print(f"Image Size: {img.shape[1]}x{img.shape[0]} px")
    print(f"Parameters: Distance={args.distance}cm, Focal={args.focal}mm, SensorW={args.sensor_w}mm")
    
    f_px = (args.focal * img.shape[1]) / args.sensor_w
    print(f"Calculated Focal Length in Pixels (F_px): {f_px:.2f}")
    print(f"Formula: (PixelSize * {args.distance*10}) / {f_px:.2f}\n")
    
    # Sort by area and show top 2
    results.sort(key=lambda x: x['area_mm2'], reverse=True)
    
    print(f"{'Object':<10} | {'Width (mm)':<12} | {'Height (mm)':<12} | {'Pixel Width':<12}")
    print("-" * 55)
    for i, r in enumerate(results[:2]):
        # Recalculate pixel width for display
        px_w = (r['w_mm'] * f_px) / (args.distance * 10)
        print(f"{r['label']:<10} | {r['w_mm']:<12.1f} | {r['h_mm']:<12.1f} | {px_w:<12.1f}")
    
    print(f"\n[!] Note: Detected {len(results)} total contours. Only showing top 2 largest.")


if __name__ == "__main__":
    main()
