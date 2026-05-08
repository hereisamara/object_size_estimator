"""
Object Size Estimation using fixed distance and calibrated camera intrinsics.
"""

import argparse
import json
import os

import cv2
import numpy as np
from scipy.spatial import distance as dist


def infer_image_size_from_calibration_results(data, calibration_path):
    extrinsics = data.get("extrinsics", [])
    if not extrinsics:
        return None

    base_dir = os.path.dirname(calibration_path)
    for item in extrinsics:
        image_name = item.get("image")
        if not image_name:
            continue

        candidate_paths = [
            os.path.join(base_dir, image_name),
            os.path.join(os.path.dirname(base_dir), image_name),
        ]
        for candidate in candidate_paths:
            image = cv2.imread(candidate)
            if image is not None:
                h, w = image.shape[:2]
                return (w, h)

    return None


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "camera_matrix" in data and "dist_coeffs" in data:
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        distortion = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1)
        image_size = tuple(data.get("image_size", [])) or None
        rms_error = data.get("rms_reprojection_error")
        calibration_format = "calibrate_camera.py"
    elif "intrinsic_matrix" in data and "distortion_coefficients" in data:
        camera_matrix = np.array(data["intrinsic_matrix"], dtype=np.float64)
        distortion = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
        image_size = infer_image_size_from_calibration_results(data, path)
        rms_error = data.get("rms_error")
        calibration_format = "calibrate_cam.py"
    else:
        raise ValueError(f"Unsupported calibration JSON format: {path}")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": distortion,
        "image_size": image_size,
        "rms_error": rms_error,
        "format": calibration_format,
        "source": path,
    }


class DepthSizeEstimator:
    def __init__(
        self,
        distance_cm,
        calibration=None,
        focal_mm=None,
        sensor_width_mm=None,
    ):
        self.dist_mm = distance_cm * 10.0
        self.calibration = calibration
        self.focal_mm = focal_mm
        self.sensor_w_mm = sensor_width_mm
        self.f_px = None
        self._adapted_calibration = None

    @staticmethod
    def _rotate_camera_matrix_90_cw(camera_matrix, image_size):
        width, height = image_size
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])

        return np.array(
            [
                [fy, 0.0, height - 1.0 - cy],
                [0.0, fx, cx],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def get_adapted_calibration(self, image_shape):
        if self.calibration is None:
            return None

        h, w = image_shape[:2]
        cache_key = (w, h)
        if self._adapted_calibration is not None and self._adapted_calibration["key"] == cache_key:
            return self._adapted_calibration

        base_matrix = self.calibration["camera_matrix"]
        base_distortion = self.calibration["dist_coeffs"]
        calib_size = self.calibration.get("image_size")

        if not calib_size:
            adapted = {
                "key": cache_key,
                "camera_matrix": base_matrix.copy(),
                "dist_coeffs": base_distortion,
                "orientation": "native",
                "size_scale": (1.0, 1.0),
                "warning": None,
            }
            self._adapted_calibration = adapted
            return adapted

        calib_w, calib_h = calib_size
        target_ratio = w / h
        native_ratio = calib_w / calib_h
        rotated_ratio = calib_h / calib_w

        if abs(target_ratio - rotated_ratio) < abs(target_ratio - native_ratio):
            oriented_matrix = self._rotate_camera_matrix_90_cw(base_matrix, (calib_w, calib_h))
            oriented_size = (calib_h, calib_w)
            orientation = "rotated_90_cw"
        else:
            oriented_matrix = base_matrix.copy()
            oriented_size = (calib_w, calib_h)
            orientation = "native"

        oriented_w, oriented_h = oriented_size
        scale_x = w / oriented_w
        scale_y = h / oriented_h

        scaled_matrix = oriented_matrix.copy()
        scaled_matrix[0, 0] *= scale_x
        scaled_matrix[1, 1] *= scale_y
        scaled_matrix[0, 2] *= scale_x
        scaled_matrix[1, 2] *= scale_y

        adapted_ratio = oriented_w / oriented_h
        warning = None
        if abs(target_ratio - adapted_ratio) > 0.02:
            warning = (
                f"calibration aspect ratio {adapted_ratio:.4f} does not closely match "
                f"image aspect ratio {target_ratio:.4f}"
            )

        adapted = {
            "key": cache_key,
            "camera_matrix": scaled_matrix,
            "dist_coeffs": base_distortion,
            "orientation": orientation,
            "size_scale": (scale_x, scale_y),
            "warning": warning,
        }
        self._adapted_calibration = adapted
        return adapted

    def calculate_focal_pixels(self, image_shape):
        if self.calibration is not None:
            adapted = self.get_adapted_calibration(image_shape)
            camera_matrix = adapted["camera_matrix"]
            fx = float(camera_matrix[0, 0])
            fy = float(camera_matrix[1, 1])
            self.f_px = (fx + fy) / 2.0
        else:
            image_width_px = image_shape[1]
            self.f_px = (self.focal_mm * image_width_px) / self.sensor_w_mm
        return self.f_px

    def pixels_to_mm(self, px):
        if not self.f_px:
            return 0
        return (px * self.dist_mm) / self.f_px

    def undistort(self, image):
        if self.calibration is None:
            return image, None

        adapted = self.get_adapted_calibration(image.shape)
        camera_matrix = adapted["camera_matrix"]
        distortion = adapted["dist_coeffs"]
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            (w, h),
            1,
            (w, h),
        )
        undistorted = cv2.undistort(image, camera_matrix, distortion, None, new_camera_matrix)
        
        # crop if ROI is valid
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]
            undistorted = cv2.resize(undistorted, (w, h))
            
        return undistorted, adapted


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
        self.calculate_focal_pixels(image.shape)

        measurements = []
        for i, c in enumerate(contours):
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box_ordered = np.int32(box)

            w_px = dist.euclidean(box[0], box[1])
            h_px = dist.euclidean(box[1], box[2])

            w_mm = self.pixels_to_mm(w_px)
            h_mm = self.pixels_to_mm(h_px)

            if w_mm < h_mm:
                w_mm, h_mm = h_mm, w_mm

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
            if j in used:
                continue
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
    p.add_argument(
        "--calibration",
        help="Path to calibration JSON generated by calibrate_camera.py",
    )
    p.add_argument("-f", "--focal", type=float, default=26.0)
    p.add_argument("-s", "--sensor-w", type=float, default=6.3)
    p.add_argument(
        "--no-undistort",
        action="store_true",
        help="Use calibrated focal length but skip image undistortion.",
    )
    p.add_argument(
        "--gt",
        help="Optional ground-truth JSON for evaluation on synthetic data",
    )
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        return

    calibration = load_calibration(args.calibration) if args.calibration else None
    estimator = DepthSizeEstimator(
        distance_cm=args.distance,
        calibration=calibration,
        focal_mm=args.focal,
        sensor_width_mm=args.sensor_w,
    )

    # Save Intermediate Steps
    os.makedirs("results/debug_v2", exist_ok=True)
    cv2.imwrite("results/debug_v2/1_original.jpg", img)

    if args.no_undistort:
        working_image = img
        adapted_calibration = estimator.get_adapted_calibration(img.shape) if calibration else None
    else:
        working_image, adapted_calibration = estimator.undistort(img)
    cv2.imwrite("results/debug_v2/1b_undistorted.jpg", working_image)

    blurred = estimator.preprocess(working_image)
    cv2.imwrite("results/debug_v2/2_preprocessed.jpg", blurred)

    mask = estimator.segment(blurred)
    cv2.imwrite("results/debug_v2/3_binary_mask.jpg", mask)

    contours = estimator.find_objects(mask)
    output, results = estimator.measure(working_image, contours)
    cv2.imwrite("results/debug_v2/4_final_measurement.jpg", output)

    print(f"\n--- DEPTH ESTIMATION BREAKDOWN ---")
    print(f"Image Size: {working_image.shape[1]}x{working_image.shape[0]} px")
    print(f"Distance: {args.distance}cm")

    if calibration is not None:
        camera_matrix = adapted_calibration["camera_matrix"]
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        print(f"Calibration File: {calibration['source']}")
        print(f"Calibration Format: {calibration['format']}")
        if calibration["rms_error"] is not None:
            print(f"Calibration RMS Error: {calibration['rms_error']:.4f}")
        print(f"Undistortion Applied: {not args.no_undistort}")
        print(f"Calibration Orientation: {adapted_calibration['orientation']}")
        sx, sy = adapted_calibration["size_scale"]
        print(f"Calibration Scale: sx={sx:.4f}, sy={sy:.4f}")
        print(f"Intrinsic fx={fx:.2f}px, fy={fy:.2f}px")
        if adapted_calibration["warning"]:
            print(f"[!] Calibration Warning: {adapted_calibration['warning']}")
    else:
        print(f"Parameters: Focal={args.focal}mm, SensorW={args.sensor_w}mm")

    f_px = estimator.calculate_focal_pixels(working_image.shape)
    print(f"Effective Focal Length in Pixels (F_px): {f_px:.2f}")
    print(f"Formula: (PixelSize * {args.distance*10}) / {f_px:.2f}\n")

    # Sort by area and show top 2
    results.sort(key=lambda x: x["area_mm2"], reverse=True)

    print(f"{'Object':<10} | {'Width (mm)':<12} | {'Height (mm)':<12} | {'Pixel Width':<12}")
    print("-" * 55)
    for i, r in enumerate(results[:2]):
        # Recalculate pixel width for display
        px_w = (r["w_mm"] * f_px) / (args.distance * 10)
        print(f"{r['label']:<10} | {r['w_mm']:<12.1f} | {r['h_mm']:<12.1f} | {px_w:<12.1f}")

    print(f"\n[!] Note: Detected {len(results)} total contours. Only showing top 2 largest.")

    if args.gt:
        evaluate(results, args.gt)


if __name__ == "__main__":
    main()
