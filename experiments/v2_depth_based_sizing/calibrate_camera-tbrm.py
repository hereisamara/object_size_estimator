"""
Camera calibration utility for the v2 depth-based sizing experiment.

Usage example:
python calibrate_camera.py \
  --images "calibration_images/*.jpg" \
  --pattern-cols 9 \
  --pattern-rows 6 \
  --square-mm 24.0 \
  --output calibration/iphone_main_cam.json
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np


def build_object_points(pattern_size, square_mm):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_mm
    return objp


def detect_corners(image_path, pattern_size):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return image, gray, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return image, gray, refined


def save_debug_image(image, pattern_size, corners, output_path):
    debug = image.copy()
    cv2.drawChessboardCorners(debug, pattern_size, corners, True)
    cv2.imwrite(output_path, debug)


def calibrate_from_images(image_paths, pattern_size, square_mm, debug_dir=None):
    objpoints = []
    imgpoints = []
    objp = build_object_points(pattern_size, square_mm)
    image_size = None
    used_images = []

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    for image_path in image_paths:
        image, gray, corners = detect_corners(image_path, pattern_size)
        if image is None:
            continue
        if corners is None:
            print(f"[skip] corners not found: {image_path}")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        image_size = gray.shape[::-1]
        used_images.append(image_path)
        print(f"[ok] corners found: {image_path}")

        if debug_dir:
            filename = os.path.basename(image_path)
            save_debug_image(
                image,
                pattern_size,
                corners,
                os.path.join(debug_dir, f"corners_{filename}"),
            )

    if len(objpoints) < 3:
        raise ValueError("Need at least 3 valid checkerboard images for calibration.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    mean_error = 0.0
    per_view_errors = []
    for i, obj in enumerate(objpoints):
        projected, _ = cv2.projectPoints(
            obj,
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs,
        )
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        per_view_errors.append(
            {
                "image": used_images[i],
                "mean_reprojection_error_px": float(error),
            }
        )
        mean_error += error

    mean_error /= len(objpoints)

    return {
        "pattern_cols": pattern_size[0],
        "pattern_rows": pattern_size[1],
        "square_size_mm": square_mm,
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "rms_reprojection_error": float(rms),
        "mean_reprojection_error_px": float(mean_error),
        "views_used": len(used_images),
        "images_used": used_images,
        "per_view_errors": per_view_errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        required=True,
        help='Glob for checkerboard images, e.g. "calibration_images/*.jpg"',
    )
    parser.add_argument(
        "--pattern-cols",
        type=int,
        required=True,
        help="Number of inner corners across the checkerboard width.",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        required=True,
        help="Number of inner corners across the checkerboard height.",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        required=True,
        help="Physical square size in millimeters.",
    )
    parser.add_argument(
        "--output",
        default="calibration/camera_calibration.json",
        help="Where to save the calibration JSON.",
    )
    parser.add_argument(
        "--debug-dir",
        default="calibration/debug_corners",
        help="Where to save checkerboard corner visualizations.",
    )
    args = parser.parse_args()

    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No images matched: {args.images}")

    pattern_size = (args.pattern_cols, args.pattern_rows)
    calibration = calibrate_from_images(
        image_paths=image_paths,
        pattern_size=pattern_size,
        square_mm=args.square_mm,
        debug_dir=args.debug_dir,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print("\n--- CALIBRATION COMPLETE ---")
    print(f"Views Used: {calibration['views_used']}")
    print(f"Image Size: {calibration['image_size'][0]}x{calibration['image_size'][1]}")
    print(f"RMS Reprojection Error: {calibration['rms_reprojection_error']:.4f}")
    print(f"Mean Reprojection Error: {calibration['mean_reprojection_error_px']:.4f}px")
    print(f"Saved To: {args.output}")


if __name__ == "__main__":
    main()
