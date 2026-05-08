import cv2
import numpy as np
import glob
import os
import sys
import json

def calibrate_camera(image_dir, rows=7, cols=9, square_size=0.025):
    """
    Calibrates camera using checkerboard images.
    
    Args:
        image_dir: Directory containing checkerboard images
        rows: Number of internal corners in height
        cols: Number of internal corners in width
        square_size: Size of one square (in your chosen unit, e.g., mm, m). 
                     Affects the scale of translation vectors.
    """
    
    # Define the dimensions of checkerboard
    CHECKERBOARD = (rows, cols)
    
    # stop the iteration when specified accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Vector for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    prev_img_shape = None
    
    # Extracting path of individual image stored in a given directory
    images = glob.glob(os.path.join(image_dir, '*.jpg'))
    images.extend(glob.glob(os.path.join(image_dir, '*.png')))
    images.sort()

    if not images:
        print(f"No images found in {image_dir}")
        return

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    valid_images = []

    print(f"Found {len(images)} images. Processing...")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            imgpoints.append(corners2)
            valid_images.append(fname)
            print(f" - Found corners in {os.path.basename(fname)}")
            
            # Optional: Draw and display the corners
            # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(0)
        else:
            print(f" - Corners NOT found in {os.path.basename(fname)}")

    if not valid_images:
        print("No valid images for calibration.")
        return

    print("\nCalibrating camera...")
    # Camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(f"Calibration successful. RMS Error: {ret}")

    # Prepare results dictionary
    results = {
        "rms_error": ret,
        "intrinsic_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "extrinsics": []
    }

    for i, fname in enumerate(valid_images):
        # rvecs is a rotation vector, can convert to matrix if needed using cv2.Rodrigues
        R, _ = cv2.Rodrigues(rvecs[i])
        
        extrinsic_data = {
            "image": os.path.basename(fname),
            "rotation_vector": rvecs[i].tolist(),
            "translation_vector": tvecs[i].tolist(),
            "rotation_matrix": R.tolist()
        }
        results["extrinsics"].append(extrinsic_data)

    # Save to JSON
    output_file = os.path.join(image_dir, "calibration_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")
    
    # Also save as numpy for easy python loading if needed
    np.savez(os.path.join(image_dir, "calibration_data.npz"), 
             mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py <image_directory> [rows] [cols]")
        print("Example: python calibrate.py ./images 7 9")
        sys.exit(1)

    image_dir = sys.argv[1]
    
    # Defaults based on previous check
    rows = 7
    cols = 9
    
    if len(sys.argv) >= 4:
        rows = int(sys.argv[2])
        cols = int(sys.argv[3])

    calibrate_camera(image_dir, rows, cols)
