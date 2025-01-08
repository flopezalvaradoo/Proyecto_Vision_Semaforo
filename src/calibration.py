from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import matplotlib.pyplot as plt
import os
import imutils
import pytesseract
from skimage import filters
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]


def write_image(filename, img):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        filename += ".jpg"
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def get_chessboard_points(chessboard_shape, dx, dy):
    chessboard = []
    for y in range(chessboard_shape[1]):
        for x in range(chessboard_shape[0]):
            chessboard.append([x * dx, y * dy, 0])

    return np.array(chessboard, dtype=np.float32)


def calibrar():
    imgs_calibration_path = glob.glob("../data/calibration/*jpg")
    imgs_calibration = load_images(imgs_calibration_path)

    corners_calibration = []
    for i in range(len(imgs_calibration)):
        corners_calibration.append(
            cv2.findChessboardCorners(imgs_calibration[i], patternSize=(9, 6))
        )

    corners_calibration_copy = copy.deepcopy(corners_calibration)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    imgs_calibration_gray = [
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs_calibration
    ]

    corners_calibration_refined = [
        (
            (cor[0], cv2.cornerSubPix(i, cor[1], (9, 6), (-1, -1), criteria))
            if cor[0]
            else (False, [])
        )
        for i, cor in zip(imgs_calibration_gray, corners_calibration_copy)
    ]
    corners_calibration = corners_calibration_refined
    imgs_calibration_copy = copy.deepcopy(imgs_calibration)

    draw_calibration_corners = []
    for i in range(len(corners_calibration)):
        if (
            corners_calibration[i][0]
            and corners_calibration[i][1] is not None
            and corners_calibration[i][1].size > 0
        ):
            img_with_corners = cv2.drawChessboardCorners(
                imgs_calibration[i],
                patternSize=(9, 6),
                corners=corners_calibration[i][1],
                patternWasFound=True,
            )
        else:
            img_with_corners = imgs_calibration[i]

        draw_calibration_corners.append(img_with_corners)

    for i in range(len(draw_calibration_corners)):
        filepath = os.path.join("../data/calibrated", f"Image_{i}_calibrated.jpg")
        write_image(filepath, draw_calibration_corners[i])

    chessboard_points = get_chessboard_points((9, 6), 30, 30)

    valid_calibration_corners = [cor[1] for cor in corners_calibration if cor[0]]

    valid_calibration_corners = np.asarray(valid_calibration_corners, dtype=np.float32)

    list_chessboard_points_calibration = [
        chessboard_points for i in valid_calibration_corners
    ]
    (
        rms_calibration,
        intrinsics_calibration,
        dist_coeffs_calibration,
        rvecs_calibration,
        tvecs_calibration,
    ) = cv2.calibrateCamera(
        list_chessboard_points_calibration,
        valid_calibration_corners,
        imgs_calibration_gray[0].shape[::-1],
        None,
        None,
    )

    extrinsics_calibration = list(
        map(
            lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)),
            rvecs_calibration,
            tvecs_calibration,
        )
    )

    return (
        rms_calibration,
        intrinsics_calibration,
        extrinsics_calibration,
        dist_coeffs_calibration,
    )
