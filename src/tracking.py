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


def read_video(videopath):
    """
    Reads a video file and returns its frames along with video properties.

    Args:
        videopath (str): The path to the video file.

    Returns:
        tuple: A tuple containing:
            - frames (list): A list of frames read from the video.
            - frame_width (int): The width of the video frames.
            - frame_height (int): The height of the video frames.
            - frame_rate (float): The frame rate of the video.
    """

    cap = cv2.VideoCapture(videopath)

    if not cap.isOpened():
        print("Error: Could not open the video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_width, frame_height, frame_rate


def cruza(video_path):
    """
    Tracks moving objects in a video and determines the crossing direction.

    Args:
        video_path (str): The path to the video file.

    Returns:
        bool: Returns True if the number of objects crossing upward is greater than downward.
              Returns False otherwise.

    Function Details:
        - Reads the video frames and properties using the `read_video` function.
        - Initializes a Kalman Filter for predicting object positions in noisy data.
        - Applies background subtraction to detect moving objects.
        - Identifies the largest contour in the frame as the primary object of interest.
        - Draws a bounding box around the detected object and predicts its future position.
        - Draws a horizontal line and counts objects crossing above or below this line.
        - Displays the video with visual annotations in real-time.
        - Stops the video display when the "q" key is pressed.
    """

    frames, frame_width, frame_height, frame_rate = read_video(video_path)

    if not frames:
        return

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

    fgbg = cv2.createBackgroundSubtractorMOG2()

    initialized = False
    last_measurement = None
    missed_frames = 0
    max_missed_frames = 10

    line_y = frame_height - 165
    line_start = (0, line_y)
    line_end = (frame_width, line_y)

    cont_arriba = 0
    cont_abajo = 0

    for frame in frames:
        fgmask = fgbg.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = np.array(
                [[np.float32(x + w // 2)], [np.float32(y + h // 2)]], dtype=np.float32
            )

            if not initialized:
                kf.statePost = np.array(
                    [[center[0][0]], [center[1][0]], [0], [0]], dtype=np.float32
                )
                kf.errorCovPost = np.eye(4, dtype=np.float32)
                initialized = True

            kf.correct(center)
            last_measurement = center
            missed_frames = 0
        else:
            missed_frames += 1
            if missed_frames > max_missed_frames and last_measurement is not None:
                kf.correct(last_measurement)
                missed_frames = max_missed_frames

        prediction = kf.predict()
        x_pred, y_pred = int(prediction[0]), int(prediction[1])

        top_left = (x_pred - w // 2, y_pred - h // 2)
        bottom_right = (x_pred + w // 2, y_pred + h // 2)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
        cv2.circle(frame, (x_pred, y_pred), 5, (0, 255, 0), -1)

        if w * h > 8000:
            if y_pred < line_y:
                cont_arriba += 1
            else:
                cont_abajo += 1

        cv2.imshow(f"frame", frame)
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    if cont_arriba < cont_abajo:
        return False
    else:
        return True

    # print(f"Frames por encima de la línea: {cont_arriba}")
    # print(f"Frames por debajo de la línea: {cont_abajo}")


# if __name__ == "__main__":
#     print(cruza("../data/videos/video_run.avi"))
