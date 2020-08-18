"""
The python script contains all the utility functions.
"""
import cv2
import numpy as np

from .config import *


def capture_video():
    # Capturing the video from the camera
    cap = cv2.VideoCapture(CAMERA)
    # self.cap = cv2.VideoCapture('http://localhost:4747/mjpegfeed?640x480')
    if not cap.read()[0]:
        cap = cv2.VideoCapture(0)
    return cap


def morphological_transformations(skin):
    blur = cv2.GaussianBlur(skin, (11, 11), 0)  # generating the gaussian blur
    blur = cv2.medianBlur(blur, 15)  # generating the median blur
    return blur


def video_roi(img):
    img = cv2.flip(img, 1)
    cv2.rectangle(img, (300, 50), (600, 350), (255, 0, 0))  # creating a rectangle on the frame
    roi = img[50:350, 300:600]  # Getting the region of interest
    return img, roi


def frame_color_masking(roi, lower_bound, upper_bound):
    ycrcb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
    '''
    The inRange function simply returns a binary mask, where white pixels (255) represent pixels that fall into 
    the upper and lower limit range and black pixels (0) do not.
    '''
    skin_region_mask = cv2.inRange(ycrcb_image, lower_bound, upper_bound)
    skin = cv2.bitwise_and(roi, roi, mask=skin_region_mask)  # Performing bitwise and operation on the image
    return skin


def load_bounds():
    """Using the lower and upper bounds saved from the hand_detection.py"""

    lower_bound = np.load(LOWER_BOUND)
    upper_bound = np.load(UPPER_BOUND)

    return lower_bound, upper_bound
