"""
Filename : hand_detection.py
Author : Dheeraj Alimchandani
Date : 09-02-2020
Usage : Detection of Skin from the Video frame
"""

import cv2
import numpy as np


class HandDetection:
    """
    The Class works on
     -- generating the trackbar for the YCrCb color boundaries
     -- Detecting the color of the skin from the video frame
    """

    def __init__(self):
        # Capturing the video from the camera
        self.cap = cv2.VideoCapture('http://192.168.2.2:4747/video')
        if not self.cap.read()[0]:
            self.cap = cv2.VideoCapture(0)

        cv2.namedWindow('Track Bar')  # window for the track bar
        self.create_trackbar()

    def nothing(self, x):
        pass

    def create_trackbar(self):
        """

        Y – Luminance or Luma component obtained from RGB after gamma correction.
        Cr = R – Y ( how far is the red component from Luma ).
        Cb = B – Y ( how far is the blue component from Luma ).

        :return: Return the Lower and Upper bound for the YCrCb
        """

        # Trackbar creation for the lower values of the YCrCb color
        cv2.createTrackbar('LY', 'Track Bar', 0, 255, self.nothing)
        cv2.createTrackbar('LCr', 'Track Bar', 0, 255, self.nothing)
        cv2.createTrackbar('LCb', 'Track Bar', 0, 255, self.nothing)

        # Trackbar creation for the upper values of the YCrCb color
        cv2.createTrackbar('UY', 'Track Bar', 255, 255, self.nothing)
        cv2.createTrackbar('UCr', 'Track Bar', 255, 255, self.nothing)
        cv2.createTrackbar('UCb', 'Track Bar', 255, 255, self.nothing)

    @staticmethod
    def get_trackbar():
        """

        :return: The list of upper values and lower values of YCrCb color
        """

        # Getting the current value of the lower YCrCb color
        ly = cv2.getTrackbarPos('LY', 'Track Bar')
        lcr = cv2.getTrackbarPos('LCr', 'Track Bar')
        lcb = cv2.getTrackbarPos('LCb', 'Track Bar')

        # Getting the current value of the Upper YCrCb color
        uy = cv2.getTrackbarPos('UY', 'Track Bar')
        ucr = cv2.getTrackbarPos('UCr', 'Track Bar')
        ucb = cv2.getTrackbarPos('UCb', 'Track Bar')

        lower_bound_values = [ly, lcr, lcb]  # List of Lower bound values
        upper_bound_values = [uy, ucr, ucb]  # List of Upper bound values

        return lower_bound_values, upper_bound_values

    def skin_detection(self):
        """
        The method takes the frame from the live video feed and from the values of the trackbar
        detects the skin color from the frame and generates the frame with only skin
        :return:
        """

        while True:
            _, img = self.cap.read()  # reading the captured frame
            # cv2.rectangle(img, (300, 100), (600, 400), (255, 0, 0)) # creating a rectangle on the
            cv2.rectangle(img, (30, 50), (330, 350), (255, 0, 0))  # creating a rectangle on the frame
            # roi = img[100:400, 300:600]
            roi = img[50:350, 30:330]  # Getting the region of interest
            roi = cv2.flip(roi, 1)  # Flipping the image

            lower_bound, upper_bound = self.get_trackbar()  # Getting the tracker values

            # generating a numpy array for the lower and upper bound color values
            lower_bound = np.array(lower_bound, np.uint8)
            upper_bound = np.array(upper_bound, np.uint8)

            ycrcb_image = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)  # changing BGR to YCrCb color

            '''
            The inRange function simply returns a binary mask, where white pixels (255) represent pixels that fall into 
            the upper and lower limit range and black pixels (0) do not.
            '''
            skin_region_mask = cv2.inRange(ycrcb_image, lower_bound, upper_bound)

            skin = cv2.bitwise_and(roi, roi, mask=skin_region_mask)  # Performing bitwise and operation on the image

            cv2.imshow('Image', cv2.flip(img, 1))  # Showing the real image
            cv2.imshow("Skin Toner", np.hstack([roi, skin]))  # Displaying the ROI and Skin Detected Image

            k = cv2.waitKey(1)
            if k == 27:  # Closing the program on press of Esc key
                break

        self.cap.release()


if __name__ == '__main__':
    HandDetection.skin_detection(HandDetection())
    cv2.destroyAllWindows()
