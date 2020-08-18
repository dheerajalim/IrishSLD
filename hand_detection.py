"""
Filename : hand_detection.py
Author : Dheeraj Alimchandani
Date : 09-02-2020
Usage : Detection of Skin from the Video frame
"""

import pymsgbox

from IrishSLD_local.frame_utils import *


class HandDetection:
    """
    The Class works on
     -- generating the trackbar for the YCrCb color boundaries
     -- Detecting the color of the skin from the video frame
    """

    def __init__(self):
        self.cap = capture_video()

        cv2.namedWindow('Track Bar')  # window for the track bar
        # cv2.resizeWindow('Track Bar', 500, 300)
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

            img, roi = video_roi(img)

            cv2.putText(img, "Press S to save the detected skin", (10, 460), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))
            lower_bound, upper_bound = self.get_trackbar()  # Getting the tracker values

            # generating a numpy array for the lower and upper bound color values
            lower_bound = np.array(lower_bound, np.uint8)
            upper_bound = np.array(upper_bound, np.uint8)

            skin = frame_color_masking(roi, lower_bound, upper_bound)

            skin = morphological_transformations(skin)  # Adding improvements over the detected skin

            cv2.imshow('Image', img)  # Showing the real image
            cv2.imshow("Skin Toner", np.hstack([roi, skin]))  # Displaying the ROI and Skin Detected Image

            k = cv2.waitKey(1)

            if k == ord('s'):  # Pressing s , saves the current YCrCb color values
                pymsgbox.alert('Detected skin bounds saved', 'Saved')
                print(lower_bound, upper_bound)
                np.save(LOWER_BOUND, lower_bound)
                np.save(UPPER_BOUND, upper_bound)

            if k == 27:  # Closing the program on press of Esc key
                break

        self.cap.release()


if __name__ == '__main__':
    HandDetection.skin_detection(HandDetection())
    cv2.destroyAllWindows()
