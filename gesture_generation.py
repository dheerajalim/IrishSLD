"""
Filename : gesture_generation.py
Author : Dheeraj Alimchandani
Date : 12-02-2020
Usage : Generation of gesture dataset
"""

import os
import random
import sqlite3

import pymsgbox

from IrishSLD_local.frame_utils import *


class GestureCreation:

    def __init__(self):

        self.gesture_id = None  # The Id for the gesture
        self.gesture_detail = None  # The details for the gesture
        self.gesture_destination = GESTURE_DESTINATION  # Gesture storage directory name
        self.gesture_db = GESTURE_DB  # Database
        self.cap = None
        self.kernel = None
        self.gesture_count = GESTURE_COUNT  # Number of gestures to be captured
        self.capture_no = 0
        self.start_capturing_gesture = False
        self.frame_count = 0
        self.image_dimension_x = IMAGE_DIMENSION_X  # Final image width
        self.image_dimension_y = IMAGE_DIMENSION_Y  # Final image height

    def gesture_details_database(self):
        """
        1. Creates the database if not present
        2. Creates the table in database , if not exists
        :return: None
        """
        # create the folder and database if not exist
        if not os.path.exists(self.gesture_destination):
            os.mkdir(self.gesture_destination)

        if not os.path.exists(self.gesture_db):
            conn = sqlite3.connect(self.gesture_db)
            create_gesture_data = "CREATE TABLE isl_gesture " \
                                  "( gesture_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE," \
                                  " gesture_detail TEXT NOT NULL )"
            conn.execute(create_gesture_data)
            conn.commit()

    def gesture_details_data(self):
        """
        Capturing the details of the gesture, gesture ID and Gesture details and inserting it into the database
        :return: None
        """
        # self.gesture_id = input('Gesture Number')
        conn = sqlite3.connect(self.gesture_db)
        query = "SELECT max(gesture_id) FROM isl_gesture "
        cursor = conn.execute(query)
        for row in cursor:
            gesture_id = row[0]
        conn.close()
        try:

            id_value = pymsgbox.prompt('Gesture Number', default=int(gesture_id) + 1, title='Gesture Information')
            self.gesture_id = int(-1 if id_value is None else id_value)
        except (ValueError, TypeError) as e:
            pymsgbox.alert('Please enter a valid integer ID', 'Alert!')
            self.gesture_details_data()
            return

        if self.gesture_id is -1:
            exit(0)
        else:
            self.gesture_id = str(self.gesture_id)

        # self.gesture_detail = input('Gesture Details')
        self.gesture_detail = pymsgbox.prompt('Gesture Details', default='', title='Gesture Information')
        if self.gesture_detail is None:
            exit(0)

        conn = sqlite3.connect(self.gesture_db)
        query = "INSERT INTO isl_gesture (gesture_id, gesture_detail) VALUES (%s, \'%s\')" % \
                (self.gesture_id, self.gesture_detail)

        try:
            conn.execute(query)
        except sqlite3.IntegrityError:
            # choice = input("Gesture with this ID already exists. Want to change the record? (y/n): ")
            choice = pymsgbox.confirm('Gesture with this ID already exists. Want to change the record?',
                                      'Confirm', ["Yes", 'No'])
            if choice.lower() == 'yes':
                cmd = "UPDATE isl_gesture SET gesture_detail" \
                      " = \'%s\' WHERE gesture_id = %s" % (self.gesture_detail, self.gesture_id)
                conn.execute(cmd)
            else:
                conn.close()
                self.gesture_details_data()
                return

        conn.commit()

    @staticmethod
    def gesture_directory(directory_name):
        """Create the directory for the new gesture"""
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)

    def capture_gesture(self):

        self.cap = capture_video()

        self.gesture_directory(f'{self.gesture_destination}/{self.gesture_id}')  # Create gesture directory

        while True:
            _, img = self.cap.read()  # reading the captured frame

            img, roi = video_roi(img)
            cv2.putText(img, "Press C to Play and Pause capturing", (10, 460), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0))

            lower_bound, upper_bound = load_bounds()

            skin = frame_color_masking(roi, lower_bound, upper_bound)

            skin = morphological_transformations(skin)

            '''generating contours'''
            edged = cv2.Canny(skin, 100, 255)
            cv2.imshow('Canny Edged', edged)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            ''' generating contours ends'''

            if len(contours) > 0:  # Executing only if the contours are  greater than 0

                contour = max(contours, key=cv2.contourArea)
                """Verifying for the contour area and number of frames"""
                if cv2.contourArea(contour) > 150 and self.frame_count > 10:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)  # getting the bounding rectangle dimensions
                    self.capture_no += 1
                    save_img = skin[y1:y1 + h1, x1:x1 + w1]  # Setting the dimension for the image
                    if w1 > h1:
                        save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                      cv2.BORDER_CONSTANT, (0, 0, 0))
                    elif h1 > w1:
                        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                      cv2.BORDER_CONSTANT, (0, 0, 0))
                    save_img = cv2.resize(save_img, (self.image_dimension_x, self.image_dimension_y))
                    rand = random.randint(0, 10)
                    if rand % 2 == 0:
                        save_img = cv2.flip(save_img, 1)

                    cv2.putText(img, "Capturing Gesture...", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (127, 255, 255))
                    cv2.imwrite(f'{self.gesture_destination}/{self.gesture_id}/{str(self.capture_no)}.jpg', save_img)

            cv2.putText(img, str(self.capture_no), (300, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0))

            cv2.imshow('Image', img)  # Showing the real image
            cv2.imshow("Skin Toner", np.hstack([roi, skin]))  # Displaying the ROI and Skin Detected Image

            k = cv2.waitKey(1)

            if k == ord('c'):
                if not self.start_capturing_gesture:
                    self.start_capturing_gesture = True
                else:
                    self.start_capturing_gesture = False
                    self.frame_count = 0

            if self.start_capturing_gesture:
                self.frame_count += 1

            if self.capture_no == self.gesture_count:
                break

            if k == 27:  # Closing the program on press of Esc key
                break

        self.cap.release()


if __name__ == '__main__':
    gesture_creation = GestureCreation()
    gesture_creation.gesture_details_database()
    gesture_creation.gesture_details_data()
    gesture_creation.capture_gesture()

    cv2.destroyAllWindows()
