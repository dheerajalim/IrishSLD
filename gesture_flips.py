"""
The python script is responsible for generating the flips for th created gestures
"""
import os

import cv2
import pymsgbox

from IrishSLD_local.config import *


class FlipGestures:

    def __init__(self):
        self.gesture_destination = GESTURE_DESTINATION  # Gesture storage directory name
        self.gesture_count = GESTURE_COUNT  # Total number of gestures to be generated

    def image_flip(self, gesture_id):
        """
        The method flips the image with the particular gesture id
        :param gesture_id: gesture id specified by the user
        :return: flipped image
        """
        images = os.listdir(f'{self.gesture_destination}/{gesture_id}')
        if len(images) >= self.gesture_count * 2:
            pymsgbox.alert(f'Gesture id {gesture_id} has {self.gesture_count * 2} or more images', 'Alert',
                           timeout=1000)
            return
        else:
            pymsgbox.alert('Flipping gesture with id ' + gesture_id + '...', 'Alert!', button='Wait', timeout=500)
            for image in images:
                img = cv2.imread(f'{self.gesture_destination}/{gesture_id}/{image}')  # Reading the image
                img = cv2.flip(img, 1)  # Flipping the image
                cv2.imwrite(f'{self.gesture_destination}/{gesture_id}/{"flip_" + image}', img)
            pymsgbox.alert('Flipping Successful for gesture id' + gesture_id, 'Succ̥ess', timeout=1000)

    def flip_all_gestures(self):
        """
        This method flips all the gesture , not just a specific gesture
        :return: None
        """
        if os.path.exists(self.gesture_destination):
            for gesture_id in os.listdir(self.gesture_destination):
                self.image_flip(gesture_id)
        else:

            pymsgbox.alert('The specified Gesture Directory does not exists, Please retry!', 'Flips Failed')
            return

        pymsgbox.alert('Flipping Process Completed', 'Alert')

    def flip_gesture(self, gesture_id):
        """
        This method flips specific gesture with a particular specific GESTURE ID
        :param gesture_id: gesture id input by the user
        :return: None
        """
        if os.path.exists(f'{self.gesture_destination}/{gesture_id}'):
            self.image_flip(gesture_id)
        else:
            pymsgbox.alert('The specified Gesture Directory does not exists, Please retry!', 'Flips Failed')
            return


def gesture_user_input():
    gesture_id = pymsgbox.prompt('Enter the specific Gesture ID to flip or N/n to flip all the gestures', default='',
                                 title='Gesture ID')
    return gesture_id


if __name__ == '__main__':
    flip_gestures = FlipGestures()
    gesture_id_user_input = gesture_user_input()

    if gesture_id_user_input in ['n', 'N']:
        flip_gestures.flip_all_gestures()

    else:
        flip_gestures.flip_gesture(gesture_id_user_input)
