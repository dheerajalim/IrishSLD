import os
import pickle
from glob import glob

import cv2
import numpy as np
import pymsgbox
from sklearn.utils import shuffle

import IrishSLD_local.config as cnf


class ModelDumping:
    """
    This class is responsible for creating the train , test and validation splits on the images and labels
    """

    def __init__(self):
        self.gesture_destination = cnf.GESTURE_DESTINATION
        self.image_label_list = list()
        self.images = None
        self.labels = None
        self.thread_stop = False

    def get_images_labels(self):

        pymsgbox.alert('Getting Images and its labels in background,'
                       ' Do not close the program. This might take some time', 'Message', button='Processing..',
                       timeout=1000)

        images = glob(f'{self.gesture_destination}/*/*.jpg')  # Getting all the images

        for image in images:
            # image_label = image[image.find(os.path.sep)+1:image.rfind(os.path.sep)]
            image_label = os.path.basename(os.path.dirname(image))
            captured_image = cv2.imread(image, 0)
            self.image_label_list.append((np.array(captured_image, dtype=np.uint8), int(image_label)))

        self.generate_image_labels()

    def generate_image_labels(self):
        self.image_label_list = shuffle(shuffle(shuffle(self.image_label_list)))
        self.images, self.labels = zip(*self.image_label_list)  # Unzipping the data to get images and labels

    @staticmethod
    def __dump_model(model_name, model):
        try:
            with open(cnf.PROJECT_FOLDER + '/' + model_name, 'wb') as model_dump:  # Dumping the model
                pickle.dump(model, model_dump)
                del model

            pymsgbox.alert(f'{model_name} model is generated..', 'Message', button='OK', timeout=500)

        except (IOError, OSError) as e:
            pymsgbox.alert(f'Error Occurred while dumping {model_name} Model', 'Error')
            print("Error Occurred while dumping Model ", e)

    def train_set(self):
        """
        Creating the train set with 60% of data from the available images
        :return:
        """
        print('Generating Training Set...')
        pymsgbox.alert('Generating Training Set...', 'Message', button='Wait', timeout=500)
        train_images = self.images[:int(0.6 * len(self.images))]
        train_labels = self.labels[:int(0.6 * len(self.labels))]
        self.__dump_model('train_images', train_images)
        self.__dump_model('train_labels', train_labels)

    def test_set(self):
        """
        Creating the test set with 20% of data from the available images
        :return:
        """
        print('Generating Test Set...')
        pymsgbox.alert('Generating Test Set...', 'Message', button='Wait', timeout=500)
        test_images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
        test_labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        self.__dump_model('test_images', test_images)
        self.__dump_model('test_labels', test_labels)

    def validation_set(self):

        """
        Creating the validation set with 20% of data from the available images
        :return:
        """
        print('Generating Validation Set...')
        pymsgbox.alert('Generating Validation Set...', 'Message', button='Wait', timeout=500)
        validation_images = self.images[int(0.8 * len(self.images)):len(self.images)]
        validation_labels = self.labels[int(0.8 * len(self.labels)):len(self.labels)]
        self.__dump_model('validation_images', validation_images)
        self.__dump_model('validation_labels', validation_labels)


if __name__ == '__main__':
    model_dumping = ModelDumping()
    model_dumping.get_images_labels()
    model_dumping.train_set()
    model_dumping.test_set()
    model_dumping.validation_set()
