"""
Filename : test_model.py
Author : Dheeraj Alimchandani
Date : 12-02-2020
Usage : Testing the trained model
"""

import os
import pickle

import cv2
import numpy as np
import pymsgbox
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as cfr

import IrishSLD_local.config as cnf


class TestModel:

    def __init__(self):
        self.x_dimension = None
        self.y_dimension = None
        self.model = None

    def __get_image_dimensions(self):
        path = os.listdir(cnf.GESTURE_DESTINATION)[0]
        image_name = os.listdir(f'{cnf.GESTURE_DESTINATION}/{path}')[0]
        image = cv2.imread(f'{cnf.GESTURE_DESTINATION}/{path}/{image_name}', 0)
        self.x_dimension, self.y_dimension = image.shape
        return self.x_dimension, self.y_dimension

    @staticmethod
    def __get_dumped_model(model_name):
        with open(cnf.PROJECT_FOLDER + model_name, 'rb') as model_dump:
            return np.array(pickle.load(model_dump), dtype=np.int32)

    def test_model(self):
        print('Starting Testing the Model ...')
        pymsgbox.alert('Starting Testing the Model ...', 'Message', timeout=1000)
        test_images = self.__get_dumped_model('test_images')
        test_labels = self.__get_dumped_model('test_labels')

        x_dimension, y_dimension = self.__get_image_dimensions()

        test_images = np.reshape(test_images, (test_images.shape[0], x_dimension, y_dimension, 1))

        self.model = load_model(cnf.KERAS_PATH)  # Loading the keras model
        pred_labels = []

        pred_probabs = self.model.predict(test_images)  # Predicitng the probabilities

        for pred_probab in pred_probabs:
            pred_labels.append(list(pred_probab).index(max(pred_probab)))

        # cm = confusion_matrix(test_labels, np.array(pred_labels))

        classification_report = cfr(test_labels, np.array(pred_labels))

        accuracy = accuracy_score(test_labels, np.array(pred_labels))  # getting the model score

        print('\n\nClassification Report')
        print('---------------------------')
        print(classification_report)
        pymsgbox.alert(classification_report, 'Classification Report')
        return accuracy * 100


if __name__ == '__main__':
    test_model = TestModel()
    test_model.test_model()
