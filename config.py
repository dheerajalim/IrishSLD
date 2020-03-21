"""
Filename : config.py
Author : Dheeraj Alimchandani
Date : 20-02-2020
Usage : Contains the configuration items
"""
CAMERA = 'http://192.168.2.2:4747/video'
# CAMERA = 'http://localhost:4747/mjpegfeed?640x480'

# PROJECT_FOLDER = ''
PROJECT_FOLDER = 'IrishSLD_local\\'
GESTURE_DESTINATION = PROJECT_FOLDER+'irish_sign_language_gestures'

KERAS_PATH = PROJECT_FOLDER+'keras_cnn_model.h5'

# GESTURE_DB = 'IrishSLD_local\\irish_sign_language_gestures_db.db'
GESTURE_DB = PROJECT_FOLDER+'irish_sign_language_gestures_db.db'

GESTURE_COUNT = 1000

IMAGE_DIMENSION_X = 50

IMAGE_DIMENSION_Y = 50

LOWER_BOUND = PROJECT_FOLDER+'lower_bound.npy'

UPPER_BOUND = PROJECT_FOLDER+'upper_bound.npy'

EPOCHS = 20

BATCH_SIZE = 500

MODEL_IMAGE = PROJECT_FOLDER+'model.png'