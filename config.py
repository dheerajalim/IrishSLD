# CAMERA = 'http://192.168.2.2:4747/video'  # Setting the path for the external camera over WIFI
CAMERA = 'http://localhost:4747/mjpegfeed?640x480'    # Setting the path for the external camera over USB

# PROJECT_FOLDER = ''
PROJECT_FOLDER = 'IrishSLD_local\\'
GESTURE_DESTINATION = PROJECT_FOLDER + 'irish_sign_language_gestures_1'  # Location of the Dataset

KERAS_PATH = PROJECT_FOLDER + 'keras_cnn_model.h5'  # Location of the trained keras model

GESTURE_DB = PROJECT_FOLDER + 'irish_sign_language_gestures_db.db'  # Location for the gesture database

GESTURE_COUNT = 1000

IMAGE_DIMENSION_X = 50

IMAGE_DIMENSION_Y = 50

LOWER_BOUND = PROJECT_FOLDER + 'lower_bound.npy'  # HSV Lower bound stored file location

UPPER_BOUND = PROJECT_FOLDER + 'upper_bound.npy'  # HSV Upper bound stored file location

EPOCHS = 10  # Number of Epochs for the model training

BATCH_SIZE = 500  # Batch size for the keras model

MODEL_IMAGE = PROJECT_FOLDER + 'model.png'  # Location for the Keras model structure that is generated

# UI Interface
WEB_FOLDER = PROJECT_FOLDER + 'web'  # The web directory for the eel framework
