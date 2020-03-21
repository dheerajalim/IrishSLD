import IrishSLD_local.config as cnf
import numpy as np
import pickle
import cv2, os
from glob import glob
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
import pymsgbox

# K.image_dim_ordering('tf')
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


class TrainModel:

    def __init__(self):
        self.x_dimension = None
        self.y_dimension = None

    def __get_image_dimensions(self):
        """
        The method is used to get the dimensions of the saved image from the dataset
        :return: X , Y dimensions of the image
        """
        path = os.listdir(cnf.GESTURE_DESTINATION)[0]
        image_name = os.listdir(f'{cnf.GESTURE_DESTINATION}/{path}')[0]
        image = cv2.imread(f'{cnf.GESTURE_DESTINATION}/{path}/{image_name}', 0)  # Reading the Image
        self.x_dimension, self.y_dimension = image.shape  # Reading the shape of the image
        return self.x_dimension, self.y_dimension

    @staticmethod
    def __get_dumped_model(model_name):

        # with open(cnf.PROJECT_FOLDER+'/'+model_name, 'rb') as model_dump:
        try:
            with open(cnf.PROJECT_FOLDER+ model_name, 'rb') as model_dump:
                # with open(model_name, 'rb') as model_dump:              # Reading the dumped model
                return np.array(pickle.load(model_dump), dtype=np.int32)  # Loading the model
        except (FileNotFoundError, NameError) as e:
            print('Error Occured : ', e)
            pymsgbox.alert(f'Error while loading the {model_name} model, Please Retry', 'Error')



    def keras_cnn_model(self):
        model = Sequential()
        model.add(Conv2D(16, (2, 2), input_shape=(self.x_dimension, self.y_dimension, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(glob(f'{cnf.GESTURE_DESTINATION}/*')), activation='softmax'))
        sgd = optimizers.SGD(lr=1e-2)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        checkpoint1 = ModelCheckpoint(cnf.KERAS_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint1]

        plot_model(model, to_file=cnf.MODEL_IMAGE, show_shapes=True)  # Plots the CNN Model
        return model, callbacks_list

    def train_model(self):
        print('Starting Training the Model ...')
        pymsgbox.alert('Starting Training the Model ...', 'Message', timeout=1000)

        '''Loading the dumped train and validation data'''
        train_images = self.__get_dumped_model('train_images')
        train_labels = self.__get_dumped_model('train_labels')
        validation_images = self.__get_dumped_model('validation_images')
        validation_labels = self.__get_dumped_model('validation_labels')

        print('Retrieved the Images and Labels for training and Validation set')
        pymsgbox.alert('Retrieved the Images and Labels for training and Validation set', 'Message', timeout=1000)

        x_dimension, y_dimension = self.__get_image_dimensions()  # Getting the image dimensions

        # print(train_images.shape)
        # print(validation_images.shape)
        # print(train_labels.shape)
        # print(validation_labels.shape)
        # print(train_labels)
        # print('--------------')

        '''Reshaping the images for the keras model'''
        train_images = np.reshape(train_images, (train_images.shape[0], x_dimension, y_dimension, 1))
        validation_images = np.reshape(validation_images, (validation_images.shape[0], x_dimension, y_dimension, 1))

        '''Performing one hot encoding on the labels'''
        train_labels = np_utils.to_categorical(train_labels)
        validation_labels = np_utils.to_categorical(validation_labels)

        # print(train_images.shape)
        # print(validation_images.shape)
        # print(train_labels.shape)
        # print(validation_labels.shape)
        # print(train_labels)
        # exit()

        print('Calling the CNN Model')
        pymsgbox.alert('Calling the CNN Model', 'Message', timeout=1000)
        model, callbacks_list = self.keras_cnn_model()  # Calling the CNN
        model.summary()  # Generating the summary
        model.fit(train_images, train_labels, validation_data=(validation_images, validation_labels), epochs=cnf.EPOCHS,
                  batch_size=cnf.BATCH_SIZE, callbacks=callbacks_list)

        scores = model.evaluate(validation_images, validation_labels, verbose=0)  # getting the model score

        accuracy = scores[1] * 100
        cnn_error = 100 - accuracy
        print("CNN Error: %.2f%%" % cnn_error)
        print("CNN Accuracy: %.2f%%" % accuracy)

        model.save(cnf.KERAS_PATH)  # Saving the Keras Model
        pymsgbox.alert('Model Successfully Trained with %.2f%% validation accuracy' % accuracy, 'Success', timeout=2000)
        K.clear_session()
        os.startfile(cnf.MODEL_IMAGE)
        return accuracy, cnn_error


if __name__ == '__main__':
    train_model = TrainModel()
    train_model.train_model()
    # K.clear_session()
