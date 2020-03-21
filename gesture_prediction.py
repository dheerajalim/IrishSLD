import IrishSLD_local.config as cnf
import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
from IrishSLD_local.frame_utils import *
from autocorrect import Speller


class GesturePrediction:

    def __init__(self):
        self.cap = None
        self.kernel = None
        self.model = load_model(cnf.KERAS_PATH)

        self.x_dimension = None
        self.y_dimension = None
        self.is_voice_on = True

    def text_to_speech(self, text):
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        if not self.is_voice_on:
            return
        while engine._inLoop:
            pass
        engine.say(text)
        engine.runAndWait()

    def __get_image_dimensions(self):
        path = os.listdir(cnf.GESTURE_DESTINATION)[0]
        image_name = os.listdir(f'{cnf.GESTURE_DESTINATION}/{path}')[0]
        image = cv2.imread(f'{cnf.GESTURE_DESTINATION}/{path}/{image_name}', 0)
        self.x_dimension, self.y_dimension = image.shape
        return self.x_dimension, self.y_dimension


    def preprocessing_image(self, img):
        x_dimension, y_dimension = self.__get_image_dimensions()
        img = cv2.resize(img, (x_dimension, y_dimension))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = np.array(img, dtype=np.float32)

        img = np.reshape(img, (1, x_dimension, y_dimension, 1))
        return img

    def predicting_gesture(self, model, image):
        processed_image = self.preprocessing_image(image)
        prediction_values = model.predict(processed_image)[0]
        prediction_labels = list(prediction_values).index(max(prediction_values))
        return max(prediction_values), prediction_labels

    def predicted_gesture_detail_db(self, prediction_labels):
        print('pred class' + str(prediction_labels))
        conn = sqlite3.connect(cnf.GESTURE_DB)
        cmd = "SELECT gesture_detail FROM isl_gesture WHERE gesture_id=" + str(prediction_labels)
        cursor = conn.execute(cmd)
        for row in cursor:
            return row[0]

    def predict_gesture_contour(self, contour, skin):
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        save_img = skin[y1:y1 + h1, x1:x1 + w1]
        detected_text = ""
        if w1 > h1:
            save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                          (0, 0, 0))
        elif h1 > w1:
            save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2), cv2.BORDER_CONSTANT,
                                          (0, 0, 0))

        prediction_values, prediction_labels = self.predicting_gesture(self.model, save_img)
        if prediction_values * 100 > 70:
            detected_text = self.predicted_gesture_detail_db(prediction_labels)

        return detected_text

    def capture_gesture(self):
        detected_text = ""
        generated_word = ""
        consistent_gesture_frame = 0

        self.cap = capture_video()

        self.kernel = morphological_kernel()    # Creating the Kernel definition

        while True:
            _, img = self.cap.read()  # reading the captured frame
            img, roi = video_roi(img)
            lower_bound, upper_bound = load_bounds()
            skin = frame_color_masking(roi, lower_bound, upper_bound)
            skin = morphological_transformations(skin, self.kernel, 2, 7)

            '''generating contours'''
            edged = cv2.Canny(skin, 100, 255)
            cv2.imshow('test', edged)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            available_text = detected_text

            if len(contours) > 0:   # Executing only if the contours are  greater than 0

                contour = max(contours, key=cv2.contourArea)
                print('area '+str(cv2.contourArea(contour)))
                if cv2.contourArea(contour) > 10:
                    print('inside')
                    detected_text = self.predict_gesture_contour(contour, skin)

                    if available_text == detected_text:
                        consistent_gesture_frame += 1
                    else:
                        consistent_gesture_frame = 0

                    if consistent_gesture_frame > 10:
                        if len(detected_text) == 1:
                            Thread(target=self.text_to_speech, args=(detected_text,)).start()
                        generated_word = generated_word + detected_text
                        if generated_word.startswith('I/Me '):
                            generated_word = generated_word.replace('I/Me ', 'I ')
                        elif generated_word.endswith('I/Me '):
                            generated_word = generated_word.replace('I/Me ', 'me ')
                        consistent_gesture_frame = 0

                elif cv2.contourArea(contour) < 1000:
                    if generated_word != '':
                        check = Speller(lang='en')
                        generated_word = check(generated_word.lower())
                        print(generated_word)
                        Thread(target=self.text_to_speech, args=(generated_word,)).start()
                    detected_text = ""
                    generated_word = ""

            else:
                if generated_word != '':
                    check = Speller(lang='en')
                    generated_word = check(generated_word.lower())
                    print(generated_word)
                    Thread(target=self.text_to_speech, args=(generated_word,)).start()
                detected_text = ""
                generated_word = ""

            prediction_screen = np.zeros((480, 640, 3), dtype=np.uint8)
            # cv2.putText(prediction_screen, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
            cv2.putText(prediction_screen, "Predicted text- " + detected_text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.putText(prediction_screen, generated_word, (30, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            if self.is_voice_on:
                cv2.putText(prediction_screen, "Voice on", (550, 460), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 127, 0))
            else:
                cv2.putText(prediction_screen, "Voice off", (550, 460), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 127, 0))

            cv2.putText(prediction_screen, "Press V to turn on/off voice", (10, 460), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255))
            res = np.hstack((img, prediction_screen))
            cv2.imshow("Recognizing gesture", res)
            cv2.imshow("roi", skin)

            k = cv2.waitKey(1)
            if k == 27:
                break

            if k == ord('v') and self.is_voice_on:
                self.is_voice_on = False
            if k == ord('v') and not self.is_voice_on:
                self.is_voice_on = True

        self.cap.release()


if __name__ == '__main__':
    gesture_prediction = GesturePrediction()
    gesture_prediction.capture_gesture()
    cv2.destroyAllWindows()
