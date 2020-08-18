import eel
import cv2
import IrishSLD_local.hand_detection as hd
import IrishSLD_local.gesture_generation as gesturegen
import IrishSLD_local.gesture_flips as gestflips
import IrishSLD_local.dump_model as dumpmodel
import IrishSLD_local.train_model as trainmodel
import IrishSLD_local.test_model as testmodel
import IrishSLD_local.gesture_prediction as gestpredict
import IrishSLD_local.config as cnf

# eel.init('IrishSLD_local\web')
eel.init(cnf.WEB_FOLDER)

@eel.expose
def hand_detection():
    hd.HandDetection.skin_detection(hd.HandDetection())
    cv2.destroyAllWindows()

@eel.expose
def gesture_generation():
    gesture_creation = gesturegen.GestureCreation()
    gesture_creation.gesture_details_database()
    gesture_creation.gesture_details_data()
    gesture_creation.capture_gesture()

    cv2.destroyAllWindows()

@eel.expose
def gesture_flips():
    flip_gestures = gestflips.FlipGestures()
    gesture_id_user_input = gestflips.gesture_user_input()

    if gesture_id_user_input in ['n', 'N']:
        flip_gestures.flip_all_gestures()

    else:
        flip_gestures.flip_gesture(gesture_id_user_input)

@eel.expose
def dump_model():
    model_dumping = dumpmodel.ModelDumping()
    model_dumping.get_images_labels()
    model_dumping.train_set()
    model_dumping.test_set()
    model_dumping.validation_set()
    return

@eel.expose
def train_model():
    train_model = trainmodel.TrainModel()
    accuracy, cnn_error = train_model.train_model()
    accuracy = float("{0:.2f}".format(accuracy))
    # cnn_error = float("{0:.2f}".format(cnn_error))
    # K.clear_session()
    return accuracy

@eel.expose
def test_model():
    test_model = testmodel.TestModel()
    accuracy = test_model.test_model()
    accuracy = float("{0:.2f}".format(accuracy))

    return accuracy

@eel.expose
def predict_gesture():
    gesture_prediction = gestpredict.GesturePrediction()
    gesture_prediction.capture_gesture()
    cv2.destroyAllWindows()


eel.start('index.html', size= (1000,600))
