print("tool.py...import keras")
import numpy as np
import keras
print("import keras")
import configuration
from real_time_detection.GUI.MLE_tool.FaceReader import FaceReader
from real_time_detection.GUI.MLE_tool.EEGReader import EEGReader
# from real_time_detection.GUI.MLE_tool.mytool import format_raw_images_data
# 该函数format_raw_images_data()放置在EmotionReader.py内即可，
# 如果放在mytool里面然后在本文件中 导入mytool，会导致循环import，文件出错

face_reader_obj = FaceReader(input_type='')
EEG_reader_obj = EEGReader(input_type='')


print("class EmotionReader")


# class EmotionReader
class EmotionReader:
    '''
    This class is used to return the emotion in real time.
    Attribute:
        input_tpye: input_type: 'file' indicates that the stream is from file.
        In other case, the stream will from the default camera.
        face_model: the model for predicting emotion by faces.
        EEG_model: the model for predicting emotion by EEG.
        todiscrete_model: the model for transforming continuous emotion (valence
        and arousal) into discrete emotion.
        face_mean: the mean matrix for normalizing faces data.
        EEG_mean: the mean matrix for normalizing EEG data.
        EEG_std: the std matrix for normalizing EEG data.
        valence_weigth: the valence weight for fusion
        aoursal_weight: the arousal weight for fusion
        cache_valence: the most recent valence, in case we don't have data to
        predict we return the recent data.
        cacha_arousal: the most recent arousal.

    '''

    def __init__(self, input_type):
        '''
        Arguments:
            input_type: 'file' indicates that the stream is from file. In other
            case, the stream will from the defalt camera.
        '''
        self.graph = GET_graph()
        self.input_type = input_type

        self.face_model = keras.models.load_model(configuration.MODEL_PATH + 'CNN_face_regression.h5')
        self.EEG_model = keras.models.load_model(configuration.MODEL_PATH + 'LSTM_EEG_regression.h5')
        self.todiscrete_model = keras.models.load_model(configuration.MODEL_PATH + 'continuous_to_discrete.h5')
        self.face_mean = np.load(configuration.DATASET_PATH + 'fer2013/X_mean.npy')
        self.EEG_mean = np.load(configuration.MODEL_PATH + 'EEG_mean.npy')
        self.EEG_std = np.load(configuration.MODEL_PATH + 'EEG_std.npy')
        (self.valence_weight, self.arousal_weight) = np.load(configuration.MODEL_PATH + 'enum_weights.npy')

        self.cache_valence, self.cache_arousal = None, None
        self.cnt = 0

        if self.input_type == 'file':
            pass
        else:
            # TODO:1
            pass
        '''
        error:
        line 327, in get_face_emotion
    features = format_raw_images_data(features, self.face_mean)
        '''

    def get_face_emotion(self):
        '''
        Returns:
            valence: the valence predicted by faces
            arousal: the arousal predicted by faces
        '''

        features = np.array(face_reader_obj.faces)  # error
        '''
        line 327, in get_face_emotion
    features = format_raw_images_data(features, self.face_mean)
        '''
        if len(features) == 0:
            return None, None
        features = format_raw_images_data(features, self.face_mean)

        with self.graph.as_default():
            (valence_scores, arousal_scores) = self.face_model.predict(features)
        face_reader_obj.faces = []
        return valence_scores.mean(), arousal_scores.mean()

    def get_EEG_emotion(self):
        '''

        line 356, in get_continuous_emotion_data

        '''

        '''
        Returns:
            valence: the valence predicted by EEG
            arousal: the arousal predicted by EEG
        '''

        X = EEG_reader_obj.features[self.cnt - 10:self.cnt]
        X = np.array([X, ])
        X -= self.EEG_mean
        X /= self.EEG_std
        print(X.shape)
        with self.graph.as_default():
            (valence_scores, arousal_scores) = self.EEG_model.predict(X)
        return valence_scores[0][0], valence_scores[0][0]

    #        line 356, in get_continuous_emotion_data
    #     face_valence, face_arousal = self.get_face_emotion()

    def get_continuous_emotion_data(self):
        '''
        Returns:
            valence: the valence value predicted by final model
            arousal: the arousal value predicted by final model
        '''
        face_valence, face_arousal = self.get_face_emotion()
        if face_valence is None:
            face_valence = self.cache_valence  # error
            '''
            error:
            line 386, in get_emotion_data
    return face_valence, face_arousal
            '''
            face_arousal = self.cache_arousal
        if self.cnt < 10 or self.input_type != 'file':
            self.cache_valence = face_valence
            self.cache_arousal = face_arousal

            return face_valence, face_arousal

        EEG_valence, EEG_arousal = self.get_EEG_emotion()

        valence = self.valence_weight * face_valence + (1 - self.valence_weight) * EEG_valence
        arousal = self.arousal_weight * face_arousal + (1 - self.arousal_weight) * EEG_arousal

        self.cache_valence = valence
        self.cache_arousal = arousal

        return valence, arousal

    def get_emotion_data(self):
        '''
        line 386, in get_emotion_data
    valence, arousal = self.get_continuous_emotion_data()
        '''

        '''
        Returns:
            cnt: the timestamp
            valence: the valence value predicted by final model.
            arousal: the arousal value predicted by final model.
            discrete_emotion: a vector contains 32 emotion scores.
            emotion_strength: the emotion strength.
        '''
        valence, arousal = self.get_continuous_emotion_data()

        X = np.array([[valence, arousal], ])
        with self.graph.as_default():
            distcrte_emotion, emotion_strength = self.todiscrete_model.predict(X)
        self.cnt += 1

        return self.cnt, valence, arousal, distcrte_emotion[0], emotion_strength[0][0]
# class EmotionReader


'''
return graph(tensorflow)
'''
print("tool.py...import tensorflow as tf")
def GET_graph():
    import tensorflow as tf
    return tf.compat.v1.get_default_graph()
# graph = tf.compat.v1.get_default_graph()


def format_raw_images_data(imgs, X_mean):
    '''
        conduct normalize and shape the image data in order to feed it directly
        to keras model

        Arguments:

            imgs: shape(?, 48, 48), all pixels are range from 0 to 255

            X_mean: shape: (48, 48), the mean of every feature in faces

        Return:

            shape(?, 48, 48, 1), image data after normalizing

    '''
    '''
    error:
     line 268, in format_raw_images_data
    imgs = np.array(imgs) - X_mean
    '''
    imgs = np.array(imgs) - X_mean  # error
    '''
    line 268, in format_raw_images_data
    imgs = np.array(imgs) - X_mean'''
    return imgs.reshape(imgs.shape[0], 48, 48, 1)