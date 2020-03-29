# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:39:49 2018
Modified on Wed Dec 11 22:05:10 2019
@author1: Yongrui Huang
@author2: Ruixin Lee
"""
print("tool.py")
import sys
print("sys")
sys.path.append('../../')
sys.path.append('../')
print("import configuration")
import configuration
print("import configuration")

print("cv22222")
import cv2
print("cv2")
import os
print("os")
import random
print("random")
import string
print("string")
import queue
print("queue")
import numpy as np
print("import numpy as np")
import real_time_detection.GUI.FaceFeatureReader as FaceFeatureReader
# FaceFeatureReader
print("FaceFeatureReader")

import tensorflow as tf
print("tool.py...import tensorflow as tf")
# graph = tf.get_default_graph()
graph = tf.compat.v1.get_default_graph()    # new tensorflow version 1.14.0
print("graph = tf.compat.v1.get_default_graph()")
face_feature_reader_obj = FaceFeatureReader.FaceFeatureReader(graph)
print("face_feature_reader_obj = FaceFeatureReader.FaceFeatureReader(graph)")

# class FaceReader
class FaceReader:
    '''
    This class is used to return the face data in real time.
   
    Attribute:
        cap: the capture stream
        faceCascade: model for detecting where the face is.
        file_name: the file name of the current frame in hard disk
        delete_queue: the queue is used to save all the delete file name
        faces: the faces for predicting the emotion, we used a set of face 
        rather than one face.
    '''
    
    def __init__(self, input_type, file_path = None):
        '''
        Arguments:
            input_type: 'file' indicates that the stream is from file. In other
            case, the stream will from the defalt camera.
        '''
        self.input_type = input_type
        if input_type == 'file':
            self.cap = cv2.VideoCapture(file_path)
        else:
            self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()
            
        cascPath = configuration.MODEL_PATH + "haarcascade_frontalface_alt.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath) 
        self.file_name = None
        self.delete_queue = queue.Queue()
        self.faces = []
        
        
    
    def delete_files(self):
        '''
        delete files for releasing the resourse.
        '''
        print("delete_files()")
        while self.delete_queue.qsize() > 10:
            file = self.delete_queue.get()
            if (os.path.exists(file)):
                os.remove(file)
    
    def get_one_face(self):
        '''
        Returns:
            one face from stream.
        '''
        print("get_one_face().start...")
        if self.input_type == 'file':
            cnt = 0
            while cnt < 15:
                self.cap.read()
                cnt += 1
        ret, frame = self.cap.read()
        print("ret, frame:")
        print(ret)
        print(frame)

        if ret is True:
            print("ret is True")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (x, y, w, h) = self.detect_face(gray)
            if (w != 0):   
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                self.faces.append(face)
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
            if self.file_name is not None:
                print("self.file_name is not None")
                del_file_name = 'static/cache_image/%s.png' % self.file_name
                self.delete_queue.put(del_file_name)
        
                if self.delete_queue.qsize() > 50:
                    self.delete_files()
            
            self.file_name = ''.join(random.sample(string.ascii_letters + string.digits, 12))
            
            cv2.imwrite('static/cache_image/%s.png' % self.file_name, frame)
            print("get_one_face().end...")
            return self.file_name
        else:
            print("ERROR")
            print("get_one_face().end...")
            return 'ERROR'
        
    def detect_face(self, gray):
        '''
        find faces from a gray image.
        Arguments:
            gray: a gray image
        Returns:
            (x, y, w, h)
            x, y: the left-up points of the face
            w, h: the width and height of the face
        '''
        print("detect_face()")
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32)
        )
        print("faces:")
        print(faces)
        print("faces' len:")
        if len(faces) > 0:
            print(len(faces))
            (x, y, w, h) = faces[0]
            print("faces[0]:")
            print((x, y, w, h))
        else:
            print("0")
            (x, y, w, h) = (0, 0, 0, 0)
            
        return (x, y, w, h)
    
    def read_face_feature(self):
        '''
        Returns:
            items: a list, the first element is the frame path while the rest 
            is the feature map.
        '''
        print("read_face_feature()")
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = self.detect_face(gray)
        if (w != 0):   
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face_feature_reader_obj.set_face(face)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness = 2)
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 12))
        frame_path = 'static/cache_image/%s.png'%random_str
        cv2.imwrite(frame_path, frame)
        
        feature_map_list = face_feature_reader_obj.read_feature_map()
        items = [frame_path, ]
        items += feature_map_list
        
        self.delete_queue.put(frame_path)
        if self.delete_queue.qsize() > 10:
            self.delete_files()
        return items
# class FaceReader

print("mne")
import mne
print("mne")
from real_time_detection.EEG import EEG_feature_extract
print("from real_time_detection.EEG import EEG_feature_extract")
# from EmotivDeviceReader import EmotivDeviceReader
from real_time_detection.GUI.EmotivDeviceReader import EmotivDeviceReader
print("from real_time_detection.GUI.EmotivDeviceReader import EmotivDeviceReader")
# import GUI.EmotivDeviceReader
emotiv_reader = EmotivDeviceReader()
emotiv_reader.test()
# import time
# time.sleep(20)

emotiv_reader.start()# error
'''
 line 183, in <module>
    emotiv_reader.start()
'''


# print("tool.py...import keras")
# import keras
# print("import keras")


print("tool.py...emotiv_reader.start()...")

# class EEGReader
class EEGReader:
    '''
    This class is used to return the EEG data in real time.
    Attribute:
        raw_EEG_obj: the data for file input. MNE object.
        timestamp: the current time. how much second.
        features: the EEG features.        
    '''
    def __init__(self, input_type, file_path=None):
        '''
        Arguments:
            input_type: 'file' indicates that the stream is from file. In other
            case, the stream will from the 'Emotiv insight' device.
        '''
        self.input_type = input_type
        if self.input_type == 'file':
            self.raw_EEG_obj = mne.io.read_raw_fif(file_path, preload=True)
            max_time = self.raw_EEG_obj.times.max()
            self.raw_EEG_obj.crop(28, max_time)
            self.raw_EEG_obj
            self.timestamp = 0.
            
            cal_raw = self.raw_EEG_obj.copy()
            cal_raw = EEG_feature_extract.add_asymmetric(cal_raw)
            
            self.features = EEG_feature_extract.extract_average_psd_from_a_trial(cal_raw, 1, 0.)
        else:
            #TODO: read EEG from devic
            self.timestamp = 0.
            pass
    
    def get_EEG_data(self):

        '''
        Return:
            EEG data: the EEG data
            timestamp: the current timestamp
        '''

        '''
        ValueError: cannot reshape array of size 0 into shape (0,5,newaxis)
        '''
        if self.input_type == 'file':
            sub_raw_obj = self.raw_EEG_obj.copy().crop(self.timestamp, 
                                               self.timestamp + 1.)
            self.timestamp += 1.
            
            show_raw_obj = sub_raw_obj.copy().pick_channels(['AF3', 'AF4', 'T7', 'T8', 'Pz'])
            
            return show_raw_obj.get_data(), self.timestamp-1., None
        else:
            self.timestamp += 1.
            data_list = emotiv_reader.get_data()
            PSD_feature = np.array(data_list)
            PSD_feature = PSD_feature.reshape(PSD_feature.shape[0], 5, -1)#error
            '''
            error:
            line 236, in get_EEG_data
    PSD_feature = PSD_feature.reshape(PSD_feature.shape[0], 5, -1)
    ValueError: cannot reshape array of size 0 into shape (0,5,newaxis)
    '''
#             raw_EEG = np.mean(PSD_feature, axis = 2)
            raw_EEG = PSD_feature[:, :, 4]
            raw_EEG_fill = np.zeros((257, 5))
            for i in range(raw_EEG.shape[0]):
                start = i*(int(257/raw_EEG.shape[0]))
                end = (i+1)*(int(257/raw_EEG.shape[0])) if i != raw_EEG.shape[0]-1 else raw_EEG_fill.shape[0]
                raw_EEG_fill[start:end, :] = raw_EEG[i]
            return raw_EEG_fill.T, self.timestamp-1., PSD_feature
# class EEGReader

print("tool.py...import keras")
import keras
print("import keras")


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
    test_arr = np.zeros([48, 48])
    print("test_arr.shape: ", test_arr.shape)
    # imgs = np.array(imgs)
    # print("imgs:")
    # print("数据类型", type(imgs))  # 打印数组数据类型
    # print("数组元素数据类型：", imgs.dtype)  # 打印数组元素数据类型
    # print("数组元素总数：", imgs.size)  # 打印数组尺寸，即数组元素总数
    # print("数组形状：", imgs.shape)  # 打印数组形状
    # print("数组的维度数目", imgs.ndim)  # 打印数组的维度数目
    # print(imgs)

    # imgs = np.array(imgs) - X_mean#error
    imgs = np.array(imgs) - test_arr  # error
    # np.array(imgs)-->(?, 48, 48), but X_mean-->(10, 85)
    # ValueError: operands could not be broadcast together with shapes (50,48,48) (10,85)
    '''
    line 268, in format_raw_images_data
    imgs = np.array(imgs) - X_mean'''
    return imgs.reshape(imgs.shape[0], 48, 48, 1)


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
        print("configuration.MODEL_PATH: ", configuration.DATASET_PATH)
        
        if self.input_type == 'file':
            pass
        else:
            #TODO:1
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
        
        features = np.array(face_reader_obj.faces)#error
        print("features[0]")
        print(features[0])
        print("features[1]")
        print(features[1])

        '''
        line 327, in get_face_emotion
    features = format_raw_images_data(features, self.face_mean)
        '''
        if len(features) == 0:
            return None, None
        print("self.face_mean:")
        print(self.face_mean)
        face_mean = np.array(self.face_mean)
        print("数据类型", type(face_mean))  # 打印数组数据类型
        print("数组元素数据类型：", face_mean.dtype)  # 打印数组元素数据类型
        print("数组元素总数：", face_mean.size)  # 打印数组尺寸，即数组元素总数
        # self.face_mean-->(10, 85)
        print("数组形状：", face_mean.shape)  # 打印数组形状
        print("数组的维度数目", face_mean.ndim)  # 打印数组的维度数目
        print("self.face_mean above")
        features = format_raw_images_data(features, self.face_mean)
        with graph.as_default():
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
        
        X = EEG_reader_obj.features[self.cnt-10:self.cnt]
        X = np.array([X, ])
        X -= self.EEG_mean
        X /= self.EEG_std
        print (X.shape)
        with graph.as_default():
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
            face_valence = self.cache_valence#error
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
        
        valence = self.valence_weight*face_valence + (1-self.valence_weight)*EEG_valence
        arousal = self.arousal_weight*face_arousal + (1-self.arousal_weight)*EEG_arousal
        
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
        
        X = np.array([[valence, arousal],])
        with graph.as_default():
            distcrte_emotion, emotion_strength = self.todiscrete_model.predict(X)
        self.cnt += 1 
        
        return self.cnt, valence, arousal, distcrte_emotion[0], emotion_strength[0][0]
# class EmotionReader

print("configuration in tool.py")

trial_path = configuration.DATASET_PATH + 'MAHNOB_HCI/18/trial_1/'
# face_reader_obj = FaceReader('file', trial_path + 'video.avi')
# EEG_reader_obj = EEGReader('file', trial_path + 'EEG.raw.fif')
# emotion_reader_obj = EmotionReader('file')
face_reader_obj = FaceReader(input_type='')
EEG_reader_obj = EEGReader(input_type='')
emotion_reader_obj = EmotionReader(input_type='')

print("__main__")
if __name__ == '__main__':
    print("real_time_detection_tool.GUI.py.__main__.start...")
    for i in range(5):
        for j in range(25):
            print("i:%d" % i)
            print("j:%d" % j)
            # face_reader_obj.get_one_face()
            # print(face_reader_obj.get_one_face())
        # print(EEG_reader_obj.get_EEG_data())
        # EEG_reader_obj.get_EEG_data()
        '''
        error:
        line 449, in <module>
    EEG_reader_obj.get_EEG_data()
        '''
        # print(emotion_reader_obj.get_emotion_data())
    print("real_time_detection_tool.GUI.py.__main__.end...")
