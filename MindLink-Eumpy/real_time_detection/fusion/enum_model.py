# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 18:52:26 2018

@author: Yongrui Huang
"""

import numpy as np
import keras
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sys
sys.path.append('../../')
import configuration

def format_raw_images_data(imgs):
    '''
        conduct normalize and shape the image data in order to feed it directly
        to keras model
       
        Arguments:
        
            imgs: shape(?, 48, 48), all pixels are range from 0 to 255
        
        Return:
        
            shape(?, 48, 48, 1), image data after normalizing
        
    '''
    X_mean = np.load(configuration.DATASET_PATH + 'fer2013/X_mean.npy')
    imgs = np.array(imgs) - X_mean
    return imgs.reshape(imgs.shape[0], 48, 48, 1)

def format_EEG(EEGs):
    '''
    conduct normalize for EEG data
    
    Arguments:
        
        EEGs: shape (?, number of feature)
        
    Returns:
        
        EEGs: shape (?, number of feature), EEGs after formating
    '''
    EEGs = np.array(EEGs)
    X_mean = np.load(configuration.MODEL_PATH + 'EEG_mean.npy')
    X_std = np.load(configuration.MODEL_PATH + 'EEG_std.npy')
    EEGs -= X_mean
    EEGs /= X_std
    
    return EEGs
    
class EnumModel():
     
    '''
       This class is used to apply CNN model to predict the valence and 
     arousal. The emotion state value (valence or arousal) ranges from 1 to
     9. We consider it as a continuous value and treat this preblem as a 
     regression problem.
     
     Attributes:
            train_faces: save all faces for training
            shape: (number of trial, number of sample, 48, 48)
            
            train_EEGs: save all EEGs for training
            shape: (number of trial, number of sample, 10, 85)

            train_valence: save the ground truth of valence
            
            train_arousal: save the ground truth of arousal
            
            face_model: the implemented model for faces,
            input shape: (the number of sample, 48, 48, 1)
            output shape: (the number of sample, 1, 1)
            
            EEG_model: the implemented model for EEGs
            input shape: (the number of sample, 10, 85)
            output shape: (the number of sample, 1, 1)
            
            test_faces: save all faces for testing
            shape: (number of trial, number of sample, 48, 48)
            
            test_EEGs: save all EEGs for testing
            shape: (number of trial, number of sample, 10, 85)
            
            test_valence: save the ground truth of valence in test
            
            test_arousal: save the ground truth of arousal in test
            
            valence_weight: the weight for valence
            
            arousal_weight: the weight for arousal
            
            cache_......: this kind of attribute are used for quicker 
            calculateing. Because the output of the face and EEG model are the
            same everytime, actually.
    '''
    
    def __init__(self):
        
        
        self.train_faces = []
        self.train_EEGs = []
        self.train_valence = []
        self.train_arousal = []
        
        self.face_model = keras.models.load_model(configuration.MODEL_PATH + 
                                                  'CNN_face_regression.h5')
        self.EEG_model = keras.models.load_model(configuration.MODEL_PATH + 
                                                  'LSTM_EEG_regression.h5')
        self.test_faces = []
        self.test_EEGs = []
        self.test_valence = []
        self.test_arousal = []
        
        self.valence_weight = .5
        self.arousal_weight = .5
        
        self.cache_face_valences = []
        self.cache_face_arousals = []
        self.cache_EEG_valences = []
        self.cache_EEG_arousals = []
        
        
    def add_train_data(self, trial_path):
        '''
        add one trial data into model for training
        
        Arguments:
            
            trial_path: the trial path of the data
        
        '''
        
        label = pd.read_csv(trial_path + 'label.csv')
        faces = np.load(trial_path + 'faces.npy')
        EEGs = np.load(trial_path + 'EEG.npy')

        EEG_total_len = len(EEGs)
        one_trial_faces = []
        one_trial_EEGs = []
        
        for i in np.arange(1, EEG_total_len, 5):
            if i + 10 > EEG_total_len:
                break
            EEG_feature = EEGs[i:i+10]
            one_trial_EEGs.append(EEG_feature)
        
        for face_feature in faces:
            one_trial_faces.append(face_feature)
            
        self.train_faces.append(one_trial_faces)
        self.train_EEGs.append(one_trial_EEGs)
        self.train_valence.append(float(label['valence']))
        self.train_arousal.append(float(label['arousal']))

    def add_test_data(self, trial_path):
        '''
        add one trial data into model for testing
        
        Arguments:
            
            trial_path: the trial path of the data
        
        '''
        
        label = pd.read_csv(trial_path + 'label.csv')
        faces = np.load(trial_path + 'faces.npy')
        EEGs = np.load(trial_path + 'EEG.npy')

        EEG_total_len = len(EEGs)
        one_trial_faces = []
        one_trial_EEGs = []
        
        for i in np.arange(1, EEG_total_len, 5):
            if i + 10 > EEG_total_len:
                break
            EEG_feature = EEGs[i:i+10]
            one_trial_EEGs.append(EEG_feature)
        
        for face_feature in faces:
            one_trial_faces.append(face_feature)
            
        self.test_faces.append(one_trial_faces)
        self.test_EEGs.append(one_trial_EEGs)
        self.test_valence.append(float(label['valence']))
        self.test_arousal.append(float(label['arousal']))
        
        
    def train(self):
        '''
        train the model
        
        '''
       
        weights = np.arange(0, 1 + 1./100., 1./100.)
       
        valence_RMSE_min, arousal_RMSE_min = 9., 9.
        for weight in weights:
            pre_valence_weight = self.valence_weight
            pre_arousal_weight = self.arousal_weight
            
            self.valence_weight = weight
            self.arousal_weight = weight
            
            predict_valence, predict_arousal = self.predict(
                    self.train_faces, self.train_EEGs)
            
            valence_RMSE, arousal_RMSE = self.cal_RMSE(predict_valence, 
                    predict_arousal, self.train_valence, self.train_arousal)
            
            print ('weight:%f, valence RMSE:%f, arousal RMSE:%f' %
                   (weight, valence_RMSE, arousal_RMSE))
            if valence_RMSE < valence_RMSE_min:
                valence_RMSE_min = valence_RMSE
            else:
                self.valence_weight = pre_valence_weight
            
            if arousal_RMSE < arousal_RMSE_min:
                arousal_RMSE_min = arousal_RMSE
            else:
                self.arousal_weight = pre_arousal_weight
            
        
    def cal_RMSE(self, predict_valences, predict_arousals, valences, arousals):
        '''
        calculate the RSME
        
        Arguments:
            
            predict_valences: the valences are predicted
            
            predict_arousals: the arousals are predicted
            
            valences: the ground truth valences
            
            arousals: the ground truth arousals
            
        Returns:
            
            valence_RMSE: the root mean squared error of valence
            
            arousal_RMSE: the root mean squared error of arousal
            
        '''
        from sklearn.metrics import mean_squared_error
        import math
        valence_RMSE = math.sqrt(mean_squared_error(predict_valences, valences))
        arousal_RMSE = math.sqrt(mean_squared_error(predict_arousals, arousals))
        
        return valence_RMSE, arousal_RMSE
                
    def predict(self, faces, EEGs, testing = False):
        '''
        predict the data using the trained model
        
        Arguments:
            
            faces: the faces data set for predicting.
            shape: (trial number, sample number, 48, 48)
            
            EEGs: the EEGs data set for predicting.
            shape: (trial number, sample number, 10, 85)
            
            testing: whether it is test mode
            
        Returns:
            
            enum_valences: the valence after fusion
            
            enum_arousals: the arousal after fusion
        '''
        
        enum_valences, enum_arousals = [], []
        if len(self.cache_EEG_arousals) > 0 and testing is False:
            for (face_valence, face_arousal, EEG_valence, EEG_arousal) in zip(
                    self.cache_face_valences, self.cache_face_arousals, 
                    self.cache_EEG_valences, self.cache_EEG_arousals):
                
                enum_valence = self.valence_weight*face_valence + (1 - self.valence_weight) * EEG_valence
                enum_valences.append(enum_valence)
                
                enum_arousal = self.arousal_weight*face_arousal + (1-self.arousal_weight) * EEG_arousal
                enum_arousals.append(enum_arousal)
                
                
            return enum_valences, enum_arousals
        
        for (one_trial_faces, one_trial_EEGs) in zip(faces, EEGs):
            one_trial_faces = format_raw_images_data(one_trial_faces)
            one_trial_EEGs = format_EEG(one_trial_EEGs)
            
            (face_valences, face_arousals) = self.face_model.predict(
                    one_trial_faces)
            (EEG_valences, EEG_arousals) = self.EEG_model.predict(
                    one_trial_EEGs)
            face_valence = np.mean(face_valences)
            face_arousal = np.mean(face_arousals)
            EEG_valence = np.mean(EEG_valences) + np.random.randint(2)
            EEG_arousal = np.mean(EEG_arousals) + np.random.randint(2)
            
            if testing is False:
                self.cache_face_valences.append(face_valence)
                self.cache_face_arousals.append(face_arousal)
                self.cache_EEG_valences.append(EEG_valence)
                self.cache_EEG_arousals.append(EEG_arousal)
#            print ('--------------------------------------')
#            print (face_valences.shape, face_arousals.shape, EEG_valences.shape, EEG_arousals.shape)
#            print ('--------------------------------------')
            
            enum_valence = self.valence_weight*face_valence + (1 - self.valence_weight) * EEG_valence
            enum_valences.append(enum_valence)
            
            enum_arousal = self.arousal_weight*face_arousal + (1-self.arousal_weight) * EEG_arousal
            enum_arousals.append(enum_arousal)
        
        return enum_valences, enum_arousals
    
    def evalute(self):
        '''
        evalute the performance
        
        '''
        
        from sklearn.metrics import mean_squared_error
        import math
        
        self.test_faces = np.array(self.test_faces)
        self.test_EEGs = np.array(self.test_EEGs)
        
        (predict_valence, predict_arousal) = self.predict(self.test_faces, 
        self.test_EEGs, testing=True)

        print ('Test valence RMSE:%f' % 
               math.sqrt(mean_squared_error(predict_valence, 
                                            self.test_valence)))
        print ('Test arousal RMSE:%f' %
               math.sqrt(mean_squared_error(predict_arousal, 
                                            self.test_arousal)))
        print ('Predict valence sample: ' + str(predict_valence[:10]))
        print ('Predict arousal sample: ' + str(predict_arousal[:10]))
       
        
        predict_valence = np.array(predict_valence) > 5
        predict_arousal = np.array(predict_arousal) > 5
        self.test_valence = np.array(self.test_valence)
        self.test_arousal = np.array(self.test_arousal)
        self.test_valence = np.array(self.test_valence) > 5
        self.test_arousal = np.array(self.test_arousal) > 5
        
        
        #for classification

        print ('valence acc: %f, f1: %f' % 
               (accuracy_score(predict_valence, self.test_valence), 
                f1_score(predict_valence, self.test_valence)))
        print ('arousal acc: %f, f1: %f' % 
               (accuracy_score(predict_arousal, self.test_arousal), 
                f1_score(predict_arousal, self.test_arousal)))
        
        print ('Train sample length: ' + str(len(self.train_valence)))
        
        print ('Test sample length: ' + str(len(self.test_valence)))
        
        save_obj = (self.valence_weight, self.arousal_weight)
        np.save(configuration.MODEL_PATH + 'enum_weights.npy', save_obj)
        
        face_valence_RMSE, face_arousal_RMSE = self.cal_RMSE(
                self.cache_face_valences, self.cache_face_arousals,
                self.train_valence, self.train_arousal)
        
        
        EEG_valence_RMSE, EEG_arousal_RMSE = self.cal_RMSE(
                self.cache_EEG_valences, self.cache_EEG_arousals,
                self.train_valence, self.train_arousal)
        
        print('face RMSE: valence: %f, aorusal: %f' % (face_valence_RMSE, 
              face_arousal_RMSE))
        print('EEG RMSE: valence: %f, aorusal: %f' % (EEG_valence_RMSE, 
              EEG_arousal_RMSE))
        
        
        predict_valence, predict_arousal = self.predict(
                    self.train_faces, self.train_EEGs)
            
        valence_RMSE, arousal_RMSE = self.cal_RMSE(predict_valence, 
                    predict_arousal, self.train_valence, self.train_arousal)
        
        print ('Fusion, valence RMSE:%f, arousal RMSE:%f' %
                   (valence_RMSE, arousal_RMSE))
        
        
        
if __name__ == '__main__':
    root_path = configuration.DATASET_PATH + 'MAHNOB_HCI/'  
    
    model = EnumModel()
    test_idx = set(range(1, 10))
    train_idx = set(range(1, 25)) - test_idx
    
    for subject_id in train_idx:
        subject_path = root_path + str(subject_id) + '/'
        
        for trial_id in range(1, 21):
            trial_path = '%s/trial_%d/'%(subject_path, trial_id)
            model.add_train_data(trial_path)
                
    model.train()
    
    for subject_id in test_idx:
        subject_path = root_path + str(subject_id) + '/'
        
        for trial_id in range(1, 21):
            trial_path = '%s/trial_%d/'%(subject_path, trial_id)
            model.add_test_data(trial_path)
    
    
    model.evalute()
    data = np.load(configuration.MODEL_PATH + 'enum_weights.npy')
    print (data)