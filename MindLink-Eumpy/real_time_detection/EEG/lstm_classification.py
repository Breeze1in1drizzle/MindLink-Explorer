# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:18:08 2018

@author: Yongrui Huang
"""

import keras.layers as L
from keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

class LstmModelClf():
    '''
        This class is used to apply lstm model to classify the valence and 
        arousal. The emotion state value (valence or arousal) ranges from 1 to
        9. We consider it as two category. When it is higher than 5, we take it
        as high category. when it is less than or equal with 5, we take it as
        low category.
        
        Attributes:
            
            train_X: the list saves all train sample
            
            train_valence: save the ground truth of valence
            
            train_arousal: save the ground truth of arousal
            
            model: the implement model for training and predicting
            
            X_mean: the mean vertor for the feature
            
            X_std: the standard deviation vertor for the feature
            
            X_test: the list saves all test sample
            
            test_valence: save the ground truth of valence in test
            
            test_arousal: save the ground truth of arousal in test
    
    
    '''
    def __init__(self):
        self.train_X = []
        self.train_valence = []
        self.train_arousal = []
        self.model = None
        self.X_mean = None
        self.X_std = None
        
        self.test_X = []
        self.test_valence = []
        self.test_arousal = []
        
    def add_train_data(self, trial_path):
        '''
        add one trial data into model for training
        
        Arguments:
            
            trial_path: the trial path of the data
        
        '''
        
        label = pd.read_csv(trial_path + 'label.csv')
        EEGs = np.load(trial_path + 'EEG.npy')

        total_len = len(EEGs)
        for i in np.arange(1, total_len, 5):
            if i + 10 > total_len:
                break
            feature = EEGs[i:i+10]
            self.train_X.append(feature)
            
            self.train_valence.append(int(label['valence'] > 5))
            self.train_arousal.append(int(label['arousal'] > 5))
            

            
    def add_test_data(self, trial_path):
        '''
        add one trial data into model for testing
        
        Arguments:
            
            trial_path: the trial path of the data
        
        '''
        
        label = pd.read_csv(trial_path + 'label.csv')
        EEGs = np.load(trial_path + 'EEG.npy') 
        
        total_len = len(EEGs)
        for i in np.arange(1, total_len, 5):
            if i + 10 > total_len:
                break
            feature = EEGs[i:i+10]
            self.test_X.append(feature)
            
            self.test_valence.append(int(label['valence'] > 5))
            self.test_arousal.append(int(label['arousal'] > 5))

            
    def train(self):
        '''
        train the model
        
        '''
        
        self.train_X = np.array(self.train_X)
        self.X_mean = np.mean(self.train_X, axis = 0)
        self.X_std = np.std(self.train_X, axis = 0)
        
        #data standardization
        self.train_X -= self.X_mean
        self.train_X /= self.X_std
        
        #train
        self.model = self.build_model((self.train_X.shape[1], self.train_X.shape[2]))
        epoch = 12
        batch_size = 256;
        self.model.fit(self.train_X, [self.train_valence, self.train_valence],
                                 verbose=2, validation_split = 0.1, 
                                 epochs=epoch, batch_size=batch_size)
        
    def predict(self, X):
        '''
        predict the data using the trained model
        
        Arguments:
            
            X: the data set for predicting.
            shape: (sample number, feature number)
        '''
        X -= self.X_mean
        X /= self.X_std
        
        (valence_scores, arousal_scores) = self.model.predict(X)
        
        return (valence_scores > 0.5, arousal_scores > 0.5)
    
    def evalute(self):
        '''
        evalute the performance
        
        '''
        self.test_X = np.array(self.test_X)
        (predict_valence, predict_arousal) = self.predict(self.test_X)
        
        print ('valence acc: %f, f1: %f' % 
               (accuracy_score(predict_valence, self.test_valence), 
                f1_score(predict_valence, self.test_valence)))
        print ('arousal acc: %f, f1: %f' % 
               (accuracy_score(predict_arousal, self.test_arousal), 
                f1_score(predict_arousal, self.test_arousal)))
        
        print ('Train sample proportion: valence: %f, arousal: %f' % (
                float(sum(self.train_valence))/len(self.train_valence), 
                float(sum(self.train_arousal))/len(self.train_arousal)))
        
        print ('Test sample proportion: valence: %f, arousal: %f' % (
                float(sum(self.test_valence))/len(self.test_valence), 
                float(sum(self.test_arousal))/len(self.test_arousal)))
        
        print ('Train sample shape: ' + str(self.train_X.shape))
        
        print ('Test sample shape: ' + str(self.test_X.shape))
        
        
    def build_model(self, input_shape):
        '''
        build the model
        
        Arguments:
            
            input_shape: the input shape of the sample.
        
        '''
        X_input = L.Input(shape = input_shape)                             
                                          
        X = L.LSTM(16, return_sequences = False)(X_input)                                 
        X = L.Dropout(0.5)(X)                                 
        X = L.BatchNormalization()(X)                                
        
        X = L.Dense(8)(X)
        X = L.Activation("relu")(X)
        X = L.Dropout(0.5)(X)
        valence_output = L.Dense(1, activation='sigmoid', name = 'valence')(X)
        arousal_output = L.Dense(1, activation='sigmoid', name = 'arousal')(X)
        outputs = [valence_output, arousal_output]
        
        model = Model(inputs = X_input, outputs = outputs)
        model.compile(optimizer='Adadelta',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        
        
        return model