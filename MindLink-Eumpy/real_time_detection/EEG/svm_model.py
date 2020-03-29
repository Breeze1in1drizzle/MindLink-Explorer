# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:52:01 2018

@author: Yongrui Huang
"""

import pandas as pd
import numpy as np
from sklearn.svm.classes import SVC
from sklearn.svm.classes import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class SvmModel():
    '''
        This class is used to apply lstm model to detect valence and 
        arousal. The emotion state value (valence or arousal) ranges from 1 to
        9. We consider it as two category. When it is higher than 5, we take it
        as high category. when it is less than or equal with 5, we take it as
        low category.
        This class also apply the regression method for treating valence and 
        arousal as continuous value and deal with this problem as it is a regression
        problem.
        
        Attributes:
            
            train_X: the list saves all train sample
            
            train_valence: save the ground truth of valence
            
            train_arousal: save the ground truth of arousal
            
            model: the implement model for training and predicting
            
            X_mean: the mean vector for the feature
            
            X_std: the standard deviation vertor for the feature
         
            X_test: the list saves all test sample
            
            test_valence: save the ground truth of valence in test
            
            test_arousal: save the ground truth of arousal in test
    
    '''
    
    def __init__(self):
        self.train_X = []
        self.train_valence = []
        self.train_arousal = []
        # if treat it as classification problem
#        self.model_valence = SVC(C=1.0)
#        self.model_arousal = SVC(C=1.0)
        
        # if deal with it as it is a regression problem
#        self.model_valence = SVR(kernel='linear', C=1)
#        self.model_arousal = SVR(kernel='linear', C=1)
        self.model_valence = SVR(C=100)
        self.model_arousal = SVR(C=100)
        # what are these above? Why SVR(C=100)?     From Ruixin Lee
        
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
  
        for feature in EEGs:
            self.train_X.append(feature)
            # if deal with it as it is a classification problem
#            self.train_valence.append(int(label['valence'] > 5))
#            self.train_arousal.append(int(label['arousal'] > 5))
            # if deal with it as it is a regression problem
            self.train_valence.append(float(label['valence']))
            self.train_arousal.append(float(label['arousal']))
            
    def add_test_data(self, trial_path):
        '''
        add one trial data into model for testing
        
        Arguments:
            
            trial_path: the trial path of the data
        
        '''
        
        label = pd.read_csv(trial_path + 'label.csv')
        EEGs = np.load(trial_path + 'EEG.npy')
  
        for feature in EEGs:
            self.test_X.append(feature)
            
            # if treat it as classification problem
#            self.test_valence.append(int(label['valence'] > 5))
#            self.test_arousal.append(int(label['arousal'] > 5))
            
            # if treat it as regression problem
            self.test_valence.append(float(label['valence']))
            self.test_arousal.append(float(label['arousal']))
            
    def train(self):
        '''
        train the model
        
        '''
        
        self.train_X = np.array(self.train_X)
        self.X_mean = np.mean(self.train_X, axis = 0)
        self.X_std = np.std(self.train_X, axis = 0)
        self.train_X = np.nan_to_num(self.train_X)
       
        # data standardization
        self.train_X -= self.X_mean
        self.train_X /= self.X_std
        
        # train
        self.model_valence.fit(self.train_X, self.train_valence)
        self.model_arousal.fit(self.train_X, self.train_arousal)
    
    def predict(self, X):
        '''
        predict the data using the trained model
        
        Arguments:
            
            X: the data set for predicting.
            shape: (sample number, feature number)
        '''
        
        X -= self.X_mean
        X /= self.X_std
        
        predict_valence = self.model_valence.predict(X)
        predict_arousal = self.model_arousal.predict(X)
        
        return (predict_valence, predict_arousal)

    def evalute(self):
        '''
        evalute the performance
        
        '''
        
        from sklearn.metrics import mean_squared_error
        import math
        self.train_X = np.array(self.train_X)
        (train_predict_valence, train_predict_arousal) =\
            self.predict(self.train_X)
        
        print('Train valence RMSE:%f' %
              math.sqrt(mean_squared_error(train_predict_valence,
                                           self.train_valence)))
        print('Train arousal RMSE:%f' %
              math.sqrt(mean_squared_error(train_predict_arousal,
                                           self.train_arousal)))
        
        self.test_X = np.array(self.test_X)

        (predict_valence, predict_arousal) = self.predict(self.test_X)

        print('Test valence RMSE:%f' %
              math.sqrt(mean_squared_error(predict_valence,
                                           self.test_valence)))
        print('Test arousal RMSE:%f' %
              math.sqrt(mean_squared_error(predict_arousal,
                                           self.test_arousal)))
        print('Predict valence sample: ' + str(predict_valence[:10]))
        print('Predict arousal sample: ' + str(predict_arousal[:10]))

        predict_valence = predict_valence > 5
        predict_arousal = predict_arousal > 5
        self.test_valence = np.array(self.test_valence)
        self.test_arousal = np.array(self.test_arousal)
        self.train_valence = np.array(self.train_valence)
        self.train_arousal = np.array(self.train_arousal)
        self.test_valence = self.test_valence > 5
        self.test_arousal = self.test_arousal > 5
        self.train_valence = self.train_valence > 5
        self.train_arousal = self.train_arousal > 5
        
        # for classification

        print('valence acc: %f, f1: %f' %
               (accuracy_score(predict_valence, self.test_valence), 
                f1_score(predict_valence, self.test_valence)))
        print('arousal acc: %f, f1: %f' %
               (accuracy_score(predict_arousal, self.test_arousal), 
                f1_score(predict_arousal, self.test_arousal)))
        
        print('Train sample proportion: valence: %f, arousal: %f' % (
                float(sum(self.train_valence))/len(self.train_valence), 
                float(sum(self.train_arousal))/len(self.train_arousal)))
        
        print('Test sample proportion: valence: %f, arousal: %f' % (
                float(sum(self.test_valence))/len(self.test_valence), 
                float(sum(self.test_arousal))/len(self.test_arousal)))
        
        print('Train sample shape: ' + str(self.train_X.shape))
        
        print('Test sample shape: ' + str(self.test_X.shape))
