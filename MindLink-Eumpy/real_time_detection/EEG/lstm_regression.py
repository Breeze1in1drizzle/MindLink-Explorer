# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:21:00 2018

@author: Yongrui Huang
"""

import keras.layers as L
from keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
from keras import regularizers
import configuration
import matplotlib.pyplot as plt

def plot_valence_training(history, filename = 'valence_EEG'):
    '''
        polt the train data image
    '''

    output_loss = history.history['valence_loss']
    val_output_loss = history.history['val_valence_loss']
    
    epochs = range(len(output_loss))
    
    plt.figure()
    plt.plot(epochs, output_loss, 'b-', label='valence train loss')
    plt.plot(epochs,  val_output_loss, 'r-', label='valence validation loss')
    plt.legend(loc='best')
    plt.title('Valence training and validation loss')
    plt.savefig(filename+'_loss' + '.png')
#     plt.show()
    
def plot_arousal_training(history, filename = 'arousal_EEG'):
    '''
        polt the train data image
    '''

    output_loss = history.history['arousal_loss']
    val_output_loss = history.history['val_arousal_loss']
    
    epochs = range(len(output_loss))
    
    plt.figure()
    plt.plot(epochs, output_loss, 'b-', label='arousal train loss')
    plt.plot(epochs,  val_output_loss, 'r-', label='arousal validation loss')
    plt.legend(loc='best')
    plt.title('Arousal training and validation loss')
    plt.savefig(filename+'_loss' + '.png')
#     plt.show()
    
class LstmModelRegression():
    '''
        This class is used to apply lstm model to predict the valence and 
        arousal. The emotion state value (valence or arousal) ranges from 1 to
        9. We consider it as a continuous value and treat this preblem as a 
        regression problem.
        
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
        
        total_len = len(EEGs)
        for i in np.arange(1, total_len, 5):
            if i + 10 > total_len:
                break
            feature = EEGs[i:i+10]
            self.test_X.append(feature)
            
            self.test_valence.append(float(label['valence']))
            self.test_arousal.append(float(label['arousal']))
    
    def build_model(self, input_shape):
        '''
        build the model
        
        Arguments:
            
            input_shape: the input shape of the sample.
        
        '''
  
        X_input = L.Input(shape = input_shape)                             
                                          
        X = L.LSTM(128, return_sequences = True, recurrent_dropout = 0.5)(X_input) 
        X = L.Activation("relu")(X)                               
        X = L.Dropout(0.5)(X)               
        X = L.BatchNormalization()(X)   
                
        X = L.LSTM(64, return_sequences = False, recurrent_dropout = 0.5)(X) 
        X = L.Activation("relu")(X)                               
        X = L.Dropout(0.5)(X)                                 
        X = L.BatchNormalization()(X)                            
               
        X = L.Dense(64, kernel_regularizer=regularizers.l2(1))(X)
        X = L.Activation("relu")(X)
        X = L.Dropout(0.5)(X)
        
        valence_output = L.Dense(1, name = 'valence')(X)
        arousal_output = L.Dense(1, name = 'arousal')(X)
        outputs = [valence_output, arousal_output]
        
        model = Model(inputs = X_input, outputs = outputs)

        model.compile(optimizer='Adadelta', loss='mean_squared_error',
                  metrics=[], loss_weights=[1., 1.])
        
        
        return model
    
    def train(self):
        
        '''
        train the model
        
        '''
        '''
        self.train_X = self.train_X[:500]
        self.train_valence = self.train_valence[:500]
        self.train_arousal = self.train_arousal[:500]
        '''
        self.train_X = np.array(self.train_X)
        self.X_mean = np.mean(self.train_X, axis = 0)
        self.X_std = np.std(self.train_X, axis = 0)
        
        #data standardization
        self.train_X -= self.X_mean
        self.train_X /= self.X_std
        self.test_X -= self.X_mean
        self.test_X /= self.X_std
        #train
        self.model = self.build_model((self.train_X.shape[1], 
                                       self.train_X.shape[2]))
        epoch = 64
        batch_size = 256
        history = self.model.fit(self.train_X, [self.train_valence, self.train_arousal], 
                       verbose=2, validation_data = (self.test_X, [self.test_valence, self.test_arousal]),
                       epochs=epoch, batch_size=batch_size)
        
        plot_valence_training(history)
        plot_arousal_training(history)
        
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
        
        return (valence_scores, arousal_scores)
    
    def evalute(self):
        '''
        evalute the performance
        
        '''
        
        self.test_X = np.array(self.test_X)
        (predict_valence, predict_arousal) = self.model.predict(self.test_X)
        
        #for regression
        from sklearn.metrics import mean_squared_error
        import math
        print ('valence RMSE:%f' % 
               math.sqrt(mean_squared_error(predict_valence, self.test_valence)))
        print ('arousal RMSE:%f' %
               math.sqrt(mean_squared_error(predict_arousal, self.test_arousal)))
        
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
        
        #for classification
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
        
        self.model.save(configuration.MODEL_PATH + 'LSTM_EEG_regression.h5')
        np.save(configuration.MODEL_PATH + 'EEG_mean.npy', self.X_mean)
        np.save(configuration.MODEL_PATH + 'EEG_std.npy', self.X_std)
        
if __name__ == '__main__':
    
    root_path = configuration.DATASET_PATH + 'MAHNOB_HCI/'  
    
    model = LstmModelRegression()
    for subject_id in range(1, 24):
        subject_path = root_path + str(subject_id) + '/'
        
        for trial_id in range(1, 21):
            trial_path = '%s/trial_%d/'%(subject_path, trial_id)
            model.add_train_data(trial_path)
                
    
    
    for subject_id in range(20, 25):
        subject_path = root_path + str(subject_id) + '/'
        
        for trial_id in range(1, 21):
            trial_path = '%s/trial_%d/'%(subject_path, trial_id)
            model.add_test_data(trial_path)
    
    model.train()
    model.evalute()