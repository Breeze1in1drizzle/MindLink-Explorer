# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:11:56 2018

@author: Yongrui Huang
"""

import numpy as np
import keras
import keras.layers as L
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sys
sys.path.append('../../')
import configuration
import matplotlib.pyplot as plt

def plot_valence_training(history, filename = 'valence_facial_expression'):
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


def plot_arousal_training(history, filename = 'arousal_facial_expression'):
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


def setup_to_finetune(model, freeze_num = 0):
    '''
        fix the convolution layer and prepare for the training of dense layer
        
        Arguments:
        
            model: the baseline model
            
            freeze_num: the number of the layers need to be fixed
    '''
    for layer in model.layers[:freeze_num]:
        layer.trainable = False
    for layer in model.layers[freeze_num:]:
        layer.trainable = True
     
    model.compile(optimizer='Adadelta', loss='mean_squared_error',
                  metrics=['accuracy'], loss_weights=[1., 1.])


def get_pretrain_model():
    '''
      load the baseline model and return the new model
      
      Return:
      
          new_model: new model which the convolution layers are fixed.
    '''
    
    old_model = keras.models.load_model(configuration.MODEL_PATH + 'CNN_expression_baseline.h5')
    feature_layer = old_model.get_layer('bottleneck').output
    x = L.Dense(32, activation='relu')(feature_layer)
    x = L.Dropout(0.5, name='new_fucking_dropout')(x)
    valence_output = L.Dense(1, name='valence')(x)
    arousal_output = L.Dense(1, name='arousal')(x)
    new_model = keras.Model(inputs=old_model.inputs, outputs=[valence_output, arousal_output])
    setup_to_finetune(new_model, freeze_num=3)
    return new_model


class CNNModel():
    '''
     This class is used to apply CNN model to predict the valence and 
     arousal. The emotion state value (valence or arousal) ranges from 1 to
     9. We consider it as a continuous value and treat this problem as a
     regression problem.
     
     Attributes:
            
            train_X: the list saves all train sample
            
            train_valence: save the ground truth of valence
            
            train_arousal: save the ground truth of arousal
            
            model: the implement model for training and predicting
                    
            X_test: the list saves all test sample
            
            test_valence: save the ground truth of valence in test
            
            test_arousal: save the ground truth of arousal in test
    
        
    '''
    
    def __init__(self):
        self.train_X = []
        self.train_valence = []
        self.train_arousal = []
        
        self.model = get_pretrain_model()
        
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
        faces = np.load(trial_path + 'faces.npy')
  
        for feature in faces:
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
        faces = np.load(trial_path + 'faces.npy')
  
        for feature in faces:
            self.test_X.append(feature)
            self.test_valence.append(float(label['valence']))
            self.test_arousal.append(float(label['arousal']))   
    
    def train(self):
        '''
        train the model
        
        '''
        '''
        self.train_X = self.train_X[:500]
        self.train_valence = self.train_valence[:500]
        self.train_arousal = self.train_arousal[:500]
        '''
        
        print(len(self.train_X))
        self.train_X = np.array(self.train_X)
        self.X_mean = np.mean(self.train_X, axis = 0)
        self.X_std = np.std(self.train_X, axis = 0)
        self.train_X = np.nan_to_num(self.train_X)
        
        #format data
        self.train_X = format_raw_images_data(self.train_X)
        
        # train model
        epochs = 64
        batch_size = 256
        history = self.model.fit(self.train_X, [self.train_valence, self.train_arousal], 
                       epochs=epochs, batch_size=batch_size, verbose=2,
                       validation_split=0.2)
        plot_valence_training(history)
        plot_arousal_training(history)
        
    def predict(self, X):
        '''
        predict the data using the trained model
        
        Arguments:
            
            X: the data set for predicting.
            shape: (sample number, feature number)
        '''
        
        X = format_raw_images_data(X)
        
        (valence_scores, arousal_scores) = self.model.predict(X)
        
        return (valence_scores, arousal_scores)
    
    def evalute(self):
        '''
        evalute the performance
        
        '''
        
        from sklearn.metrics import mean_squared_error
        import math
        
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
        
        self.model.save(configuration.MODEL_PATH + 'CNN_face_regression.h5')


if __name__ == '__main__':
    
    root_path = configuration.DATASET_PATH + 'MAHNOB_HCI/'  
    
    model = CNNModel()
   
    for subject_id in range(1, 20):
        subject_path = root_path + str(subject_id) + '/'
        
        for trial_id in range(1, 21):
            trial_path = '%s/trial_%d/' % (subject_path, trial_id)
            model.add_train_data(trial_path)

    model.train()

    for subject_id in range(20, 25):
        subject_path = root_path + str(subject_id) + '/'
        
        for trial_id in range(1, 21):
            trial_path = '%s/trial_%d/'%(subject_path, trial_id)
            model.add_test_data(trial_path)

    model.evalute()
