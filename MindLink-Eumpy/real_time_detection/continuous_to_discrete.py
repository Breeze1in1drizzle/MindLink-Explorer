# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:25:25 2018

@author: Yongrui Huang

This script apply 3 different methods to turn continuous emotion (valence and
arousal) to discrete emotion(joy, anger...) based on the self-report from DEAP
dataset.

Method 1:
    Find the nearest position and take it as results, call KNN() method.
    
Method 2:
    use a one-task DNN network to fit it.
    See DNN()
    
Method 3:
    use a multi-task DNN network to fit it.
    Task 1: discrete emotion
    Task 2: the strength of the emotion
    See DNN_multitask()
    
"""

import pandas as pd
import numpy as np
import keras.layers as L
from keras.models import Model
import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys

sys.path.append('../')

import configuration


def plot_training(history):
    '''
        plot the train data image
    '''
    
    output_loss = history.history['loss']
    val_output_loss = history.history['val_loss']
    
    len_epochs = len(output_loss)
    
    output_loss = history.history['loss'][int(len_epochs/2):]
    val_output_loss = history.history['val_loss'][int(len_epochs/2):]
    
    epochs = range(len(output_loss))
    
    plt.figure()
    plt.plot(epochs, output_loss, 'b-', label='train loss')
    plt.plot(epochs,  val_output_loss, 'r-', label='validation loss')
    plt.legend(loc='best')
    plt.title('Training and validation loss')
    plt.show()


def build_model_multitask():
    
    X_input = L.Input(shape=(2,))
    x = L.Dense(64, activation='relu')(X_input)
    x = L.Dropout(0.5)(x)
    
    emotion_x = L.Dense(32, activation='relu')(x)
    emotion = L.Dense(16, activation='softmax', name='emotion')(emotion_x)
    
    strength = L.Dense(1, name = 'strength')(x)
    
    model = Model(inputs = X_input, outputs=[emotion, strength])
    model.compile(optimizer='Adadelta', 
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  metrics=['accuracy'], 
                  loss_weights=[.7, .3])
    
    return model


def DNN_multitask():
    data = pd.read_csv(configuration.DATASET_PATH + 'DEAP/' + 
                       'online_ratings.csv')

    data = data[data['Wheel_slice']!=0]
    
    data = data.sample(frac=1).reset_index(drop=True)
    train_data = data[:int(len(data)*.8)]
    test_data = data[int(len(data)*.8):]

    X = train_data[['Valence', 'Arousal']]
    X = np.array(X)    
    emotion_ground_truth = train_data['Wheel_slice']-1
    strength_ground_truth = train_data['Wheel_strength']
    
    y = []
    y.append(keras.utils.to_categorical(emotion_ground_truth, num_classes = 16))
    y.append(strength_ground_truth)
    
    model = build_model_multitask()
    epoch = 256
    batch_size = 128
    history = model.fit(X, y, verbose=2, validation_split=0.2,
                       epochs=epoch, batch_size=batch_size)
    
    test_X = test_data[['Valence', 'Arousal']]
    test_X = np.array(test_X)
    emotion_ground_truth = np.array(test_data['Wheel_slice']-1)
    strength_ground_truth = test_data['Wheel_strength']

    pre_emotion, pre_strength = model.predict(test_X)
   
    pre_emotion = np.argmax(pre_emotion, axis=1)
    
    unique, counts = np.unique(emotion_ground_truth, return_counts=True)
    counts = np.array(counts)
    
    counts = counts/float(len(test_data))
    print (dict(zip(unique, counts)))
    print (accuracy_score(emotion_ground_truth, pre_emotion))
    print (confusion_matrix(emotion_ground_truth, pre_emotion))
    plot_training(history)
    model.save(configuration.MODEL_PATH + 'continuous_to_discrete.h5')


def build_model():
    X_input = L.Input(shape = (2,))    
    x = L.Dense(64, activation = 'relu')(X_input)
    x = L.Dropout(0.5)(x)
    emotion = L.Dense(16, activation='softmax', name = 'emotion')(x)
   
    model = Model(inputs=X_input, outputs=emotion)
    model.compile(optimizer='Adadelta', 
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])
    return model


def DNN():
    data = pd.read_csv('online_ratings.csv')
    data = data[data['Wheel_slice'] != 0]
    
    data = data.sample(frac=1).reset_index(drop=True)

    train_data = data[:int(len(data)*.8)]
    test_data = data[int(len(data)*.8):]

    X = train_data[['Valence', 'Arousal']]
    X = np.array(X)    
    emotion_ground_truth = np.array(train_data['Wheel_slice']-1)
    
    emotion_ground_truth_vec = keras.utils.to_categorical(emotion_ground_truth, num_classes=16)
    # emotion_ground_truth_vec = emotion_ground_truth_vec.T * strength_ground_truth / 4
    # emotion_ground_truth_vec = emotion_ground_truth_vec.T
    model = build_model()
    epoch = 128
    batch_size = 128
    history = model.fit(X, emotion_ground_truth_vec,
                        verbose=2, validation_split = 0.2, 
                       epochs=epoch, batch_size=batch_size)
    test_X = test_data[['Valence', 'Arousal']]
    test_X = np.array(test_X)
    emotion_ground_truth = np.array(test_data['Wheel_slice']-1)
    pre_emotion_vec = model.predict(test_X)
    
    pre_emotion = np.argmax(pre_emotion_vec, axis = 1)
    
    unique, counts = np.unique(emotion_ground_truth, return_counts=True)
    counts = np.array(counts)
    
    counts = counts/float(len(test_data))
    print(dict(zip(unique, counts)))
    print(accuracy_score(emotion_ground_truth, pre_emotion))
    print(confusion_matrix(emotion_ground_truth, pre_emotion))
    plot_training(history)
    

def distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.linalg.norm(v1-v2)


def KNN():
    data = pd.read_csv('online_ratings.csv')
    data = data[data['Wheel_slice'] != 0]
    
    train_data = data[:int(len(data)*.8)]
    test_data = data[int(len(data)*.8):]
    
    emotion_map = np.zeros(shape = (10, 10, 17))
    
    for index, row in train_data.iterrows():
        valence = row['Valence']
        arousal = row['Arousal']
        wheel_slice = row['Wheel_slice']
        wheel_strength = row['Wheel_strength']
        emotion_map[valence][arousal][wheel_slice] += wheel_strength
    
    acc = 0.
    for idx, row in test_data.iterrows():
        valence = row['Valence']
        arousal = row['Arousal']
        wheel_slice = row['Wheel_slice']
        wheel_strength = row['Wheel_strength']
        predict_emotion_vec = emotion_map[int(valence)][int(arousal)]
        predict_emotion = np.argmax(predict_emotion_vec)
        if predict_emotion == wheel_slice:
            acc += 1
    print(float(acc)/len(test_data))


if __name__ == '__main__':
    print("continuous_to_discrete.py..__main__.start...")
    # DNN_multitask()
    model = keras.models.load_model(configuration.MODEL_PATH + 
                                    'continuous_to_discrete.h5')
    emotion_map = ['Pride', 'Elation', 'Joy', 'Satisfaction', 'Reief', 'Hope',
                   'Interet', 'Surprise', 'Sadness', 'Fear', 'Shame', 'Guilt',
                   'Envy', 'Disgust', 'Contempt', 'Anger']
    for i in range(1, 10):
        for j in range(1, 10):
            X = np.array([[i, j]])
            pre, _ = model.predict(X)
            pre = pre[0]
#            print (pre)
            print(i, j, emotion_map[np.argmax(pre)])
    print("continuous_to_discrete.py..__main__.end...")
