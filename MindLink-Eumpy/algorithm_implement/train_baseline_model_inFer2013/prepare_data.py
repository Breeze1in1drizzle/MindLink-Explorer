# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 08:53:25 2018

@author: Yongrui Huang
"""

import pandas as pd
import numpy as np
from numpy import uint8
import matplotlib.pyplot as plt
from sklearn.model_selection._split import train_test_split


# origin: (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
# new_type: 0=Happy, 1=Sad, 2=Surprise, 3=Neutral
# show_demon_image(img)
def show_demon_image(img):
    print("prepare_data.py..show_demon_image(img).start...")

    '''
    This is actually show_demo_image(img) but not "demon" 
    
    Arguments:
        
        img: numpy array represent an image
    '''
    
    plt.figure("demon")
    plt.imshow(img)
    # plt.axis('off')
    plt.show()

    print("prepare_data.py..show_demon_image(img).end...")
    return 0
# show_demon_image(img)


# load_data(extend_disgust)
def load_data(extend_disgust):
    print("prepare_data.py..load_data(extend_disgust).start...")
    '''
    extract data from 'fer2013.csv' file
    
    Arguments:
    
        extend_digust: whether to extend disgust class

        #--------------------------#
        where is disgust class?      From Ruixin Lee
        #--------------------------#

    return: numpy array -like
        
        train_X:       shape(?,48,48)
        
        validation_X:  shape(?,48,48) 
        
        train_y:       shape(?, )
        
        validation_y:  shape(?, )
    '''
    
    data = pd.read_csv("../../dataset/fer2013/fer2013.csv")####fer2013.csv
    
    X = []
    y = []
    for (pixels, emotion) in zip(data['pixels'], data['emotion']):
        # if emotion == 0 or emotion == 1 or emotion == 2:
        #   continue
        img = np.array((pixels.split(' ')), dtype=uint8)
        img = img.reshape((48, 48))
        # img = cv2.equalizeHist(img)
        y.append(emotion)
        X.append(img)
    
    if extend_disgust:
        # extend disgust facial expression data,
        # in order to overcome the problem that class 'disgust' has much less sample than other class.

        # It seems that "disgust" is more difficult to detect,
        # there might be some class like disgust in other package.      From Ruixin Lee
        disgust_image = np.load('../../dataset/fer2013/extend_disgust.npy')####extend_disgust.npy
        X.extend(disgust_image)
        y.extend(np.ones((len(disgust_image),)))
    
    X = np.array(X, dtype=uint8)
    y = np.array(y, dtype=uint8)
    X = X.astype('float32')

    train_X, validation_X, train_y, validation_y = \
        train_test_split(X, y, test_size=0.2, random_state=0)

    print("prepare_data.py..load_data(extend_disgust).end...")
    return train_X, validation_X, train_y, validation_y
# load_data(extend_disgust)


# __main__
if __name__ == '__main__':
    print("prepare_data.py..__main__.start...")

    train_X, validation_X, train_y, validation_y = load_data(extend_disgust = True)
    
    # save data for quicker loading
    np.save("../../dataset/fer2013/train_X.npy", train_X)
    np.save("../../dataset/fer2013/train_y.npy", train_y)
    np.save("../../dataset/fer2013/validation_X.npy", validation_X)
    np.save("../../dataset/fer2013/validation_y.npy", validation_y)
    
    # save mean for normalization
    X_mean = np.mean(train_X, axis = 0)
    np.save("../../dataset/fer2013/X_mean.npy", X_mean)

    print("prepare_data.py..__main__.end...")
# __main__
