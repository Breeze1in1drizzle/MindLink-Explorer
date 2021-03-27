# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:21:00 2018
Modified on Fri Mar 12 19:54:49 2021

@author0: Yongrui Huang
@author1: Ruixin Lee
@author2: Xiaojian Liu
"""

import sys

import keras.layers as L
import numpy as np
from keras.models import Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

sys.path.append('../../')
from keras import regularizers
# import configuration
import matplotlib.pyplot as plt
import glob
import os
import xml.dom.minidom


def plot_valence_training(history, filename='valence_EEG'):
    '''
        plot the training data image
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
    # plt.show()


def plot_arousal_training(history, filename='arousal_EEG'):
    '''
        plot the train data image
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
    # plt.show()


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

        # label = pd.read_csv(trial_path + 'label.csv')
        trial_path = trial_path.replace('\\', '/') + r'/'
        print("trail path: ", trial_path)
        dom = xml.dom.minidom.parse(trial_path+'session.xml')
        session = dom.documentElement
        arousal = float(session.getAttribute("feltArsl"))
        # print("arousal: ", arousal, "  (type: ", type(arousal), ")")
        valence = session.getAttribute("feltVlnc")
        # print("valence: ", valence, "  (type: ", type(valence), ")")
        EEGs = np.load(trial_path + 'EEG.npy')

        total_len = len(EEGs)
        for i in np.arange(1, total_len, 5):
            if i + 10 > total_len:
                break
            feature = EEGs[i:i+10]      # 选择特征值
            self.train_X.append(feature)
            
            # self.train_valence.append(float(label['valence']))
            self.train_valence.append(float(valence))
            # self.train_arousal.append(float(label['arousal']))
            self.train_arousal.append(float(arousal))

    def add_test_data(self, trial_path):
        '''
        add one trial data into model for testing
        Arguments:
            trial_path: the trial path of the data
        '''
        # label = pd.read_csv(trial_path + 'label.csv')
        trial_path = trial_path.replace('\\', '/') + r'/'
        print("trail path: ", trial_path)
        dom = xml.dom.minidom.parse(trial_path + 'session.xml')
        session = dom.documentElement
        arousal = float(session.getAttribute("feltArsl"))
        # print("arousal: ", arousal, "  (type: ", type(arousal), ")")
        valence = session.getAttribute("feltVlnc")
        # print("valence: ", valence, "  (type: ", type(valence), ")")
        EEGs = np.load(trial_path + 'EEG.npy') 
        
        total_len = len(EEGs)
        for i in np.arange(1, total_len, 5):
            if i + 10 > total_len:
                break
            feature = EEGs[i:i+10]
            self.test_X.append(feature)
            
            # self.test_valence.append(float(label['valence']))
            self.test_valence.append(float(valence))
            # self.test_arousal.append(float(label['arousal']))
            self.test_arousal.append(float(arousal))
    
    def build_model(self, input_shape):
        '''
        build the model
        Arguments:
            input_shape: the input shape of the sample.
        '''
        X_input = L.Input(shape=input_shape)
                                          
        X = L.LSTM(128, return_sequences=True, recurrent_dropout=0.5)(X_input)
        X = L.Activation("relu")(X)                               
        X = L.Dropout(0.5)(X)               
        X = L.BatchNormalization()(X)
                
        X = L.LSTM(64, return_sequences=False, recurrent_dropout=0.5)(X)
        X = L.Activation("relu")(X)
        X = L.Dropout(0.5)(X)
        X = L.BatchNormalization()(X)
               
        X = L.Dense(64, kernel_regularizer=regularizers.l2(1))(X)
        X = L.Activation("relu")(X)
        X = L.Dropout(0.5)(X)
        
        valence_output = L.Dense(1, name='valence')(X)
        arousal_output = L.Dense(1, name='arousal')(X)
        outputs = [valence_output, arousal_output]
        
        model = Model(inputs=X_input, outputs=outputs)

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
        print("train_X: ", self.train_X.shape)

        self.X_mean = np.mean(self.train_X, axis=0)     # train_X训练集求均值？
        self.X_std = np.std(self.train_X, axis=0)
        
        # data standardization
        self.train_X -= self.X_mean
        self.train_X /= self.X_std
        # print(self.test_X)
        # print(self.X_mean)
        self.test_X -= self.X_mean
        self.test_X /= self.X_std

        print("self.train_X.shape[1]", self.train_X.shape[1])
        print("self.train_X.shape[2]", self.train_X.shape[2])
        # train
        self.model = self.build_model(
            (self.train_X.shape[1], self.train_X.shape[2])
        )
        # epoch = 64
        epoch = 128
        batch_size = 256
        history = self.model.fit(self.train_X, [self.train_valence, self.train_arousal], 
                       verbose=2, validation_data=(self.test_X, [self.test_valence, self.test_arousal]),
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
        
        # for regression
        from sklearn.metrics import mean_squared_error
        import math
        print('valence RMSE:%f' %
               math.sqrt(mean_squared_error(predict_valence, self.test_valence)))
        print('arousal RMSE:%f' %
               math.sqrt(mean_squared_error(predict_arousal, self.test_arousal)))

        # 这里进行了分类处理
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
        
        # self.model.save(sconfiguration.MODEL_PATH + 'LSTM_EEG_regression.h5')
        self.model.save('D:/myworkspace/mypyworkspace/MindLink-Eumpy/LSTM_EEG_regression.h5')
        np.save('D:/myworkspace/mypyworkspace/MindLink-Eumpy/EEG_mean.npy', self.X_mean)
        np.save('D:/myworkspace/mypyworkspace/MindLink-Eumpy/EEG_std.npy', self.X_std)


def leave_one_subject_out_validation():
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*'
    print('model = LstmModelRegression()')
    model = LstmModelRegression()

    dirPath = glob.iglob(dataset_path)  # 有了'/*'才可以遍历*里面的东西，没有就只能在文件夹外面
    # path = dataset_path.replace('/*', '')

    num_of_folder = 0
    # num_of_fif = 0
    # num_of_success_fif = 0
    for i in range(0, 39):
        # i 作为leave-one-subject-out validation的测试集的那个被试的id
        print("Iteration ", i, ".")
        if i == 32:
            continue
        for big_file in dirPath:
            num_of_folder += 1
            print("num_of_folder: ", num_of_folder)
            files = os.listdir(big_file)
            print("big_file: ", big_file)
            print("files: ", files)
            for file in files:
                print("file: ", file)
                if file.endswith("EEG.npy"):
                    # print("big_file: ", big_file)
                    folder_num = big_file[len(dataset_path) - 1:]  # 获取相关数字
                    subject_id = int(int(folder_num) / 100)
                    if subject_id == i:
                        print("subject ", subject_id, ", add_test_data")
                        model.add_test_data(big_file)
                    else:
                        print("subject ", subject_id, ", add_train_data")
                        model.add_train_data(big_file)
        print("train.....")
        model.train()
        print("evaluae.....")
        model.evalute()


def main():
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*'
    model = LstmModelRegression()

    dirPath = glob.iglob(dataset_path)  # 有了'/*'才可以遍历*里面的东西，没有就只能在文件夹外面
    path = dataset_path.replace('/*', '')

    num_of_folder = 0

    for big_file in dirPath:
        num_of_folder += 1
        print("num_of_folder: ", num_of_folder)
        files = os.listdir(big_file)
        folder_num = big_file[len(dataset_path) - 1:]  # 获取相关数字
        subject_id = int(int(folder_num) / 100)
        if (subject_id == 14 or subject_id == 18 or subject_id == 32):
            continue
        for file in files:
            if file.endswith("EEG.npy"):
                if(0 <= subject_id) and (subject_id <= 37):
                    model.add_train_data(big_file)
                if (28 <= subject_id) and (subject_id <= 38):  # 35->38
                    model.add_test_data(big_file)

    print("train.....")
    model.train()
    print("evaluae.....")
    model.evalute()


def normal_train_subject_independent_once():
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*'
    dirPath = glob.iglob(dataset_path)  # 有了'/*'才可以遍历*里面的东西，没有就只能在文件夹外面

    model = LstmModelRegression()

    num_of_folder = 0
    subject_id = 0

    for big_file in dirPath:
        num_of_folder += 1
        print("num_of_folder: ", num_of_folder)
        files = os.listdir(big_file)

        folder_num = big_file[len(dataset_path) - 1:]  # 获取相关数字
        subject_id = int(int(folder_num) / 100)
        if (subject_id == 14 or subject_id == 18 or subject_id == 32):
            continue
        for file in files:
            if file.endswith("EEG.npy"):
                # print("big_file: ", big_file)
                if (0 <= subject_id) and (subject_id <= 37):
                    model.add_train_data(big_file)
                if (28 <= subject_id) and (subject_id <= 38):     # 35->38
                    model.add_test_data(big_file)
    # print("train.....")
    # model.train(out_subject=233,
    #             save_path='C:/Users/ps/Desktop/experiments/leave-one-subject-out_validation/FLF-mahnob/')
    # print("evaluae.....")
    # model.evalute(out_subject=233,
    #               save_path='C:/Users/ps/Desktop/experiments/leave-one-subject-out_validation/FLF-mahnob/')


if __name__ == '__main__':
    import time

    start = time.time()
    main()
    # leave_one_subject_out_validation()
    end = time.time()
    print("耗时：%.2f秒" % (end - start))