# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:52:01 2018
Modified on Sun Mar 14 00:26:35 2021

@author0: Yongrui Huang
@author1: Ruixin Lee
"""

import pandas as pd
import numpy as np
import glob
from sklearn.svm import SVC
from sklearn.svm import SVR
# from sklearn.svm.classes import SVC
# from sklearn.svm.classes import SVR
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import xml.dom.minidom
import os


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

        # label = pd.read_csv(trial_path + 'label.csv')
        trial_path = trial_path.replace('\\', '/') + r'/'
        # print("trail path: ", trial_path)
        dom = xml.dom.minidom.parse(trial_path + 'session.xml')
        session = dom.documentElement
        # arousal = float(session.getAttribute("feltArsl"))

        # label处理，把非数字去除
        import re
        arousal = session.getAttribute("feltArsl")
        print("arousal: ", arousal, "  (type: ", type(arousal), "), (len: ", len(arousal), ").")
        # for word in arousal:
        #     if (ord(word) < 48 or ord(word) > 57):
        #         arousal = arousal.replace(word, '')

        valence = session.getAttribute("feltVlnc")
        print("valence: ", valence, "  (type: ", type(valence), "), (len: ", len(valence), ").")
        # for word in valence:
        #     if (ord(word) < 48 or ord(word) > 57):
        #         valence = valence.replace(word, '')

        Eyes = np.load(trial_path + 'Eye.npy')

        for feature in Eyes:
            self.train_X.append(feature)
            # if deal with it as it is a classification problem
            #            self.train_valence.append(int(label['valence'] > 5))
            #            self.train_arousal.append(int(label['arousal'] > 5))
            # if deal with it as it is a regression problem
            self.train_valence.append(float(valence))
            self.train_arousal.append(float(arousal))

    def add_test_data(self, trial_path):
        '''
        add one trial data into model for testing

        Arguments:

            trial_path: the trial path of the data

        '''

        # label = pd.read_csv(trial_path + 'label.csv')
        trial_path = trial_path.replace('\\', '/') + r'/'
        # print("trail path: ", trial_path)
        dom = xml.dom.minidom.parse(trial_path + 'session.xml')
        session = dom.documentElement
        arousal = session.getAttribute("feltArsl")
        # for word in arousal:
        #     if (ord(word) < 48 or ord(word) > 57):
        #         arousal = arousal.replace(word, '')
        # print("arousal: ", arousal, "  (type: ", type(arousal), ")")
        valence = session.getAttribute("feltVlnc")
        # for word in valence:
        #     if (ord(word) < 48 or ord(word) > 57):
        #         valence = valence.replace(word, '')
        # print("valence: ", valence, "  (type: ", type(valence), ")")

        Eyes = np.load(trial_path + 'Eye.npy')

        for feature in Eyes:
            self.test_X.append(feature)

            # if treat it as classification problem
            #            self.test_valence.append(int(label['valence'] > 5))
            #            self.test_arousal.append(int(label['arousal'] > 5))

            # if treat it as regression problem
            self.test_valence.append(float(valence))
            self.test_arousal.append(float(arousal))

    def train(self, out_subject=0, save_path=''):
        '''
        train the model

        '''

        self.train_X = np.array(self.train_X)
        self.X_mean = np.mean(self.train_X, axis=0)
        self.X_std = np.std(self.train_X, axis=0)
        self.train_X = np.nan_to_num(self.train_X)

        # data standardization
        self.train_X -= self.X_mean
        self.train_X /= self.X_std
        print("train_X: ", self.train_X.shape)
        # train
        print("start.valence.fit...")
        self.model_valence.fit(self.train_X, self.train_valence)
        print("valence completed!\nstart.arousal.fit...")
        self.model_arousal.fit(self.train_X, self.train_arousal)
        print("arousal completed!")

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

    def evalute(self, out_subject=0, save_path=''):
        '''
        evalute the performance

        '''
        import pandas as pd
        from sklearn.metrics import mean_squared_error
        import math

        save_path = save_path + 'subject_' + str(out_subject) + '/'

        self.train_X = np.array(self.train_X)
        (train_predict_valence, train_predict_arousal) = \
            self.predict(self.train_X)

        train_valence_RMSE = math.sqrt(mean_squared_error(train_predict_valence, self.train_valence))
        train_arousal_RMSE = math.sqrt(mean_squared_error(train_predict_arousal, self.train_arousal))
        print('Train valence RMSE:%f' % train_valence_RMSE)
        print('Train arousal RMSE:%f' % train_arousal_RMSE)

        self.test_X = np.array(self.test_X)

        (predict_valence, predict_arousal) = self.predict(self.test_X)

        test_valence_RMSE = math.sqrt(mean_squared_error(predict_valence, self.test_valence))
        test_arousal_RMSE = math.sqrt(mean_squared_error(predict_arousal, self.test_arousal))
        predict_valence_sample = str(predict_valence[:10])
        predict_arousal_sample = str(predict_arousal[:10])
        print('Test valence RMSE:%f' % test_valence_RMSE)
        print('Test arousal RMSE:%f' % test_arousal_RMSE)
        print('Predict valence sample: ' + predict_valence_sample)
        print('Predict arousal sample: ' + predict_arousal_sample)

        dict1 = {
            "train_valence_RMSE": train_valence_RMSE,
            "train_arousal_RMSE": train_arousal_RMSE,
            "test_valence_RMSE": test_valence_RMSE,
            "test_arousal_RMSE": test_arousal_RMSE,
            "predict_valence_sample": predict_valence_sample,
            "predict_arousal_sample": predict_arousal_sample
        }
        df1 = pd.DataFrame(data=dict1, index=[0])

        df1.to_csv(save_path + "statistic_regression.csv", index=False)

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
        acc_valence = accuracy_score(predict_valence, self.test_valence)
        f1_valence = f1_score(predict_valence, self.test_valence)
        acc_arousal = accuracy_score(predict_arousal, self.test_arousal)
        f1_arousal = f1_score(predict_arousal, self.test_arousal)
        print('valence acc: %f, f1: %f' % (acc_valence, f1_valence))
        print('arousal acc: %f, f1: %f' % (acc_arousal, f1_arousal))

        train_sample_proportion_valence = float(sum(self.train_valence)) / len(self.train_valence)
        train_sample_proportion_arousal = float(sum(self.train_arousal)) / len(self.train_arousal)
        print('Train sample proportion: valence: %f, arousal: %f' % (
            train_sample_proportion_valence, train_sample_proportion_arousal))

        test_sample_proportion_valence = float(sum(self.test_valence)) / len(self.test_valence)
        test_sample_proportion_arousal = float(sum(self.test_arousal)) / len(self.test_arousal)
        print('Test sample proportion: valence: %f, arousal: %f' % (
            test_sample_proportion_valence, test_sample_proportion_arousal))

        train_sample_shape = str(self.train_X.shape)
        test_sample_shape = str(self.test_X.shape)
        print('Train sample shape: ' + train_sample_shape)
        print('Test sample shape: ' + test_sample_shape)

        dict2 = {
            "acc_valence": acc_valence,
            "f1_valence": f1_valence,
            "acc_arousal": acc_arousal,
            "f1_arousal": f1_arousal,
            "train_sample_proportion_valence": train_sample_proportion_valence,
            "train_sample_proportion_arousal": train_sample_proportion_arousal,
            "test_sample_proportion_valence": test_sample_proportion_valence,
            "test_sample_proportion_arousal": test_sample_proportion_arousal,
            "train_sample_shape": train_sample_shape,
            "test_sample_shape": test_sample_shape
        }
        df2 = pd.DataFrame(data=dict1, index=[0])

        df2.to_csv(save_path + "statistic_classification.csv", index=False)


def leave_one_subject_out_validation():
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*'
    print('model = LstmModelRegression()')
    model = SvmModel()

    dirPath = glob.iglob(dataset_path)  # 有了'/*'才可以遍历*里面的东西，没有就只能在文件夹外面
    # path = dataset_path.replace('/*', '')

    num_of_folder = 0
    # num_of_fif = 0
    # num_of_success_fif = 0
    subject_id = 0
    for i in range(0, 39):
        # i 作为leave-one-subject-out validation的测试集的那个被试的id
        print("Iteration ", i, ".")
        if i == 32 or i == 14 or i == 18:
            continue
        for big_file in dirPath:
            num_of_folder += 1
            print("num_of_folder: ", num_of_folder)
            files = os.listdir(big_file)
            print("big_file: ", big_file)
            # print("files: ", files)
            folder_num = big_file[len(dataset_path) - 1:]  # 获取相关数字
            subject_id = int(int(folder_num) / 100)
            if (subject_id == 14 or subject_id == 18 or subject_id ==32):
                continue
            for file in files:
                # print("file: ", file)
                if file.endswith("Eye.npy"):
                    # print("big_file: ", big_file)
                    if subject_id == i:
                        # print("subject ", subject_id, ", add_test_data")
                        model.add_test_data(big_file)
                    else:
                        # print("subject ", subject_id, ", add_train_data")
                        model.add_train_data(big_file)
        print("train.....")
        model.train(out_subject=i, save_path='C:/Users/ps/Desktop/experiments/leave-one-subject-out_validation/Eye-mahnob/')
        print("evaluate.....")
        model.evalute(out_subject=i, save_path='C:/Users/ps/Desktop/experiments/leave-one-subject-out_validation/Eye-mahnob/')
        print("evaluate.completed!")


def normal_train():
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*'
    print('model = LstmModelRegression()')
    model = SvmModel()

    dirPath = glob.iglob(dataset_path)  # 有了'/*'才可以遍历*里面的东西，没有就只能在文件夹外面
    # path = dataset_path.replace('/*', '')

    num_of_folder = 0
    # num_of_fif = 0
    # num_of_success_fif = 0
    subject_id = 0

    # [0,]
    i = 0   # 去除掉的subject的id号码

    for big_file in dirPath:
        num_of_folder += 1
        print("num_of_folder: ", num_of_folder)
        files = os.listdir(big_file)
        # print("big_file: ", big_file)
        # print("files: ", files)
        folder_num = big_file[len(dataset_path) - 1:]  # 获取相关数字
        subject_id = int(int(folder_num) / 100)
        if (subject_id == 14 or subject_id == 18 or subject_id == 32):
            continue
        for file in files:
            print("file: ", file)
            if file.endswith("Eye.npy"):
                # print("big_file: ", big_file)
                if subject_id == i:
                    print("subject ", subject_id, ", add_test_data")
                    model.add_test_data(big_file)
                else:
                    print("subject ", subject_id, ", add_train_data")
                    model.add_train_data(big_file)
    print("train.....")
    model.train(out_subject=i,
                save_path='C:/Users/ps/Desktop/experiments/leave-one-subject-out_validation/Eye-mahnob/')
    print("evaluate.....")
    model.evalute(out_subject=i,
                    save_path='C:/Users/ps/Desktop/experiments/leave-one-subject-out_validation/Eye-mahnob/')
    print("evaluate completed")


if __name__ == "__main__":
    import time
    start = time.time()
    # leave_one_subject_out_validation()
    normal_train()
    end = time.time()
    print("耗时：%.2f秒" % (end-start))
