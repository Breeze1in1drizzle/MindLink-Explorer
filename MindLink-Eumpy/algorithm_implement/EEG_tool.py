# -*- coding: utf-8 -*-
"""
Created on Tue Oct 2 12:55:02 2018

@author: Yongrui Huang
"""

from sklearn.svm.classes import SVC
import numpy as np
import pandas as pd
import mne


# extract_EEG_feature(raw_EEG_obj)
def extract_EEG_feature(raw_EEG_obj):
    print("EEG_tool.py..extract_EEG_feature(raw_EEG_obj).start...")
    '''
        Extract PSD feature from raw EEG data
        
        :argument:
        
            raw_EEG_obj: raw objects from MNE library

            How raw_EEG_obj construct its data? How to understand this class.       From Ruixin Lee
            raw_EEG_obj = mne.io.read_raw_fif(trial_path + 'EEG.raw.fif', preload=True, verbose='ERROR')
            We get raw_EEG_obj by this method above.

        
        :returns:
        
            average_feature: extracted feature
    
    '''
    
    # select 14 electrodes
    EEG_raw = raw_EEG_obj.pick_channels(
        [
            'Fp1', 'T7', 'CP1', 'Oz',
            'Fp2', 'F8', 'FC6', 'FC2',
            'Cz',  'C4', 'T8',  'CP6',
            'CP2', 'PO4'
        ]
    )
    EEG_data_frame = EEG_raw.to_data_frame()
    
    # calculate three symmetric pairs
    EEG_data_frame['T7-T8'] = EEG_data_frame['T7'] - EEG_data_frame['T8']
    EEG_data_frame['Fp1-Fp2'] = EEG_data_frame['Fp1'] - EEG_data_frame['Fp2']
    EEG_data_frame['CP1-CP2'] = EEG_data_frame['CP1'] - EEG_data_frame['CP2']
    
    # extract PSD feature from different frequency
    EEG_raw_numpy = np.array(EEG_data_frame).T
    EEG_theta = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=4, h_freq=8, verbose='ERROR')
    EEG_slow_alpha = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=8, h_freq=10, verbose='ERROR')
    EEG_alpha = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=8, h_freq=12, verbose='ERROR')
    EEG_beta = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=12, h_freq=30, verbose='ERROR')
    EEG_gamma = mne.filter.filter_data(EEG_raw_numpy, sfreq=256, l_freq=30, h_freq=45, verbose='ERROR')
    
    # concat them together
    features = np.concatenate((EEG_theta, EEG_slow_alpha, EEG_alpha, EEG_beta, EEG_gamma), axis=0)
    
    # get average in each second for decreasing noise and reduce the number of samples for quicker training.
    left_idx = 0
    len_features = features.shape[1]
    features_list = []
    while left_idx < len_features:
        sub_features = features[:, left_idx:left_idx+256] if left_idx+256 < len_features else features[:, left_idx:]
        features_list.append(np.average(sub_features, axis=1))
        left_idx += 256
    average_feature = np.array(features_list)

    print("EEG_tool.py..extract_EEG_feature(raw_EEG_obj).end...")
    return average_feature
# extract_EEG_feature(raw_EEG_obj)


# class EEGModel
class EEGModel:
    '''
        This class allow EEG model become an independent model like facial 
        expression model rather than two separated model.
        
        Attributes:
            
            valence_model: model for classifying valence
            
            arousal_model: model for classifying arousal
            
            X: the list that saves all EEGs features
            
            y_valence: the valence label list, ground true
            
            y_arousal: the arousal label list, ground true
            
            mean: the mean matrix for train data
            
            std: the std matrix for train data
    '''

    # __init__(self)
    def __init__(self):
        print("EEGModel.__init__.start...")

        self.valence_model = SVC(C=1.0)
        self.arousal_model = SVC(C=1.0)
        self.X = []
        self.y_valence = []
        self.y_arousal = []
        self.mean = None
        self.std = None

        print("EEGModel.__init__.end...")
        return 0
    # __init__(self)

    # standardization(self, X)
    def standardization(self, X):
        print("EEGModel.standardization(self, X).start...")
        '''
            This method takes a matrix for input, and output the matrix after 
            standardization.
            
            Arguments:
                
                X: a matrix
                    
            Returns:
                
                X: a matix with standardization
        '''
        if self.mean is None and self.std is None:
            self.X = np.array(self.X)
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0)
        X -= self.mean
        X /= self.std

        print("EEGModel.standardization(self, X).end...")
        return X
    # standardization(self, X)

    # train(self)
    def train(self):
        print("EEGModel.train(self).start...")
        '''
            train valence_model and arousal_model using EEG data
        '''
        self.X = self.standardization(self.X)
        self.valence_model.fit(self.X, self.y_valence)
        self.arousal_model.fit(self.X, self.y_arousal)

        print("EEGModel.train(self).end...")
        return 0
    # train(self)

    # add_one_trial_data(self, trial_path, preprocessed = False)
    def add_one_trial_data(self, trial_path, preprocessed=False):
        print("EEGModel.add_one_trial_data(self, trial_path, preprocessed = False).start...")
        '''
        It was used for reading one-trial data from trial_path and put them
        into X, valence_y, arousal_y
        
        Arguments:
           
            trial_path: the file path of the trial
            
            preprocessed: whether the EEG data is preprocessed
            if it's true, it means that the EEG data has already been preprocessed
            
        '''
        
        # load EEG data
        if preprocessed is False:   # that means we should extract EEG's feature first
            raw_EEG_obj = mne.io.read_raw_fif(trial_path + 'EEG.raw.fif', preload=True, verbose='ERROR')
            EEGs = extract_EEG_feature(raw_EEG_obj)
        else:
            EEGs = np.load(trial_path + 'EEG.npy')      # How can I get EEG.npy
        label = pd.read_csv(trial_path + 'label.csv')
        
        for EEG in EEGs:
            self.X.append(EEG)
            self.y_valence.append(int(label['valence'] > 5))
            self.y_arousal.append(int(label['arousal'] > 5))

        print("EEGModel.add_one_trial_data(self, trial_path, preprocessed = False).end...")
        return 0
    # add_one_trial_data(self, trial_path, preprocessed = False)

    def predict_one_trial(self, trial_path, preprocessed=False):
        '''
             use model to predict one trial
             
             Arguments:
             
                 trial_path: the trial's path
                 
                 preprocessed: whether the EEG data is preprocessed
             
            Return:
            
                A: whether the valence was correctly predict. 
                (1 stands for correct 0 otherwise)
                
                B: whether the arousal was correctly predict. 
                (1 stands for correct 0 otherwise)
        '''
        # load trial data
        if preprocessed is False:
            raw_EEG_obj = mne.io.read_raw_fif(trial_path + 'EEG.raw.fif', preload=True, verbose='ERROR')
            EEGs = extract_EEG_feature(raw_EEG_obj)
        else:
            EEGs = np.load(trial_path + 'EEG.npy')
        
        EEGs = self.standardization(EEGs)
        label = pd.read_csv(trial_path + 'label.csv')
        predict_valences, predict_arousals = self.valence_model.predict(EEGs), self.arousal_model.predict(EEGs)
        predict_valence = np.sum(predict_valences) / float(len(predict_valences)) > 0.5
        predict_arousal = np.sum(predict_arousals) / float(len(predict_arousals)) > 0.5
        ground_true_valence = int(label['valence']) > 5
        ground_true_arousal = int(label['arousal']) > 5
        # Does this sentence means that if the result > 5, it will get the result as final result?      From Ruixin Lee
         
        return (predict_valence == ground_true_valence), (predict_arousal == ground_true_arousal)

    def predict_one_trial_scores(self, trial_path, preprocessed=False):
        '''
             use model to predict one trial
             
             Arguments:
             
                 trial_path: the trial's path
                 
                 preprocessed: whether the EEG data is preprocessed
             
            Return:
            
                score_valence: the scores of valence predicted by face model
                
                score_arousal: the scores of arousal predicted by EEG model
        '''
        # load trial data
        if preprocessed is False:
            raw_EEG_obj = mne.io.read_raw_fif(trial_path + 'EEG.raw.fif', preload=True, verbose='ERROR')
            EEGs = extract_EEG_feature(raw_EEG_obj)
        else:
            # we put the preprocessed data in "EEG.npy"
            EEGs = np.load(trial_path + 'EEG.npy')
        
        EEGs = self.standardization(EEGs)   # standardization
        predict_valences, predict_arousals = self.valence_model.predict(EEGs), self.arousal_model.predict(EEGs)
         
        score_valence = np.sum(predict_valences)/float(len(predict_valences))
        score_arousal = np.sum(predict_arousals)/float(len(predict_arousals))
         
        return score_valence, score_arousal
    
    def predict_one_trial_results(self, trial_path, preprocessed=False):
        '''
             use model to predict one trial
             
             Arguments:
                 
                 trial_path: the trial's path
                 
                 preprocessed: whether the EEG data is preprocessed
             
            Return:
            
                result_valence: the results of valence predicted by face model
                
                result_arousal: the results of arousal predicted by EEG model
        '''
        score_valence, score_arousal = self.predict_one_trial_scores(trial_path, preprocessed)
        result_valence = score_valence > 0.5
        result_arousal = score_arousal > 0.5
        
        return result_valence, result_arousal
# class EEGModel
