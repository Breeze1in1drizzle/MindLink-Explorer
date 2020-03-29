# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 18:01:20 2018

@author: Yongrui Huang

This script performs some test for subject independent problem
"""

import sys
sys.path.append('../../')
import configuration
from real_time_detection.EEG import svm_model
from real_time_detection.EEG import lstm_regression

def sensitivity_test():
    '''
    To see whether the model will be unstable if the data changes.
    
    For each subject, we train in other 23 subjects and test on it, so when 
    perform testing in a subject, its data will not be seen by model.
    
    '''
    root_path = configuration.DATASET_PATH + 'MAHNOB_HCI/'  
    
    
    for i in range(1, 25):
        print ('subject: %d' % i)
        model = lstm_regression.LstmModelRegression()
        
        train_subject = set(range(1, 25)) - set([i,])
        test_subject = set(range(1, 25)) - train_subject
        
        
        for subject_id in train_subject:
            subject_path = root_path + str(subject_id) + '/'
            
            for trial_id in range(1, 21):
                trial_path = '%s/trial_%d/'%(subject_path, trial_id)
                model.add_train_data(trial_path)
                    
        model.train()
        
        for subject_id in test_subject:
            subject_path = root_path + str(subject_id) + '/'
            
            for trial_id in range(1, 21):
                trial_path = '%s/trial_%d/'%(subject_path, trial_id)
                model.add_test_data(trial_path)
        
        
        model.evalute()

def test():
    root_path = configuration.DATASET_PATH + 'MAHNOB_HCI/'  
    
    model = svm_model.SvmModel()
   
    for subject_id in range(1, 8):
        subject_path = root_path + str(subject_id) + '/'
        
        for trial_id in range(1, 21):
            trial_path = '%s/trial_%d/'%(subject_path, trial_id)
            model.add_train_data(trial_path)
                
    model.train()
    
    for subject_id in range(20, 25):
        subject_path = root_path + str(subject_id) + '/'
        
        for trial_id in range(1, 21):
            trial_path = '%s/trial_%d/'%(subject_path, trial_id)
            model.add_test_data(trial_path)
    
    
    model.evalute()

import time

if __name__ == '__main__':
    start = time.time()
    test()
    print (time.time() - start)
   