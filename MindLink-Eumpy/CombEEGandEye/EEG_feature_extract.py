# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:21:27 2018
Modified on Fri Mar 12 14:47:50 2021

@author1: Yongrui Huang
@author2: Ruixin Lee
@author3: Xiaojian Liu

This script is used for extracting average PSD (Power spectral density) features 
from raw EEG data. In the meantime, it also puts the features into hard disk in
'.npy' format.

The precess is described as follows.
14 channels and 3 asymmetric pair channels are picked.
For each channel, we extract their PSD feature in each unit second(e.g. 1 second)
with a 50% overlap. 

Note: change tha name MAHNOB-HCI and you can extract your own data!

"""
import glob
import os

import mne
import sys
sys.path.append('../../../')
# import configuration
import numpy as np


def raw_features(trial_path):
    '''
        This method return raw feature from one trial. It was implemented canse
        I want to see the whether the feature extracting could make a different 
        and it did.
        
        Arguments:
           
            trial_path: the path in file system of the trial
        
        Returns:
            
            The raw features. numpy-array-like.
        
    '''
    raw_obj = mne.io.read_raw_fif(trial_path + 'EEG.raw.fif', preload=True, verbose='ERROR')
    raw_obj = add_asymmetric(raw_obj) 
    total_time = int(raw_obj.times.max())
    features = []
    for i in range(0, total_time-1):
        sub_raw_obj = raw_obj.copy().crop(i, i+1)
        data = np.mean(sub_raw_obj.get_data(), axis = 1)
        features.append(data)
    return np.array(features)


def get_average_psd(sub_raw_obj, fmin, fmax):
    '''
    This method returns a the average log psd feature for a MNE raw object
    
    Arguments:
        
        sub_raw_obj: a raw object from MNE library
        
        fmin: the minium frequency you are intreseted
        
        fmax: the maximum frequency you are intreseted
        
    Returns:
        
        average_psd: the required psd features, numpy array like. 
        shape: (the number of the features, )
    
    '''
    try:
        psds, freq = mne.time_frequency.psd_multitaper(sub_raw_obj, fmin=fmin, fmax=fmax, n_jobs=4, verbose='ERROR')
        # print("psds:\n", type(psds), "\n", psds.shape)
        # print("psds:\n", psds)
    except:
        print("get_avg_psds error")
        return False
    # preventing overflow
    psds[psds <= 0] = 1
    psds = 10 * np.log10(psds)
    average_psd = np.mean(psds, axis=1)
    return average_psd


def extract_average_psd_from_a_trial(raw_obj, average_second, overlap):
    '''
    This method returns the average log psd features for a trial
    
    Arguments:
        
        raw_obj: a MNE raw object contains the information from a trial.
        
        average_second: the time unit for average the psd.
        
        overlap: how much overlap will be used.
        
    Returns:
        features: the features of multiply sample.
        shape (the sample number, the feature number)
    
    '''
    
    assert overlap >= 0 and overlap < 1
    
    total_time = int(raw_obj.times.max())
    features = []
    move = average_second * (1 - overlap)
    for start_second in np.arange(0, total_time, move):
        if (start_second + average_second > total_time):
            break
        sub_raw_obj = raw_obj.copy().crop(start_second, 
                                  start_second + average_second)
        
        theta_average_psd = get_average_psd(sub_raw_obj, 4, 8)
        slow_alpha_average_psd = get_average_psd(sub_raw_obj, 8, 10)
        alpha_average_psd = get_average_psd(sub_raw_obj, 8, 12)
        beta_average_psd = get_average_psd(sub_raw_obj, 12, 30)
        gammar_average_psd = get_average_psd(sub_raw_obj, 30, 45)
        if ((theta_average_psd is False) or (slow_alpha_average_psd is False)
                or (alpha_average_psd is False) or (beta_average_psd is False)
                or (gammar_average_psd is False)):
            return False

        feature = np.concatenate((theta_average_psd, slow_alpha_average_psd, 
                                  alpha_average_psd, beta_average_psd, 
                                  gammar_average_psd), axis=None)
        features.append(feature)
    
    return np.array(features)


def get_a_channel_data_from_raw(raw_obj, channel_name):
    '''
        Arguments:
            
            raw_obj: raw object from MNE library.
            
            channel_name: the name of the channel.
            
        Returns:
            
            the numpy array data for the channel.
    
    '''
    return np.array(raw_obj.copy().pick_channels([channel_name]).get_data()[0])


def add_asymmetric(raw_obj):
    '''
        add asymmetric pair into a MNE raw object and return it. This 3 pair 
        are added. (T7-T8, Fp1-Fp2 and CP1-CP2)
        
        Arguments:
            
            raw_obj: the MNE raw object.
            
        Returns:
            
            new_raw_obj: the new MNE raw object with more channel.
    
    '''
    selected_channels = ['Fp1', 'T7', 'CP1', 'Oz', 'Fp2', 'F8', 'FC6', 'FC2', 
                       'Cz', 'C4', 'T8', 'CP6', 'CP2', 'PO4']
    raw_obj.pick_channels(selected_channels)
    raw_data = raw_obj.get_data()
    
    T7_channel = get_a_channel_data_from_raw(raw_obj, 'T7')
    T8_channel = get_a_channel_data_from_raw(raw_obj, 'T8')
    T7_T8 = T7_channel - T8_channel
    
    Fp1_channel = get_a_channel_data_from_raw(raw_obj, 'Fp1')
    Fp2_channel = get_a_channel_data_from_raw(raw_obj, 'Fp2')
    Fp1_Fp2 = Fp1_channel - Fp2_channel
    
    CP1_channel = get_a_channel_data_from_raw(raw_obj, 'CP1')
    CP2_channel = get_a_channel_data_from_raw(raw_obj, 'CP2')
    CP1_CP2 = CP1_channel - CP2_channel
    
    asymmetric_data = np.array([T7_T8, Fp1_Fp2, CP1_CP2])
    
    data = np.append(raw_data, asymmetric_data, axis = 0)
    
    channel_names = selected_channels + ['T7-T8', 'Fp1-Fp2', 'CP1-CP2']
    # print(channel_names)
    channel_types = ['eeg' for i in range(len(channel_names))]
    sfreq = 256
    # montage = 'standard_1005'  #  没用了，
    # info = mne.create_info(channel_names, sfreq, channel_types, montage, verbose='ERROR')m
    info = mne.create_info(channel_names, sfreq, channel_types, verbose='ERROR')
    new_raw_obj = mne.io.RawArray(data, info, verbose='ERROR')
    return new_raw_obj


def raw_to_numpy_one_trial(EEG_path, save_path):
    '''
        read one-trial raw EEG from hard dick, extract the feature and write it
        back.
        
        Arguments:
            
            path: the trial's path
        
    '''
    # print("\nsave_path:\n", save_path, "\n")
    # EEG_path = path + 'EEG.raw.fif'
    raw = mne.io.read_raw_fif(EEG_path, preload=True, verbose='ERROR')
    
    raw = add_asymmetric(raw)
    data = extract_average_psd_from_a_trial(raw, 1, 0.5)
    # error_path = save_path
    if data is False:
        return save_path
    number0 = EEG_path.split(".")
    number1 = number0[0].split("Trial")
    length0 = len(number1)
    detail = number1[length0-1]
    # save_path = save_path+'/'+detail+'/'
    save_path = save_path + '/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #################################
    #################################
    np.save(save_path+'EEG.npy', data)
    #################################
    #################################
    # print(save_path+'EEG.npy')
    return True


def exist_nan_data(EEG_path):
    '''
        check whether NAN data exist after extracting the feature
        
        Arguments:
            
            path: the trial path
            
        Returns:
            
            True if NAN data exist else False.
    '''
    # EEG_path = path + 'EEG.npy'
    import os
    if os.path.exists(EEG_path) == False:
        return True
    data = np.load(EEG_path)
  
    return True if np.sum(np.isnan(data)) > 0 else False


def exist_inf_data(EEG_path):
    '''
    
        check whether INF data exist after extracting the feature
        
        Arguments:
            
            path: the trial path
            
        Returns:
            
            True if INF data exist else False.
    
    '''
    # EEG_path = path + 'EEG.npy'
    import os
    if os.path.exists(EEG_path) == False:
        return True
    data = np.load(EEG_path)
    
    return True if np.sum(np.isinf(data)) > 0 else False


if __name__ == '__main__':
    # ROOT_PATH = configuration.DATASET_PATH + 'newMAHNOB_HCI/'
    # dataset_path = configuration.DATASET_PATH + "newMAHNOB_HCI/*"
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*'
    dirPath = glob.iglob(dataset_path)
    path = dataset_path.replace('/*', '')
    # print(dirPath)
    num_of_folder = 0
    num_of_fif = 0
    num_of_success_fif = 0
    error_list = []

    # test = 0

    for big_file in dirPath:
        num_of_folder += 1
        print("num_of_folder: ", num_of_folder)
        files = os.listdir(big_file)
        for file in files:
            if file.endswith(".fif"):

                # test += 1

                num_of_fif += 1
                file_path = os.path.join(big_file, file)  # 路径+文件名
                file_path = file_path.replace('\\', '/')
                print(file_path)
                # number0 = file_path.split(".")
                # number1=number0[0].split("Trial")
                # length0 = len(number1)
                # detail = number1[length0-1]
                # print(detail)
                # print(big_file.replace('\\', '/'))
                # if exist_nan_data(trial_path) or exist_inf_data(trial_path):
                result = raw_to_numpy_one_trial(file_path, big_file.replace('\\', '/'))
                num_of_success_fif += 1
                if result is not True:
                    print("error error error !!! !!!\n", result)
                    num_of_success_fif -= 1
                    error_list.append(result)
                #     print(test)
                #     npy_EEG = np.load(big_file.replace('\\', '/') + '/EEG.npy')
                #     print("npy_EEG:\n", npy_EEG.shape)        # (x, 85)
                #     break
                # if test >= 1:




    print("num_of_fif: ", num_of_fif)
    print("num_of_success_fif: ", num_of_success_fif)
    print("num_of_folder: ", num_of_folder)
    print("error list:\n", error_list)
    print("main.end......")
