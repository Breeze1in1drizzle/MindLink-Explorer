# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:21:27 2018
@author: Yongrui Huang

This script is used for extracting average PSD (Power spectral density) features
from raw EEG data. In the meantime, it also puts the features into hard disk in
'.npy' format.

The precess is described as follows.
14 channels and 3 asymmetric pair channels are picked.
For each channel, we extract their PSD feature in each unit second(e.g. 1 second)
with a 50% overlap.

Note: change tha name MAHNOB-HCI and you can extract your own data!
"""
import mne
import sys
sys.path.append('../../')
# import configuration
import numpy as np
import os
import matplotlib.pyplot as plt


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
        data = np.mean(sub_raw_obj.get_data(), axis=1)
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
    psds, freq = mne.time_frequency.psd_multitaper(sub_raw_obj, fmin=fmin, fmax=fmax, n_jobs=100, verbose='ERROR')
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
    print("raw_obj:\n", raw_obj)    # <RawArray | 17 x 30720 (120.0 s), ~4.0 MB, data loaded>
    total_time = int(raw_obj.times.max())
    features = []
    move = average_second * (1-overlap)     # 确定步长
    print("total_time:\n", total_time, "\nmove:\n", move)   # index: 0->237   （199*2-1）个数字
    # i = 1
    for start_second in np.arange(0, total_time, move):     # 时间120s，所以index为：0->119
        if start_second + average_second > total_time:
            break
        sub_raw_obj = raw_obj.copy().crop(start_second, start_second + average_second)
        # print("i: ", i)
        # i += 1
        theta_average_psd = get_average_psd(sub_raw_obj, 4, 8)
        # print("theta_average_psd:\n", theta_average_psd, "\ntheta_average_psd")
        # [1*17]
        slow_alpha_average_psd = get_average_psd(sub_raw_obj, 8, 10)
        # print("slow_alpha_average_psd:\n", slow_alpha_average_psd, "\nslow_alpha_average_psd")
        alpha_average_psd = get_average_psd(sub_raw_obj, 8, 12)
        beta_average_psd = get_average_psd(sub_raw_obj, 12, 30)
        gammar_average_psd = get_average_psd(sub_raw_obj, 30, 45)

        feature = np.concatenate((
                theta_average_psd, slow_alpha_average_psd,
                alpha_average_psd, beta_average_psd,
                gammar_average_psd), axis=None)

        # print("feature:\n", type(feature), "\n", feature, "\nfeature length x:\n", len(feature))

        features.append(feature)    # 注意这里是整体的features，所以有's'，是features，不是feature
    print("len of features:\n", len(features), "\n", len(features[0]))      # 每个feature array 长度为85，一共237个(list长度为237)
    print("type of features:\n", type(features))
    print(features)
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


# I doubt it's symmetric but not asymmetric because it's called symmetric in paper...... From RuixinLee
def add_asymmetric(raw_obj):
    '''
        add asymmetric pair into a MNE raw object and return it. This 3 pair
        are added. (T7-T8, Fp1-Fp2 and CP1-CP2)
        Arguments:
            raw_obj: the MNE raw object.
        Returns:
            new_raw_obj: the new MNE raw object with more channel.
    '''
    # ['AF3', 'F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'AF4', 'F4', 'F8', 'FC6', 'T8', 'P8', 'O2']
    selected_channels = [
        'Fp1', 'T7', 'CP1', 'Oz',
        'Fp2', 'F8', 'FC6', 'FC2',
        'Cz', 'C4', 'T8', 'CP6',
        'CP2', 'PO4'
    ]
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

    data = np.append(raw_data, asymmetric_data, axis=0)

    channel_names = selected_channels + ['T7-T8', 'Fp1-Fp2', 'CP1-CP2']
    channel_types = ['eeg' for i in range(len(channel_names))]
    sfreq = 256

    # montage = 'standard_1005'

    # info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types, montage, verbose='ERROR)
    # verbose='ERROR'
    # 以前的版本有mongtage，现在的没有了，不需要这个参数
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=channel_types, verbose='ERROR')  # verbose='ERROR'
    # ValueError: Invalid value for the 'verbose' parameter (when a string).
    # Allowed values are 'DEBUG', 'INFO', 'WARNING', 'ERROR', and 'CRITICAL',
    # but got 'STANDARD_1005' instead.
    new_raw_obj = mne.io.RawArray(data, info, verbose='ERROR')
    return new_raw_obj


def raw_to_numpy_one_trial(folder_path=None):
    '''
        read one-trial raw EEG from hard dick, extract the feature and write it
        back.
        Arguments:
            path: the trial's path
    '''
    # EEG_path = path + 'EEG.raw.fif'
    if folder_path==None:
        path = ''
        EEG_path = 'origin.fif'
    else:
        path = folder_path
        EEG_path = folder_path + r'\origin.fif'
        # 前面要加一个'r'进行一次转义，后面这里也要加上一个'r'，即"r'\origin.fif'"
        print("EEG_path:\n", EEG_path)
    raw = mne.io.read_raw_fif(EEG_path, preload=True, verbose='ERROR')
    print("successfully raw")
    raw = add_asymmetric(raw)
    print("successfully add asy:\n", raw)
    data = extract_average_psd_from_a_trial(raw, 1, 0.5)
    print("raw info:\n", raw.info)
    ##################################################
    ##################################################
    ##################################################
    """
    案例：
    获取10-20秒内的良好的MEG数据

    # 根据type来选择 那些良好的MEG信号(良好的MEG信号，通过设置exclude="bads") channel,
    结果为 channels所对应的的索引
    """
    picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    t_idx = raw.time_as_index([10., 20.])
    # data, times = raw[picks, t_idx[0]:t_idx[1]]
    # plt.plot(times, data.T)
    # plt.title("Sample channels")
    sf = raw.info['sfreq']
    ch_names = raw.info['ch_names']
    print("ch_names:\n", ch_names)      # 17个类型的channel，其中14个是原始的BCI channel，3个是计算得出的非对称channels
    print("sf:\n", sf)
    ##################################################
    ##################################################
    ##################################################
    np.save(path+'EEG.npy', data)
    print(path+'EEG.npy')


def exist_nan_data(path):
    '''
        check whether NAN data exist after extracting the feature
        Arguments:
            path: the trial path
        Returns:
            True if NAN data exist else False.
    '''
    EEG_path = path + 'EEG.npy'
    import os
    if os.path.exists(EEG_path) == False:
        return True
    data = np.load(EEG_path)
    return True if np.sum(np.isnan(data)) > 0 else False


def exist_inf_data(path):
    '''
        check whether INF data exist after extracting the feature
        Arguments:
            path: the trial path
        Returns:
            True if INF data exist else False.
    '''
    EEG_path = path + 'EEG.npy'
    import os
    if os.path.exists(EEG_path) == False:
        return True
    data = np.load(EEG_path)
    return True if np.sum(np.isinf(data)) > 0 else False


if __name__ == "__main__":

    # ROOT_PATH = configuration.DATASET_PATH + 'MAHNOB_HCI/'

    for subject_id in range(1, 25):
        subject_path = ROOT_PATH + str(subject_id)+'/'
        print('start subject %d' % subject_id)

        for trial_id in range(1, 21):
            trial_path = subject_path + 'trial_' + str(trial_id) + '/'

            # if exist_nan_data(trial_path) or exist_inf_data(trial_path):
            raw_to_numpy_one_trial(trial_path)
    # abs_path = os.path.abspath(__file__)
    # ROOT_PATH = abs_path.replace('\\', '/')[:-30]
    # print(ROOT_PATH)
    # path = r"D:\myworkspace\mypyworkspace\MindLink-Eumpy\data\mahnob_example\EEG_files"
    # raw_to_numpy_one_trial(path=path)
    # raw_to_numpy_one_trial()
    # EEG_path = 'origin.fif'
    # raw = mne.io.read_raw_fif(EEG_path, preload=True, verbose='ERROR')
    print("loading successfully")