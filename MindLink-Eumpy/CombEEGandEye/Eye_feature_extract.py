'''
瞳孔直径数据存储在csv文件当中，现在把它构造成 伪"EEG"格式的mne数据结构中，
方便进行计算，同时能够统一接口，便于后续的数据融合。
'''

import pandas as pd
import numpy as np
import mne


def create_mne_object_from_pupil_diameter(data, sample_rate=60):
    data = np.array([data])
    # ch_types = ['eye']
    ch_types = ['eeg']
    ch_names = ['pupil_diameter']
    info = mne.create_info(ch_names=ch_names, sfreq=sample_rate, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw


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
    psds, freq = mne.time_frequency.psd_multitaper(sub_raw_obj, fmin=fmin, fmax=fmax, n_jobs=4, verbose='ERROR')
    # preventing overflow
    psds[psds <= 0] = 1
    psds = 10 * np.log10(psds)
    average_psd = np.mean(psds, axis=1)
    return average_psd


def Eye_extract_average_psd_from_a_trial(file_path, save_path, average_second=1, overlap=0.5):

    assert overlap >= 0 and overlap < 1

    data = pd.DataFrame(pd.read_csv(file_path))
    data = data['PupilDiameter']

    raw_obj = create_mne_object_from_pupil_diameter(data=data, sample_rate=60)

    # print(raw_obj)

    total_time = int(raw_obj.times.max())

    features = []

    move = average_second * (1 - overlap)  # 确定步长

    # print("total_time:\n", total_time, "\nmove:\n", move)  # index: 0->237   （199*2-1）个数字
    # i = 1

    for start_second in np.arange(0, total_time, move):  # 时间120s，所以index为：0->119
        if start_second + average_second > total_time:
            break
        sub_raw_obj = raw_obj.copy().crop(start_second, start_second + average_second)
        try:
            first_psd = get_average_psd(sub_raw_obj, fmin=0.0, fmax=0.5)
            # print("first_psd: ", type(first_psd), ", ", first_psd)
            second_psd = get_average_psd(sub_raw_obj, fmin=0.5, fmax=1.0)
            # print("second_psd: ", type(second_psd), ", ", second_psd)
        except:
            print("Eye error!!!")
            return save_path
        feature = np.concatenate((first_psd, second_psd), axis=None)
        features.append(feature)
    npy_features = np.array(features)
    # print("npy_features:\n", npy_features.shape)      # (x, 2)
    np.save(save_path+'Eye.npy', npy_features)
    # print("features:\n", type(feature), "\n", features, "\nlen: ", len(features))
    return True


def test():
    # data = pd.DataFrame(pd.read_csv('P1-Rec1-All-Data-New_Section_2.csv'))
    # data = pd.DataFrame(pd.read_csv('P11-Rec1-All-Data-New_Section_30.csv'))
    # data = data['PupilDiameter']
    Eye_extract_average_psd_from_a_trial(file_path='P11-Rec1-All-Data-New_Section_30.csv', save_path='', average_second=1, overlap=0.5)


if __name__ == "__main__":
    # myFT1()
    test()
