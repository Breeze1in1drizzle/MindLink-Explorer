'''
@date: 2021.03.09
@author: Ruixin Lee
This file is for implementations of several kinds of Fourier Transform,
such as Fast Fourier Transform, Short-Time Fourier Transform and so on.
对于MAHNOB-HCI数据集来说，一个文件代表的是 119s 的数据采集。眼动数据集每秒60个样本，采样率60Hz。
其实是接近120s，但是第一秒没到60个，所以一般从第2s开始计算。所以是119秒

这个文件要做的主要就是，将这瞳孔直径信号分为 4个 频率的波，并且计算对应的功率。其实也就是计算瞳孔直径的PSD特征。
最终，
'''


import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def MySTFT_V1():
    '''
    reference: https://www.cnblogs.com/klchang/p/9280509.html
    :return: None
    '''
    import scipy.io.wavfile
    # Read wav file
    # "OSR_us_000_0010_8k.wav" is downloaded from http://www.voiptroubleshooter.com/open_speech/american.html
    sample_rate, signal = scipy.io.wavfile.read("OSR_us_000_0010_8k.wav")
    # print("sample rate:\n", sample_rate)    # sample rate is 8000.

    # Get speech data in the first 2 seconds
    # sample rate is 8000, here the length of list of the signal is 1600, [0, 2×8000).
    signal = signal[0:int(2. * sample_rate)]    # input signal --> 2 seconds，提取前2秒的信号

    print("signal:\n", signal, "\nsignal type:\n", type(signal), "\nsignal length:\n", len(signal))
    # one dimensional array, type: <class 'numpy.ndarray'>, length: 1600

    # Calculate the short time fourier transform
    pow_spec = stft_calculation(signal, sample_rate)
    # print("power spectral density:\n", pow_spec)
    print("power spectral density:\n", pow_spec,
          "\npow_spec type:\n", type(pow_spec), "\npow_spec length y:\n", len(pow_spec),
          "\npow_spec x:", len(pow_spec[0]))

    plt.imshow(pow_spec)
    plt.tight_layout()
    plt.show()

    # print("END!!!")

    return None


# sample_rate = 60Hz
def sample_rate_calculation(filepath=None):
    '''
    该函数能够利用时间戳计算存储在某种信号的文本文件中的采样率
    该函数暂时默认仅用于MAHNOB-HCI-TAGGING数据集的眼动采样频率计算
    :return: 采样率 sample_rate
    '''
    sample_rate = 0
    df = pd.DataFrame(pd.read_csv(filepath))
    time_stamp = df['Timestamp']
    print(type(time_stamp))
    time_stamp = np.array(time_stamp)
    print(type(time_stamp))
    print(time_stamp)
    return sample_rate


def MySTFT_V2(filepath=None):
    sample_rate = 60    # 眼动信号的采样频率为60Hz
    signal_file = pd.DataFrame(pd.read_csv(filepath))
    signal_file['PupilDiameter'] = signal_file['PupilRight']-signal_file['PupilLeft']
    # 得到'PupilDiameter'列后，开始计算
    signal = np.abs(np.array(signal_file['PupilDiameter']))
    print("signal:\n", signal)
    # print("sample rate:\n", sample_rate)    # sample rate is 8000.
    # Get speech data in the first 2 seconds
    # sample rate is 8000, here the length of list of the signal is 1600, [0, 2×8000).
    signal = signal[0:int(2. * sample_rate)]  # input signal --> 2 seconds，提取前2秒的信号
    print("signal:\n", signal, "\nsignal type:\n", type(signal), "\nsignal length:\n", len(signal))
    # one dimensional array, type: <class 'numpy.ndarray'>, length: 1600
    # Calculate the short time fourier transform
    pow_spec = stft_calculation(signal=signal, sample_rate=60, frame_size=0.025, frame_stride=0.05)
    # print("power spectral density:\n", pow_spec)
    print("power spectral density:\n", pow_spec,
          "\npow_spec type:\n", type(pow_spec), "\npow_spec length y:\n", len(pow_spec),
          "\npow_spec x:", len(pow_spec[0]))
    plt.imshow(pow_spec)
    plt.tight_layout()
    plt.show()
    print("END!!!")
    return None


def stft_calculation(signal, sample_rate=16000, frame_size=0.025,  # frame_size设置为 25ms，即0.025s
              frame_stride=0.01, winfunc=np.hamming, NFFT=512):
    '''
    :param signal: 输入的信号，这里是一个一维数组
    :param sample_rate: 采样频率，又称采样率，每秒采集的样本的数量，单位是赫兹(Hz)
    :param frame_size: 将信号分为较短的帧的尺寸size，在语音处理中，通常帧大小在 20ms 到 40ms之间
    :param frame_stride: 相邻帧的滑动尺寸或跳跃尺寸，通常帧的滑动尺寸在 10ms到 20ms之间，这里设置为 10ms，即 0.01s
    :param winfunc: 窗函数采用汉明窗函数 (Hamming Function)
    :param NFFT: 在每一帧，进行512点快速傅里叶变换，即NFFT==512
    :return: pow_frames
    '''
    # Calculate the number of frames from the signal
    # frame_size指每次提取的大小，也就是每个“帧”的大小，每次选取一定时间内的信号。
    # 比如，frame_size==1，则选取一秒内的信号。而采样频率若为50Hz，则一共有1×50个信号，如果是2秒，且采样率为50Hz，则2×50==100个信号样本。
    frame_length = frame_size * sample_rate     # 样本的长度，这里是0.025×8000==200，即每次200个信号样本
    frame_step = frame_stride * sample_rate     # 步长frame_step，用每次滑动的尺寸乘以采样率来获得。
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    print("frame step: ", frame_step)

    delta_length = float(np.abs(signal_length-frame_length))
    num_frames = int(np.ceil(delta_length/frame_step)) + 1

    # zero padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    np_title_1 = np.tile(np.arange(0, frame_length), (num_frames, 1))
    np_title_2 = np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    indices = np_title_1 + np_title_2

    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_length)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute power spectrum
    pow_frames = (1.0/NFFT) * (mag_frames**2)

    print("pow_frames:\n", type(pow_frames))
    print(pow_frames.shape)     # (41, 257)
    print(pow_frames)

    return pow_frames


if __name__ == '__main__':
    # MySTFT_V1()
    # sample_rate_calculation(filepath='../../data/mahnob_example/2/P1-Rec1-All-Data-New_Section_2.csv')
    # pass
    MySTFT_V2(filepath='../../data/mahnob_example/2/P1-Rec1-All-Data-New_Section_2.csv')
