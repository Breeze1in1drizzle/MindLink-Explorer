from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def myFT1():
    # 产生一个测试信号，振幅为2的正弦波，其频率在3kHZ缓慢调制，振幅以指数形式下降的白噪声
    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500 * np.cos(2 * np.pi * 0.25 * time)
    carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
    noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time / 5)
    x = carrier + noise

    # 计算并绘制STFT的大小
    f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def create_mne_objects():
    '''
    创建mne objects的核心模板
    :return:
    '''
    import mne
    import numpy as np
    # Create arbitrary data
    sfreq = 1000  # Sampling frequency
    times = np.arange(0, 10, 0.001)  # Use 10000 samples (10s)
    # 从1到10-->[1,10)，步长为0.001（这里0.001即采样周期，也就是每次采样的时间间隔：1/1000）
    # 初始化时间序列[1，10)即10秒，频率1s 1000次，即10秒10000次，10s就是一共采样的时间
    '''
    这里我需要确定：emotiv epoc+的采样率，我记得是128Hz
    时间窗口可以任意我设置，这里设置50s(采集1分钟以内的数据)
    '''

    # 这里是数据
    sin = np.sin(times * 10)  # Multiplied by 10 for shorter cycles
    cos = np.cos(times * 10)
    sinX2 = sin * 2
    cosX2 = cos * 2
    # Numpy array of size 4 X 10000.
    data = np.array([sin, cos, sinX2, cosX2])
    # 这里是数据

    # 定义通道的名字以及类型
    # ch_types = ['eeg' for i in range(14)]     通道类型都是eeg
    # ch_names = []     # emotiv epoc+的14个有效通道
    # Definition of channel types and names.
    ch_types = ['mag', 'mag', 'grad', 'grad']
    ch_names = ['sin', 'cos', 'sinX2', 'cosX2']

    # Create an info object.
    # It is also possible to use info from another raw object.
    # 创建info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create a dummy mne.io.RawArray object
    # 这里的data是np.array的矩阵
    raw = mne.io.RawArray(data, info)

    # Scaling of the figure.
    # For actual EEG/MEG data different scaling factors should be used.
    scalings = {'mag': 2, 'grad': 2}    # 分别说明各个不同种类的信息有多少列，这里是mag两列，grad两列

    raw.plot(n_channels=4, scalings=scalings, title='Data from arrays',
             show=True, block=True)

    # It is also possible to auto-compute scalings
    scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
    raw.plot(n_channels=4, scalings=scalings, title='Auto-scaled Data from arrays',
             show=True, block=True)
    picks = mne.pick_types(
        info, meg=False, eeg=True, stim=False,
        include=ch_names
    )
    raw.save("raw.fif", picks=picks, overwrite=True)

    # 载入刚才保存的数据并打印信息
    raw_fif_data = mne.io.read_raw_fif("raw.fif", preload=True, verbose='ERROR')
    print(raw_fif_data.info)



def create_mne_object_from_pupil_diameter(data, sample_rate=60):
    import mne
    import numpy as np
    data = np.array([data])
    # ch_types = ['eye']
    ch_types = ['eeg']
    ch_names = ['pupil_diameter']
    info = mne.create_info(ch_names=ch_names, sfreq=sample_rate, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    # scalings = {'eye': 1}
    scalings = {'eeg': 1}
    raw.plot(n_channels=1, scalings=scalings, title='Data from arrays',
             show=True, block=True)

    # It is also possible to auto-compute scalings
    scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
    raw.plot(n_channels=1, scalings=scalings, title='Auto-scaled Data from arrays',
             show=True, block=True)
    picks = mne.pick_types(
        info, meg=False, eeg=True, stim=False,
        include=ch_names
    )
    raw.save("raw.fif", picks=picks, overwrite=True)

    # 载入刚才保存的数据并打印信息
    raw_fif_data = mne.io.read_raw_fif("raw.fif", preload=True, verbose='ERROR')
    print(raw_fif_data.info)
    return raw_fif_data


if __name__ == "__main__":
    # myFT1()
    import pandas as pd
    data = pd.DataFrame(pd.read_csv('P1-Rec1-All-Data-New_Section_2.csv'))
    data = data['PupilDiameter']
    pupil_diameter_in_mne = create_mne_object_from_pupil_diameter(data=data, sample_rate=60)
