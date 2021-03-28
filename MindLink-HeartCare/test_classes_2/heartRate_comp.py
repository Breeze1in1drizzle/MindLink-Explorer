# -*- coding: utf-8 -*-
'''
这个文件用于提供心率计算的算法：将实时得到的视频信息（4维的 frames）进行欧拉放大，并存储在内存当中
'''

import cv2
import numpy as np
import dlib
import time
from scipy import signal

import Queue

# from cv2 import pyrUp, pyrDown


class heartRateComputation(object):
    '''
    该类只提供计算方法，工具类，实验完成后应该改名为 tools系列的工具类
    '''
    def __init__(self):
        self.MIN_HZ = 0.83
        self.MAX_HZ = 3.33
        self.DEBUG_MODE = False

    def set_parameters(self, MIN_HZ=0.83, MAX_HZ=3.33, DEBUG_MODE=False):
        self.MIN_HZ = MIN_HZ
        self.MAX_HZ = MAX_HZ
        self.DEBUG_MODE = DEBUG_MODE

    def get_forehead_roi(self, face_points):
        '''
        Gets the region of interest for the forehead.
        :param face_points:
        :return:
        '''
        # Store the points in a Numpy array so we can easily get the min and max for x and y via slicing
        points = np.zeros((len(face_points.parts()), 2))
        for i, part in enumerate(face_points.parts()):
            points[i] = (part.x, part.y)

        min_x = int(points[21, 0])
        min_y = int(min(points[21, 1], points[22, 1]))
        max_x = int(points[22, 0])
        max_y = int(max(points[21, 1], points[22, 1]))
        left = min_x
        right = max_x
        top = min_y - (max_x - min_x)
        bottom = max_y * 0.98
        return int(left), int(right), int(top), int(bottom)

    def get_nose_roi(self, face_points):
        '''
        Gets the region of interest for the nose.
        :param face_points:
        :return:
        '''
        points = np.zeros((len(face_points.parts()), 2))
        for i, part in enumerate(face_points.parts()):
            points[i] = (part.x, part.y)
        # Nose and cheeks
        min_x = int(points[36, 0])
        min_y = int(points[28, 1])
        max_x = int(points[45, 0])
        max_y = int(points[33, 1])
        left = min_x
        right = max_x
        top = min_y + (min_y * 0.02)
        bottom = max_y + (max_y * 0.02)
        return int(left), int(right), int(top), int(bottom)

    def get_full_roi(self, face_points):
        '''
        Gets region of interest that includes forehead, eyes, and nose.
        Note:  Combination of forehead and nose performs better.
        This is probably because this ROI includes eyes,
        and eye blinking adds noise.
        :param face_points:
        :return:
        '''
        points = np.zeros((len(face_points.parts()), 2))
        for i, part in enumerate(face_points.parts()):
            points[i] = (part.x, part.y)
        # Only keep the points that correspond to the internal features of the face (e.g. mouth, nose, eyes, brows).
        # The points outlining the jaw are discarded.
        min_x = int(np.min(points[17:47, 0]))
        min_y = int(np.min(points[17:47, 1]))
        max_x = int(np.max(points[17:47, 0]))
        max_y = int(np.max(points[17:47, 1]))

        center_x = min_x + (max_x - min_x) / 2
        left = min_x + int((center_x - min_x) * 0.15)
        right = max_x - int((max_x - center_x) * 0.15)
        top = int(min_y * 0.88)
        bottom = max_y
        return int(left), int(right), int(top), int(bottom)

    def compute_bpm(self, filtered_values, fps, buffer_size, last_bpm):
        '''
        Calculate the pulse in beats per minute (BPM)
        :param filtered_values:
        :param fps:
        :param buffer_size:
        :param last_bpm:
        :return:
        '''
        # Compute FFT
        fft = np.abs(np.fft.rfft(filtered_values))
        # Generate list of frequencies that correspond to the FFT values
        freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)
        # Filter out any peaks in the FFT that are not within our range of [MIN_HZ, MAX_HZ]
        # because they correspond to impossible BPM values.
        while True:
            max_idx = fft.argmax()
            bps = freqs[max_idx]
            if bps < self.MIN_HZ or bps > self.MAX_HZ:
                if self.DEBUG_MODE:
                    print('BPM of {0} was discarded.'.format(bps * 60.0))
                fft[max_idx] = 0
            else:
                bpm = bps * 60.0
                break
        # It's impossible for the heart rate to change more than 10% between samples,
        # so use a weighted average to smooth the BPM with the last BPM.
        if last_bpm > 0:
            bpm = (last_bpm * 0.9) + (bpm * 0.1)
        return bpm

    def get_avg(self, roi1, roi2):
        '''
        Averages the green values for two arrays of pixels
        :param roi1:
        :param roi2:
        :return:
        '''
        roi1_green = roi1[:, :, 1]      # RGB
        roi2_green = roi2[:, :, 1]
        avg = (np.mean(roi1_green) + np.mean(roi2_green)) / 2.0
        # print('avg: ')
        # print(avg)
        return avg

    def get_roi_avg(self, frame, view, face_points, draw_rect=True):
        '''
        Get the average value for the regions of interest.
        It will also draw a green rectangle around
        the regions of interest, if requested.
        :param frame: img captured from camera
        :param view: np.array style of frame
        :param face_points: combination of key points of (X_left, Y_top) and (X_right, Y_bottom) of human face
        :param draw_rect: if draw rectangular in certain position such as forehead and nose
        :return:
        '''
        # Get the regions of interest.
        fh_left, fh_right, fh_top, fh_bottom = self.get_forehead_roi(face_points)   # 获取前额的位置
        nose_left, nose_right, nose_top, nose_bottom = self.get_nose_roi(face_points)   # 获取鼻子的位置

        # 这里可以考虑继续增加不同位置的监测点

        # Draw green rectangles around our regions of interest (ROI)
        if draw_rect:   # 在特定的位置绘制矩阵框（鼻子和前额）
            cv2.rectangle(
                view, (fh_left, fh_top), (fh_right, fh_bottom),
                color=(0, 255, 0), thickness=2)
            cv2.rectangle(
                view, (nose_left, nose_top), (nose_right, nose_bottom),
                color=(0, 255, 0), thickness=2)
        # 将前额与鼻子区域的图片从整体图片中分割出来
        # Slice out the regions of interest (ROI) and average them
        fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
        nose_roi = frame[nose_top:nose_bottom, nose_left:nose_right]
        return self.get_avg(roi1=fh_roi, roi2=nose_roi)     # 一个数值

    def sliding_window_demean(self, signal_values, num_windows):
        '''
        num_windows = 15
        :param signal_values: detrended signals of roi avg values
        :param num_windows:
        :return:
        '''
        # round-->四舍五入浮点数使之成为整数，将信号分割为 num_windows 个窗口，每个窗口 size 为 window_size
        window_size = int(round(len(signal_values) / num_windows))
        # 构造一个相同形状的zeros矩阵
        demeaned = np.zeros(signal_values.shape)
        for i in range(0, len(signal_values), window_size):     # 每次移动 window_size 个步长
            if i + window_size > len(signal_values):    # 如果超出数组长度，则进行一些处理
                window_size = len(signal_values) - i
            curr_slice = signal_values[i: i + window_size]  # 当前的数据段
            if self.DEBUG_MODE and curr_slice.size == 0:
                print('Empty Slice: size={0}, i={1}, window_size={2}'.format(signal_values.size, i, window_size))
                print(curr_slice)
            demeaned[i:i + window_size] = curr_slice - np.mean(curr_slice)  # 当前数据段减去本数据的平均值
        return demeaned     # 根据时间窗对每一段的数据减去平均值（归一化的一个步骤）

    def butterworth_filter(self, data, low, high, sample_rate, order=5):
        '''
        Creates the specified Butterworth filter and applies it.
        :param data: avg singals from Green channel of roi, which have already been demeaned
        :param low: low frequency
        :param high: high frequency
        :param sample_rate: frames per seconds
        :param order:
        :return:
        '''
        nyquist_rate = sample_rate * 0.5        # 最大允许的抽样间隔称为“奈奎斯特间隔”
        low /= nyquist_rate
        high /= nyquist_rate
        b, a = signal.butter(order, [low, high], btype='band')      # 为什么order是等于5？？？
        return signal.lfilter(b, a, data)

    def filter_signal_data(self, values, fps):
        '''
        values = roi_avg_values # 不同时间中各个roi区域计算的均值组成的时序序列，每个元素都是roi区域的均值
        fps = fps
        :param values: a temporal sequence (Python List) of average values of green channel of regions of interest
        :param fps: frames per seconds of camera 每秒采集的图片的数量
        :return:
        '''
        # Ensure that array doesn't have infinite or NaN values
        values = np.array(values)
        np.nan_to_num(values, copy=False)

        # Smooth the signal by detrending and demeaning
        # from scipy import signal
        # 参考这个链接 https://blog.csdn.net/weixin_43031412/article/details/102698121
        # 去除信号的线性趋势
        detrended = signal.detrend(values, type='linear')

        # 首先sliding window demean
        demeaned = self.sliding_window_demean(signal_values=detrended, num_windows=15)

        # Filter signal with Butterworth bandpass filter
        # 然后butterworth filter
        filtered = self.butterworth_filter(data=demeaned,
                            low=self.MIN_HZ, high=self.MAX_HZ, sample_rate=fps, order=5)
        # print 'filetered:'
        # print type(filtered)  # np.array
        # print len(filtered)   # 根据采集的数据而定，一般为103或104
        # print filtered
        return filtered


