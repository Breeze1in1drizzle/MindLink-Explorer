# -*- coding: utf-8 -*-
'''
@date: Mar 26 2021
@author: Ruixin Lee
'''


import cv2
import numpy as np
import dlib
import time
from scipy import signal

import Queue

import EVM_V1 as EVM

# from cv2 import pyrUp, pyrDown


class heartRateObserver(object):
    '''
    This class is for non-contact heart rate detection via video.
    This class provides more than one methods to for its main function.
    This class supports both online and offline analyses of video.
    '''
    def __init__(self):
        # Constants
        self.WINDOW_TITLE = 'MindLink-HeartCare'
        self.BUFFER_MAX_SIZE = 500  # Number of recent ROI average values to store

        # Number of recent ROI average values to show in the pulse graph
        self.MAX_VALUES_TO_GRAPH = 50

        self.MIN_HZ = 0.83  # 50 BPM - minimum allowed heart rate
        self.MAX_HZ = 3.33  # 200 BPM - maximum allowed heart rate

        self.frame_queue = Queue.Queue(maxsize=10)      # 定义一个全局队列，进行数据控制

        '''
        Minimum number of frames required before heart rate is computed.
        Higher values are slower, but
        more accurate.
        '''
        self.MIN_FRAMES = 100

        self.DEBUG_MODE = False
    ######################################################################
    ######################################################################
    ######################################################################

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

    def get_max_abs(self, lst):
        '''
        Returns maximum absolute value from a list
        :param lst:
        :return:
        '''
        return max(max(lst), -min(lst))

    def draw_graph(self, signal_values, graph_width, graph_height):
        '''
        Draws the heart rate graph in the GUI window.
        :param signal_values:
        :param graph_width:
        :param graph_height:
        :return:
        '''
        graph = np.zeros((graph_height, graph_width, 3), np.uint8)
        scale_factor_x = float(graph_width) / self.MAX_VALUES_TO_GRAPH

        # Automatically rescale vertically based on the value with largest absolute value
        max_abs = self.get_max_abs(signal_values)
        scale_factor_y = (float(graph_height) / 2.0) / max_abs

        midpoint_y = graph_height / 2
        for i in range(0, len(signal_values) - 1):
            curr_x = int(i * scale_factor_x)
            curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
            next_x = int((i + 1) * scale_factor_x)
            next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
            cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1)
        return graph

    def draw_bpm(self, bpm_str, bpm_width, bpm_height):
        '''
        Draws the heart rate text (BPM) in the GUI window.
        :param bpm_str:
        :param bpm_width:
        :param bpm_height:
        :return:
        '''
        bpm_display = np.zeros((bpm_height, bpm_width, 3), np.uint8)
        bpm_text_size, bpm_text_base = cv2.getTextSize(bpm_str, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2.7,
                                                       thickness=2)
        bpm_text_x = int((bpm_width - bpm_text_size[0]) / 2)
        bpm_text_y = int(bpm_height / 2 + bpm_text_base)
        cv2.putText(bpm_display, bpm_str, (bpm_text_x, bpm_text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=2.7, color=(0, 255, 0), thickness=2)
        bpm_label_size, bpm_label_base = cv2.getTextSize('BPM', fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6,
                                                         thickness=1)
        bpm_label_x = int((bpm_width - bpm_label_size[0]) / 2)
        bpm_label_y = int(bpm_height - bpm_label_size[1] * 2)
        cv2.putText(bpm_display, 'BPM', (bpm_label_x, bpm_label_y),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0, 255, 0), thickness=1)
        return bpm_display

    def draw_fps(self, frame, fps):
        '''
        Draws the current frames per second in the GUI window.
        :param frame:
        :param fps:
        :return:
        '''
        cv2.rectangle(frame, (0, 0), (100, 30), color=(0, 0, 0), thickness=-1)
        cv2.putText(frame, 'FPS: ' + str(round(fps, 2)), (5, 20), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1, color=(0, 255, 0))
        return frame

    def draw_graph_text(self, text, color, graph_width, graph_height):
        '''
        Draw text in the graph area
        :param text:
        :param color:
        :param graph_width:
        :param graph_height:
        :return:
        '''
        graph = np.zeros((graph_height, graph_width, 3), np.uint8)
        text_size, text_base = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, thickness=1)
        text_x = int((graph_width - text_size[0]) / 2)
        text_y = int((graph_height / 2 + text_base))
        cv2.putText(
            graph, text, (text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1, color=color, thickness=1)

        return graph

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

    def shut_down(self, webcam):
        '''
        Clean up
        :param webcam:
        :return:
        '''
        webcam.release()
        cv2.destroyAllWindows()
        exit(0)

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

    def run_pulse_observer(self, detector, predictor, webcam, window_title):
        '''
        Main function.
        :param detector: detects human face, returns a combination of (X_left, Y_top) and (X_right, Y_bottom)
        :param predictor: predicts key points of human face
        :param webcam: reads video signals from camera
        :param window_title: name of window
        :return:
        '''
        roi_avg_values = []
        graph_values = []
        times = []
        last_bpm = 0
        graph_height = 200
        graph_width = 0
        bpm_display_width = 0

        # 这里应该做一个控制：先读取一个时间序列上的frames，并用来进行了欧拉放大后，再进行心率计算
        # 具体步骤如下：
        # 首先，设置一个while循环，不断读取数据，并保证数据量不高于BufferMax，也就是__init__函数中自定义的缓存最大数量
        # 这个Buffer相当于一个缓存队列，不断地读取（入队），同时在数量超出本身后删除最前面的数据（出队）
        # 设置一个线程来进行上述的while循环，并设置一个全局的队列来控制
        #
        # 这里是要设置一个线程进行队列的读取
        # while cv2.getWindowProperty(window_title, 0) == 0:
        #     ret_val, frame = webcam.read()
        #     if not ret_val:
        #         print("ERROR: Unable to read from webcam. Exiting.")
        #         self.shut_down(webcam)
        #     self.frame_queue.put(frame)


        ###
        # 应该另外写一个类，然后在本文件的这个地方申请一个这个类的实例。
        # 用这个类来进行数据读取，同时这个类提供一个数据返回的接口，供本文件的类的函数使用。
        # 这个类可以写在EVM里面，因为要进行了欧拉放大处理之后，再将数据存入内存队列中
        ###

        ###
        # 所以这么一想，应该分成三个类：
        # 第一个：负责读取摄像头的视频，进行欧拉放大，然后再存入内存队列当中。这个类要控制队列的大小，保持数据量的稳定。（这个类甚至可以拆分成为两个类）
        # 第二个：调用内存中已经欧拉放大了的图像，进行心率计算
        # 第三个：window展示（实时可视化界面）
        ###

        ###
        # 又或者可以这样子来分不同的类
        # 第一个类：负责读取摄像头视频，构造并控制内存队列，开启不同的线程，同时负责GUI的展示
        # 第二个类：算法类，包含EVM算法以及计算心率算法
        ###

        # cv2.getWindowProperty() returns -1 when window is closed by user.
        while cv2.getWindowProperty(window_title, 0) == 0:
            ret_val, frame = webcam.read()

            # ret_val == False if unable to read from webcam
            if not ret_val:
                print("ERROR: Unable to read from webcam. Exiting.")
                self.shut_down(webcam)

            '''
            Make copy of frame before we draw on it.  We'll display the copy in the GUI.
            The original frame will be used to compute heart rate.
            '''
            print 'type of frame: ', type(frame), ', ', frame.shape
            view = np.array(frame)      # 将摄像头读取到的图像转为 np.array的格式
            print 'type of view: ', type(view), ', ', view.shape

            # Heart rate graph gets 75% of window width. BPM gets 25%.
            # 计算window窗口大小？？
            if graph_width == 0:
                graph_width = int(view.shape[1] * 0.75)
                if self.DEBUG_MODE:
                    print('Graph width = {0}'.format(graph_width))
            if bpm_display_width == 0:
                bpm_display_width = view.shape[1] - graph_width

            # Detect face using dlib
            faces = detector(frame, 0)      # 得到关键点坐标（左上角和右下角）
            # print("faces type: ", type(faces))
            # test_faces = np.array(faces[0])
            # print("faces size: ", len(test_faces))
            # print("test_faces type: ", type(test_faces))
            # print(test_faces)
            if len(faces) == 1:     # faces表示人脸的矩阵方框的范围，只允许一个人脸
                face_points = predictor(frame, faces[0])    # frame: 图像, faces[0]: 左上角和右下角的坐标，框定人脸位置
                # 计算得到了一个avg数值（通过对绿色通道的平均数值计算而来）
                ####################################################
                # 如果要###########进行欧拉放大，可以考虑在这个位置附近进行
                # 在这里进行图像的欧拉放大，然后返回一个已经放大了色彩的 new_frame
                # 让 frame = new_frame，其他的不用改动
                #########################################
                roi_avg = self.get_roi_avg(frame=frame, view=view, face_points=face_points, draw_rect=True)
                # 时间序列中（各个roi区域的绿色通道数值的均值）作为一个元素，这个队列有多个元素
                roi_avg_values.append(roi_avg)
                times.append(time.time())

                # Buffer is full, so pop the value off the top to get rid of it
                # 设定了缓存的上限，如果超出了则删除最前的那个（先进先出，队列）
                if len(times) > self.BUFFER_MAX_SIZE:
                    roi_avg_values.pop(0)
                    times.pop(0)

                curr_buffer_size = len(times)   # 计算当前缓存了多少个 roi_avg 数值

                # Don't try to compute pulse until we have at least the min. number of frames
                if curr_buffer_size > self.MIN_FRAMES:      # 必须要通过多张图片才能够计算心率
                    # Compute relevant times
                    # times是动态队列，用当前的时间（最后一个）减去最开始的时间计算时长
                    time_elapsed = times[-1] - times[0]
                    # 当前有多少张图片 除以 时长 --> 得到每秒的图片数量，即每秒的frames的数量
                    fps = curr_buffer_size / time_elapsed  # frames per second
                    # Clean up the signal data
                    # 开始针对处理后的数据进行计算
                    filtered = self.filter_signal_data(values=roi_avg_values, fps=fps)

                    graph_values.append(filtered[-1])   # append最后一个值，，，用来draw pulse graph
                    if len(graph_values) > self.MAX_VALUES_TO_GRAPH:
                        graph_values.pop(0)

                    # Draw the pulse graph
                    graph = self.draw_graph(graph_values, graph_width, graph_height)
                    # Compute and display the BPM
                    bpm = self.compute_bpm(filtered, fps, curr_buffer_size, last_bpm)
                    bpm_display = self.draw_bpm(str(int(round(bpm))), bpm_display_width, graph_height)
                    last_bpm = bpm
                    # Display the FPS
                    if self.DEBUG_MODE:
                        view = self.draw_fps(view, fps)
                else:
                    # If there's not enough data to compute HR, show an empty graph with loading text and
                    # the BPM placeholder
                    pct = int(round(float(curr_buffer_size) / self.MIN_FRAMES * 100.0))
                    loading_text = 'Computing pulse: ' + str(pct) + '%'
                    graph = self.draw_graph_text(loading_text, (0, 255, 0), graph_width, graph_height)
                    bpm_display = self.draw_bpm('--', bpm_display_width, graph_height)
            else:
                '''
                No faces detected, so we must clear the lists of values and timestamps.
                Otherwise there will be a gap in timestamps when a face is detected again.
                '''
                del roi_avg_values[:]
                del times[:]
                graph = self.draw_graph_text('No face detected', (0, 0, 255), graph_width, graph_height)
                bpm_display = self.draw_bpm('--', bpm_display_width, graph_height)

            graph = np.hstack((graph, bpm_display))
            view = np.vstack((view, graph))

            cv2.imshow(window_title, view)

            key = cv2.waitKey(1)
            # Exit if user presses the escape key
            if key == 27:
                self.shut_down(webcam)

    def run(self):
        '''
        run an implementation of this class named "heartRateObserver"
        :return:
        '''
        # detect frontal face.
        # 参考博客：dlib.get_frontal_face_detector
        # 获取人脸框
        detector = dlib.get_frontal_face_detector()
        # Predictor pre-trained model can be downloaded from:
        # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
        try:
            # predict roi??? （68个关键点模型地址）返回值：人脸关键点预测器
            # 参考博客：https://blog.csdn.net/weixin_44493841/article/details/93503934
            predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        except RuntimeError as e:
            print('ERROR:  \'shape_predictor_68_face_landmarks.dat\' was not found in current directory.   ' \
                  'Download it from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2')
            return
        print('predictor successfully loaded')

        webcam = cv2.VideoCapture(0)    # 打开摄像头，webcam可以逐帧读取图像
        print('webcam successfully started')

        if not webcam.isOpened():
            print('ERROR: Unable to open webcam. Verify that webcam is connected and try again. Exiting.')
            webcam.release()
            return
        print('webcam is open')

        cv2.namedWindow(self.WINDOW_TITLE)
        print('named window')

        self.run_pulse_observer(detector=detector, predictor=predictor,
                                webcam=webcam, window_title=self.WINDOW_TITLE)
        print('successfully started window')

        # run_pulse_observer() returns when the user has closed the window.  Time to shut down.
        self.shut_down(webcam)


if __name__ == "__main__":
    HRO = heartRateObserver()
    HRO.run()
