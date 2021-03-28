# -*- coding: utf-8 -*-
'''
这个文件用于可视化其他类计算出来的心率的结果
同时该类提供不同的方法读取视频数据，统一作为 4维的 frames数据传输至计算的类当中
包含EVM算法以及后续的心率计算算法，同时包含系统的GUI
'''

import cv2
import numpy as np
import dlib
import time
from scipy import signal

from multiprocessing import Queue
from multiprocessing import Process
import heartRate_comp as HRC

# from cv2 import pyrUp, pyrDown

webcam_G = cv2.VideoCapture(0)  # webcam Global
if webcam_G.isOpened()==False:
    print 'webcam Global is False.'
    exit(0)


def run(heartRateObserver=None):
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

    heartRateObserver.webcam = cv2.VideoCapture(0)  # 打开摄像头，webcam可以逐帧读取图像

    if not heartRateObserver.webcam.isOpened():
        print 'ERROR: Unable to open webcam. Verify that webcam is connected and try again. Exiting.'
        heartRateObserver.webcam.release()
        return

    cv2.namedWindow(heartRateObserver.WINDOW_TITLE)

    heartRateObserver.start(webcam=heartRateObserver.webcam)   # 开始采集数据

    heartRateObserver.run_pulse_observer(detector=detector, predictor=predictor,
                            webcam=heartRateObserver.webcam, window_title=heartRateObserver.WINDOW_TITLE)

    # run_pulse_observer() returns when the user has closed the window.  Time to shut down.
    heartRateObserver.shut_down(heartRateObserver.webcam)
    if heartRateObserver.webcam.isOpened():
        print 'failed to shut down webcam!'


class heartRateExperiments(object):
    '''
    This class is for non-contact heart rate detection via video.
    This class provides more than one methods to for its main function.
    This class supports both online and offline analyses of video.
    '''
    def __init__(self, heartRateComp=None):
        '''
        :param heartRateComp: provides methods of EVM and heart rate computation
        '''
        # Constants
        self.WINDOW_TITLE = 'MindLink-HeartCare'
        self.BUFFER_MAX_SIZE = 500  # Number of recent ROI average values to store
        # Number of recent ROI average values to show in the pulse graph
        self.MAX_VALUES_TO_GRAPH = 50
        self.MIN_HZ = 0.83  # 50 BPM - minimum allowed heart rate
        self.MAX_HZ = 3.33  # 200 BPM - maximum allowed heart rate
        self.raw_data_queue = Queue(maxsize=-1)      # 定义一个全局队列，进行数据控制

        # 这个类里面有一个线程，启动后一直维持着数据采集，视频读取，存到队列里面

        '''
        Minimum number of frames required before heart rate is computed.
        Higher values are slower, but
        more accurate.
        '''
        self.MIN_FRAMES = 100
        self.DEBUG_MODE = False

        self.hrc = heartRateComp
        self.hrc.set_parameters(MIN_HZ=self.MIN_HZ, MAX_HZ=self.MAX_HZ, DEBUG_MODE=self.DEBUG_MODE)

        self.window_title = 'MindLink-HeartCare'
        self.num_start = 0
        self.detector = dlib.get_frontal_face_detector()
        # Predictor pre-trained model can be downloaded from:
        # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
        try:
            # predict roi??? （68个关键点模型地址）返回值：人脸关键点预测器
            # 参考博客：https://blog.csdn.net/weixin_44493841/article/details/93503934
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        except RuntimeError as e:
            print('ERROR:  \'shape_predictor_68_face_landmarks.dat\' was not found in current directory.   ' \
                'Download it from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2')
            return
        self.webcam = cv2.VideoCapture(0)  # 打开摄像头，webcam可以逐帧读取图像
        print '__init__, type of self.webcam: ', type(self.webcam)
        if not self.webcam.isOpened():
            print 'ERROR: Unable to open webcam. Verify that webcam is connected and try again. Exiting.'
            self.webcam.release()
            return
        cv2.namedWindow(self.WINDOW_TITLE)
        self.start()   # 开始采集数据
        self.run_pulse_observer()
        # run_pulse_observer() returns when the user has closed the window.  Time to shut down.
        self.shut_down(self.webcam)
        if self.webcam.isOpened():
            print 'failed to shut down webcam!'
            ######################################################################
        ######################################################################
        ######################################################################

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

    def shut_down(self, webcam):
        '''
        Clean up
        :param webcam:
        :return:
        '''
        # webcam.release()
        self.webcam.release()
        cv2.destroyAllWindows()
        exit(0)

    def start(self):
        '''
        start a sub-process to record video images
        :return:
        '''
        self.num_start += 1
        sub_process = Process(target=self.read_img_from_camera)
        sub_process.start()

    def read_img_from_camera(self):
        print 'sub process started!!!'
        # flag = cv2.getWindowProperty(self.window_title, 0)
        # print 'flag: ', flag
        while (1):
            if self.raw_data_queue.qsize() >= self.BUFFER_MAX_SIZE:     # 队列满了
                print 'continue'
                continue    # 不读取数据
            else:
                print 'else::::'
                ret_val, frame = webcam_G.read()
                # ret_val == False if unable to read from webcam
                print 'read_img_from_camera: ', type(frame)
                if not ret_val:
                    print("ERROR: Unable to read from webcam. Exiting.")
                    self.shut_down(self.webcam)
                self.raw_data_queue.put(frame)      # 将原始的图片装载入队列里面

    def run_pulse_observer(self):
        '''
        Main function.
        :param detector: detects human face, returns a combination of (X_left, Y_top) and (X_right, Y_bottom)
        :param predictor: predicts key points of human face
        :param webcam: reads video signals from camera
        :param window_title: name of window
        :return:
        '''
        self.window_title = self.window_title
        roi_avg_values = []
        graph_values = []
        times = []
        last_bpm = 0
        graph_height = 200
        graph_width = 0
        bpm_display_width = 0

        print 'before while'
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

        # 这里start一个线程thread，摄像头读取图像，然后欧拉放大，再把数据存入数据结构（内存）

        # 这个循环不断读取上述内存的数据，提取roi区域并计算心率
        # cv2.getWindowProperty() returns -1 when window is closed by user.
        flag = cv2.getWindowProperty(self.window_title, 0)
        print 'flag: ', flag
        while cv2.getWindowProperty(self.window_title, 0) == 0:
        # while self.raw_data_queue.qsize() > 0:
            print 'in while'
            frame = self.raw_data_queue.get()       # 读取队列的数据
            print frame
            print 'get frames'
            '''
            # 这一步的读取数据，应该改为从队列当中读取
            ret_val, frame = self.webcam.read()

            # ret_val == False if unable to read from webcam
            if not ret_val:
                print("ERROR: Unable to read from webcam. Exiting.")
                self.shut_down(self.webcam)
            '''
            ###############################################
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
            faces = self.detector(frame, 0)      # 得到关键点坐标（左上角和右下角）
            # print("faces type: ", type(faces))
            # test_faces = np.array(faces[0])
            # print("faces size: ", len(test_faces))
            # print("test_faces type: ", type(test_faces))
            # print(test_faces)
            if len(faces) == 1:     # faces表示人脸的矩阵方框的范围，只允许一个人脸
                face_points = self.predictor(frame, faces[0])    # frame: 图像, faces[0]: 左上角和右下角的坐标，框定人脸位置
                # 计算得到了一个avg数值（通过对绿色通道的平均数值计算而来）
                ####################################################
                # 如果要###########进行欧拉放大，可以考虑在这个位置附近进行
                # 在这里进行图像的欧拉放大，然后返回一个已经放大了色彩的 new_frame
                # 让 frame = new_frame，其他的不用改动
                #########################################
                #########################################
                #########################################
                # 这里要使用  一张  frame
                #########################################
                #########################################
                roi_avg = self.hrc.get_roi_avg(frame=frame, view=view,
                                               face_points=face_points, draw_rect=True)
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
                    filtered = self.hrc.filter_signal_data(values=roi_avg_values, fps=fps)

                    graph_values.append(filtered[-1])   # append最后一个值，，，用来draw pulse graph
                    if len(graph_values) > self.MAX_VALUES_TO_GRAPH:
                        graph_values.pop(0)

                    # Draw the pulse graph
                    graph = self.draw_graph(graph_values, graph_width, graph_height)    # 画心率表格
                    # Compute and display the BPM
                    bpm = self.hrc.compute_bpm(filtered, fps, curr_buffer_size, last_bpm)
                    # 显示心率的数字
                    bpm_display = self.draw_bpm(str(int(round(bpm))), bpm_display_width, graph_height)
                    last_bpm = bpm
                    # Display the FPS
                    if self.DEBUG_MODE:
                        view = self.draw_fps(view, fps)     # 展示一秒多少张frames
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

            cv2.imshow(self.window_title, view)

            key = cv2.waitKey(1)
            # Exit if user presses the escape key
            if key == 27:
                self.shut_down(self.webcam)


if __name__ == "__main__":
    HRC_ = HRC.heartRateComputation()
    HRE = heartRateExperiments(HRC_)
    # run(HRE)
