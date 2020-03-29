# encoding: utf-8
'''
Created on Dec 18, 2018

@author: Yongrui Huang
'''

import time
from array import *
from ctypes import *
from sys import exit
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np


class EmotivDeviceReader(object):
    '''
    classdocs
    This class is used to read EEG data from emotiv
    Attributes:
        queue: the queue save EEG data
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.queue = Queue(maxsize=-1)
        # num_EDR = 0  # 记录创建了多少个EmotivDeviceReader
        self.num_start = 0  # 记录start了多少个线程

    def test(self):
        print("real_time_detection.GUI.EmotivDeviceReader.py now test.")
        print("test test test test test")

    # check_status(self)
    def check_status(self):

        print("EmotivDeviceReader.py.check_status(self).start...")
        '''
        check if the device is connect correctly, if not, exit this process
        '''

        if self.libEDK.IEE_EngineConnect(create_string_buffer(b"Emotiv Systems-5")) != 0:
            print("Failed to start up Emotiv Engine.")
            exit()
        else:
            print("Successfully start up Emotiv Engine.")
        print("EmotivDeviceReader.py.check_status(self).end...")
    # check_status(self)

    # loop(self)
    def loop(self):
        print("EmotivDeviceReader.py..loop(self).start...")
        '''
        the loop is used to continuously read data from device
        '''
        try:
            self.libEDK = cdll.LoadLibrary("win64/edk.dll")
        except Exception as e:
            print('Error: cannot load EDK lib:', e)
            exit()
        print("EmotivDeviceReader.py...successfully connect")

        self.IEE_EmoEngineEventCreate = self.libEDK.IEE_EmoEngineEventCreate
        self.IEE_EmoEngineEventCreate.restype = c_void_p
        self.eEvent = self.IEE_EmoEngineEventCreate()
        # print("self.eEvent = self.IEE_EmoEngineEventCreate()")

        self.IEE_EmoEngineEventGetEmoState = self.libEDK.IEE_EmoEngineEventGetEmoState
        self.IEE_EmoEngineEventGetEmoState.argtypes = [c_void_p, c_void_p]
        self.IEE_EmoEngineEventGetEmoState.restype = c_int
        # print("self.IEE_EmoEngineEventGetEmoState.restype = c_int")
        
        self.IEE_EmoStateCreate = self.libEDK.IEE_EmoStateCreate
        self.IEE_EmoStateCreate.restype = c_void_p
        self.eState = self.IEE_EmoStateCreate()
        # print("self.eState = self.IEE_EmoStateCreate()")
        
        self.IEE_EngineGetNextEvent = self.libEDK.IEE_EngineGetNextEvent
        self.IEE_EngineGetNextEvent.restype = c_int
        self.IEE_EngineGetNextEvent.argtypes = [c_void_p]
        # print("self.IEE_EngineGetNextEvent.argtypes = [c_void_p]")
        
        self.IEE_EmoEngineEventGetUserId = self.libEDK.IEE_EmoEngineEventGetUserId
        self.IEE_EmoEngineEventGetUserId.restype = c_int
        self.IEE_EmoEngineEventGetUserId.argtypes = [c_void_p , c_void_p]
        # print("self.IEE_EmoEngineEventGetUserId.argtypes = [c_void_p , c_void_p]")
        
        self.IEE_EmoEngineEventGetType = self.libEDK.IEE_EmoEngineEventGetType
        self.IEE_EmoEngineEventGetType.restype = c_int
        self.IEE_EmoEngineEventGetType.argtypes = [c_void_p]
        # print("self.IEE_EmoEngineEventGetType.argtypes = [c_void_p]")
        
        self.IEE_EmoEngineEventCreate = self.libEDK.IEE_EmoEngineEventCreate
        self.IEE_EmoEngineEventCreate.restype = c_void_p
        # print("self.IEE_EmoEngineEventCreate.restype = c_void_p")
        
        self.IEE_EmoEngineEventGetEmoState = self.libEDK.IEE_EmoEngineEventGetEmoState
        self.IEE_EmoEngineEventGetEmoState.argtypes = [c_void_p, c_void_p]
        self.IEE_EmoEngineEventGetEmoState.restype = c_int
        # print("self.IEE_EmoEngineEventGetEmoState.restype = c_int")

        self.IEE_EmoStateCreate = self.libEDK.IEE_EmoStateCreate
        self.IEE_EmoStateCreate.argtype = c_void_p
        self.IEE_EmoStateCreate.restype = c_void_p
        # print("self.IEE_EmoStateCreate.restype = c_void_p")
        
        self.IEE_FFTSetWindowingType = self.libEDK.IEE_FFTSetWindowingType
        self.IEE_FFTSetWindowingType.restype = c_int
        self.IEE_FFTSetWindowingType.argtypes = [c_uint, c_void_p]
        # print("self.IEE_FFTSetWindowingType.argtypes = [c_uint, c_void_p]")
        
        self.IEE_GetAverageBandPowers = self.libEDK.IEE_GetAverageBandPowers
        self.IEE_GetAverageBandPowers.restype = c_int
        self.IEE_GetAverageBandPowers.argtypes = [c_uint, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
        # print("self.IEE_GetAverageBandPowers.argtypes = [c_uint, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]")

        self.IEE_EngineDisconnect = self.libEDK.IEE_EngineDisconnect
        self.IEE_EngineDisconnect.restype = c_int
        self.IEE_EngineDisconnect.argtype = c_void_p
        # print("self.IEE_EngineDisconnect.argtype = c_void_p")
        
        self.IEE_EmoStateFree = self.libEDK.IEE_EmoStateFree
        self.IEE_EmoStateFree.restype = c_int
        self.IEE_EmoStateFree.argtypes = [c_void_p]
        # print("self.IEE_EmoStateFree.argtypes = [c_void_p]")
        
        self.IEE_EmoEngineEventFree = self.libEDK.IEE_EmoEngineEventFree
        self.IEE_EmoEngineEventFree.restype = c_int
        self.IEE_EmoEngineEventFree.argtypes = [c_void_p]
        # print("self.IEE_EmoEngineEventFree.argtypes = [c_void_p]")
        
        self.check_status()
        print("EmotivDeviceReader.py...self.check_status()...")
        
        userID = c_uint(0)
        user   = pointer(userID)
        ready  = 0
        state  = c_int(0)
        
        alphaValue     = c_double(0)
        low_betaValue  = c_double(0)
        high_betaValue = c_double(0)
        gammaValue     = c_double(0)
        thetaValue     = c_double(0)
        
        alpha     = pointer(alphaValue)
        low_beta  = pointer(low_betaValue)
        high_beta = pointer(high_betaValue)
        gamma     = pointer(gammaValue)
        theta     = pointer(thetaValue)
        channelList = array('I', [3, 7, 9, 12, 16])   # IED_AF3, IED_AF4, IED_T7, IED_T8, IED_Pz

        loop_times = 0  # count how many times did while(1) run
        # while(1)
        while(1):
            loop_times += 1
            state = self.IEE_EngineGetNextEvent(self.eEvent)
            
            data = []
            if state == 0:
                eventType = self.IEE_EmoEngineEventGetType(self.eEvent)
                self.IEE_EmoEngineEventGetUserId(self.eEvent, user)
                if eventType == 16:  # libEDK.IEE_Event_enum.IEE_UserAdded
                    ready = 1
                    self.IEE_FFTSetWindowingType(userID, 1);  # 1: libEDK.IEE_WindowingTypes_enum.IEE_HAMMING
                    print("User added")
                            
                if ready == 1:
                    for i in channelList:
                        result = c_int(0)
                        result = self.IEE_GetAverageBandPowers(userID, i, theta, alpha, low_beta, high_beta, gamma)
                        
                        if result == 0:    # EDK_OK
                            print("theta: %.6f, alpha: %.6f, low beta: %.6f, high beta: %.6f, gamma: %.6f \n" %
                                  (thetaValue.value, alphaValue.value, low_betaValue.value,
                                   high_betaValue.value, gammaValue.value))
                            one_read_data = [thetaValue.value, alphaValue.value,
                                             low_betaValue.value, high_betaValue.value, gammaValue.value]
                            if len(one_read_data) > 0:
                                data += one_read_data
            elif state != 0x0600:
                print("Internal error in Emotiv Engine ! ")
            
            if len(data) > 0:
                self.queue.put(np.array(data))

            # --------------- #
            # sleep_time = 0.5
            # print("sleep(%f)" % sleep_time)
            # print("loop_times(%d)" % loop_times)
            # time.sleep(sleep_time)

            # if loop_times >= 50:
            #     break

        # while(1)

        print("EmotivDeviceReader.py..loop(self).end...")
        return 0
    # loop(self)
            
    def start(self):
        '''
        start a sub-process
        '''
        print("sub_process")
        self.num_start += 1
        print("num_start: %d " % self.num_start)
        sub_process = Process(target=self.loop)    # self.loop is the loop(self) function above
        print("sub_process.start().start")
        sub_process.start()
        print("sub_process.start().end")
        #error when run __main__ in the tool.py
        '''
        line 204, in start
    sub_process.start()
        '''

    def get_data(self):
        '''
        read psd data
        Returns:
        theta, alpha, low_beta, high_beta, gamma in order
        IED_AF3, IED_AF4, IED_T7, IED_T8, IED_Pz in order
        
        '''
        print("EmotivDeviceReader.get_data().start...")
        data_list = []
        while self.queue.qsize() > 0:
            ele = self.queue.get()
            data_list.append(ele)
        print("data_list[0]")
        print(data_list[0])
        print("data_list[1]")
        print(data_list[1])
        # print(data_list[2])
        print("EmotivDeviceReader.get_data().end...")
        return data_list
    

# __main__
if __name__ == '__main__':
    print("EmotivDeviceReader.py..__main__.start...")
    device_reader = EmotivDeviceReader()
    print("device_reader.start()")
    device_reader.start()
    print("device_reader.start()")
    time.sleep(5)
    print("for 5 loop: data")
    for i in range(5):
        print("i:%d" % i)
        data = device_reader.get_data()
        data = np.array(data)
        print(data)
        time.sleep(1)
    print("EmotivDeviceReader.py..__main__.end...")
# __main__
