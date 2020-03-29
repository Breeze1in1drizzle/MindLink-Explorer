# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:42:40 2018 by Yongrui Huang
Modified on Tue Oct 29th 16:16:50 2019 by Ruixin Lee
@author: Yongrui Huang
"""

import sys
sys.path.append('../')
from data_collection_framework.util import record
import configuration

import os
import platform
import time
import ctypes

from array import *
from ctypes import *
import multiprocessing


if sys.platform.startswith('win32'):
    import msvcrt
elif sys.platform.startswith('linux'):
    import atexit
    from select import select


from ctypes import *
import numpy as np


class EEGRecorder(record.AbstractRecorder):
    '''
        This class gives an example of how to use AbsRecorder to record EEG data.
        Specially, the devices we used is emotiv insight.
        It should be noted that this code may not work well for your envirnoment, 
      since the device (emotiv insight) is relied on different platform and 
      different version of progarm language. I used windows64 ana python 3.5 
      for my development.
        You may noticed that the code here is very complex. However, I just copied
      the code that emotiv company released and modify just a litte bit, basically.

        Oct 29 2019
        Now we use Emotiv Epoc+ to record EEG signals which has 14 channels.
    '''
    def __init__(self, name):
        record.AbstractRecorder.__init__(self, name)
    
    def record_one_sample(self):
        '''
                This method is supposed to read one data sample from information
            source. For EEG using emotiv insight, it means recording signal from
            5 different channel (i.e, IED_AF3, IED_AF4, IED_T7, IED_T8, IED_Pz) and 
            each channel's PSD features (i.e. theta, alpha, low_beta, high_beta, gamma).
            25 features totally to be treated as a sample.

                Oct 29 2019
                Now we use Emotiv Epoc+ with 14 channels to record EEG signals.
                My mission is to complete the connection method and make it more convenient
            for others and suitable for all kinds of platforms.

        '''
        #------------------------------------
        print("configuration.ROOT_PATH:")
        print(configuration.ROOT_PATH)
        print("configuration.ROOT_PATH.")
        # load_str = configuration.ROOT_PATH + "data_collection_frame/util/win64/edk/1.edk.dll"
        print("load_str:")
        # print(load_str)
        print("load_str.")
        #--------------------------------------

        # self.libEDK = cdll.LoadLibrary(configuration.ROOT_PATH + "data_collection_frame/util/win64/edk.dll")
        # load sdk
        self.libEDK = cdll.loadLibrary("win64/edk.dll")
        # print("self.libEDK")
        # print(self.libEDK)
        # print("self.libEDK")

        # cdll.LoadLibrary()???
        # self.libEDK = cdll.LoadLibrary(load_str)
        # print("self.libEDK")
        # print(self.libEDK)
        # print("self.libEDK")

        # Create an Emotiv Engine Event
        IEE_EmoEngineEventCreate = self.libEDK.IEE_EmoEngineEventCreate
        IEE_EmoEngineEventCreate.restype = c_void_p
        self.eEvent = IEE_EmoEngineEventCreate()
        IEE_EmoEngineEventGetEmoState = self.libEDK.IEE_EmoEngineEventGetEmoState
        IEE_EmoEngineEventGetEmoState.argtypes = [c_void_p, c_void_p]
        IEE_EmoEngineEventGetEmoState.restype = c_int
        IEE_EmoStateCreate = self.libEDK.IEE_EmoStateCreate
        IEE_EmoStateCreate.restype = c_void_p
        eState = IEE_EmoStateCreate()
        
        userID = c_uint(0)
        user   = pointer(userID)
        ready  = 0
        state  = c_int(0)
        IEE_EngineConnect = self.libEDK.IEE_EngineConnect
        
        #add code here
        IEE_EngineConnect.restype = c_int
        IEE_EngineConnect.argtypes = [c_void_p]
        
        IEE_EngineGetNextEvent = self.libEDK.IEE_EngineGetNextEvent
        IEE_EngineGetNextEvent.restype = c_int
        IEE_EngineGetNextEvent.argtypes = [c_void_p]
        
        IEE_EmoEngineEventGetUserId = self.libEDK.IEE_EmoEngineEventGetUserId
        IEE_EmoEngineEventGetUserId.restype = c_int
        IEE_EmoEngineEventGetUserId.argtypes = [c_void_p , c_void_p]
        
        IEE_EmoEngineEventGetType = self.libEDK.IEE_EmoEngineEventGetType
        IEE_EmoEngineEventGetType.restype = c_int
        IEE_EmoEngineEventGetType.argtypes = [c_void_p]
        
        IEE_EmoEngineEventCreate = self.libEDK.IEE_EmoEngineEventCreate
        IEE_EmoEngineEventCreate.restype = c_void_p
        
        IEE_EmoEngineEventGetEmoState = self.libEDK.IEE_EmoEngineEventGetEmoState
        IEE_EmoEngineEventGetEmoState.argtypes = [c_void_p, c_void_p]
        IEE_EmoEngineEventGetEmoState.restype = c_int
        
        IEE_EmoStateCreate = self.libEDK.IEE_EmoStateCreate
        IEE_EmoStateCreate.argtype = c_void_p
        IEE_EmoStateCreate.restype = c_void_p
        
        IEE_FFTSetWindowingType = self.libEDK.IEE_FFTSetWindowingType
        IEE_FFTSetWindowingType.restype = c_int
        IEE_FFTSetWindowingType.argtypes = [c_uint, c_void_p]
        
        IEE_GetAverageBandPowers = self.libEDK.IEE_GetAverageBandPowers
        IEE_GetAverageBandPowers.restype = c_int
        IEE_GetAverageBandPowers.argtypes = [c_uint, c_int, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p]
        
        IEE_EngineDisconnect = self.libEDK.IEE_EngineDisconnect
        IEE_EngineDisconnect.restype = c_int
        IEE_EngineDisconnect.argtype = c_void_p
        
        IEE_EmoStateFree = self.libEDK.IEE_EmoStateFree
        IEE_EmoStateFree.restype = c_int
        IEE_EmoStateFree.argtypes = [c_void_p]
        
        IEE_EmoEngineEventFree = self.libEDK.IEE_EmoEngineEventFree
        IEE_EmoEngineEventFree.restype = c_int
        IEE_EmoEngineEventFree.argtypes = [c_void_p]
        # finish adding code

        # init frequency dataf
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
        
        channelList = array('I',[3, 7, 9, 12, 16])   # IED_AF3, IED_AF4, IED_T7, IED_T8, IED_Pz 
        if self.libEDK.IEE_EngineConnect(create_string_buffer(b"Emotiv Systems-5")) != 0:
            print(self.libEDK.IEE_EngineConnect("Emotiv Systems-5"))
            print("Emotiv Engine start up failed.")
            return
     
        state =  IEE_EngineGetNextEvent(self.eEvent)
        
        if state == 0:
            
            eventType = IEE_EmoEngineEventGetType(self.eEvent)
            IEE_EmoEngineEventGetUserId(self.eEvent, user)
                
            if eventType == 64:  # self.libEDK.IEE_Event_enum.IEE_UserAdded
                ready = 1
                self.libEDK.IEE_FFTSetWindowingType(userID, 1);
                # 1: self.libEDK.IEE_WindowingTypes_enum.IEE_HAMMING
                             
            if ready == 1:
                EEG_row = np.zeros(25)
                j = 0
                for i in channelList:
                    result = c_int(0)
                    result = self.libEDK.IEE_GetAverageBandPowers(
                        userID, i, theta, alpha, low_beta, high_beta, gamma
                    )
                    if result == 0:    # EDK_OK
                        EEG_row[j*5+0], EEG_row[j*5+1], EEG_row[j*5+2], EEG_row[j*5+3], EEG_row[j*5+4] = (
                                thetaValue.value, alphaValue.value, low_betaValue.value,
                                high_betaValue.value, gammaValue.value
                        )
                    j += 1
                    
                return EEG_row
        elif state != 0x0600:
            print("Internal error in Emotiv Engine ! ")
        else:
            print('Noe event for EEG device!')

    def release_resourse_in_one_trial(self):
        pass
    

if __name__ == '__main__':
    print("main.start")
    eeg_recorder = EEGRecorder('EEG')
    for i in range(100):
        print("i: %d" % i)
        print(eeg_recorder.record_one_sample())
    print("main.end")
