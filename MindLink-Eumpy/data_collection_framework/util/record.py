# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:45:35 2018 by Yongrui Huang
Modified on Tue Oct 29th 16:00:30 2019 by Ruixin Lee
@author1: Yongrui Huang
@author2: Ruixin Lee
"""

from abc import ABCMeta, abstractmethod
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import configuration


class MyFirstPyClass(metaclass=ABCMeta):

    def __init__(self):
        pass

    def PyClass1(self):
        print("PyClass1.start......")
        print("PyClass1.end......")


class AbstractRecorder(metaclass=ABCMeta):
    '''
            If you want to create your own recorder. For example, you want to
        record ECG (Electrocardiogram), you just need to inherit this class 
        and implement two method (record_one_sample and release_resourse_in_one_trial).
        In most time, you just need to implement one method record_one_sample.
        
        Attributes:
            trial_id: the current id of the trial, it will change during recording
            
            queue: the queue for sharing data in different process.
            
            data_list: the list for saving data in RAM. Everytime one trial ends,
          its data will be written into hard disk and then be clear.
            
            sub_process: the sub-process for read data from sensor. We need the 
          father process to provide GUI during recording.
            
            save_file_name: the data file's name, a string
            
            base_bath: the path that saves the data
    '''

    def __init__(self, name):
        '''
            initialize the attribute
            
            Arguments:
            
                name: this name will eventually become the data file's name.
        '''
        self.trial_id = None
        self.queue = Queue(maxsize=-1)
        self.data_list = []   # needs to be rewrite
        self.sub_process = None
        self.save_file_name = None
        self.base_path = configuration.COLLECT_DATA_PATH
        self.save_file_name = name
        
    def record(self):
        '''
            This method starts in a new process.
            I use a loop to continue reading data into queue from sensors. It
          will be terminated when the trial ends.
        '''
        while True:
            self.queue.put(self.record_one_sample())
        
    def start_one_trial(self, trail_id):
        '''
            when one trial starts, we clear the queue and start a new process
            to read data.
            
            Aruguments:
            
                trial_id: the current trial's id
        '''
        self.trial_id = trail_id
        while self.queue.qsize() > 0:
            self.queue.get()
        self.sub_process = Process(target=self.record)
        self.sub_process.start()
        
    def end_one_trial(self):
        '''
            When one trial ends, we follow these steps.
            
            1. retrieve data from queue and put them into data_list.
            
            2. terminate the process.
            
            3. save the data in data_list into hard disk.
            
            4. reset data_list.
            
            5. release all resourse in one trial.
        '''
        
        # can't use 'queue.empty() is False' here, it will jump out of loop
        # before the queue is empty. stuck here for a long time but still don't konw why
        while self.queue.qsize() > 0:
            ele = self.queue.get()
            self.data_list.append(ele)
        print("self.data_list: \n", self.data_list)
       
        self.sub_process.terminate()
        # self.save()       # needs to be rewrite
        self.data_list = []
        self.release_resourse_in_one_trial()
    
    def save(self):
        '''
            save the data in data_list into hard disk
        '''
        subject_id = np.load('subject_id.npy')
        # path =\
        #     self.base_path + str(subject_id) + '/trial_'\
        #     + str(self.trial_id) + '/' + self.save_file_name + '.npy'
        path_save = "../dataset/collected_dataset/" + str(subject_id) + "/trial_"\
                    + str(self.trial_id) + "/" + self.save_file_name + ".npy"
        path_load = ""
        
        data_numpy_arr = np.array(self.data_list)
        
        np.save(path_save, data_numpy_arr)

    @abstractmethod
    def record_one_sample(self):
        '''
            This is the key method you should implement. It is used for reading
          one sample from your sensor. For example, you will return an image if
          your minimum data unit (one sample) in your model is an image. Similarly,
          you will return a video if your minimum data unit (one sample) in
          your model is a video. Please check how I implement it in face_record.py.
          
          Returns:
              One sample, the minimum data unit for training.
        '''
        
        pass
    
    @abstractmethod
    def release_resourse_in_one_trial(self):
        '''
          You will implement this method to release resourse when one trial ends.
        '''
        pass


if __name__ == '__main__':
    print("main...")
    print("main...")
    pass
