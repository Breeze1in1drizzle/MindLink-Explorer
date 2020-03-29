# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:56:13 2018

@author: Yongrui Huang
"""

import sys
sys.path.append('../')

from data_collection_framework.util import record
import cv2


# class FacialExpressionRecorder(record.AbstractRecorder)
class FacialExpressionRecorder(record.AbstractRecorder):
    
    '''
          This class gives an example of how to use AnstractRecorder to record 
        faces data.
          We just need to implement the function 'record_one_sample', to collect
        one sample of data from camera. Please notes that here we collected the 
        whole image without finding one's face, if you intend to use the collected,
        you will have to find the faces from the image using opencv or other tools.
          As for release resourse, we have no resourse to release here.
    '''
    
    def __init__(self, name):
        record.AbstractRecorder.__init__(self, name)

    # record_one_sample(self)
    def record_one_sample(self):
        '''
                This method is supposed to read one data sample from information
            source. For facial expression, we use opencv API to read one  image
            from camera. In my demo, the camera is the standard camera from my 
            laptop. The steps are simply as follows.
                1. opencv a handle
                2. read an image from the handle
                3. return the image
        '''
        
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        
        return frame
    # record_one_sample(self)
        
    def release_resourse_in_one_trial(self):
        pass

# class FacialExpressionRecorder(record.AbstractRecorder)
