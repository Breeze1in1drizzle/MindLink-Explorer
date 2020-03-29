# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:38:13 2018

@author: Yongrui Huang

    Here we call participants "subject"

          This package provides a GUI for collecting online data in an emotion experiment.
      The process of the emotion experiment is as follows.
      
      The experiment consists of several trials.
      
      1. At the beginning of each trial, there is a 10-second countdown in the
    center of the screen to capture each subject’s attention and to serve as 
    an indicator of the next clip. 
      
      2. After the countdown is complete, movie clips that include different 
    emotional states are presented on the full screen. While the movie clips
    are playing, we record the signals of the subjects.
      
      3. After each trial ends, a SAM form will appear to let subject evaluate their emotion
    of which the subject is watching clips.
      
      Both signals and emotion ground truth labels will be saved in the
    hard disk.
    
      In this package, we collected EEG and facial expression data as examples.
      
      Here follow some instructions about how to use this framework.
      
      1. If you just want to collect EEG and facial expression dataset. just run
    start_GUI.py using command 'python start_GUI.py' and it will start the experiment.
      
      2. If you want to use it in your own code. You can follow the example 
    presented in 'start_GUI.py'. Firstly, you can create a 'DataCollection' object,
    then, add some 'recorder' to it. Finally, you call the method start().
    As for the 'recorder', please see 3.
    
      3. If you want to record different singnal, for example, ECG, you will have 
    to create a new class who inheirs 'AbstractRecorder' in 'util'. Please see 
    the class document in 'AbstractRecorder', where it will give you more detail 
    about how to create a recorder.
    
      Here it will show you how this framework works.
      
      1. When you call start() method in the DataCollection class. It all begins.
      
      2. An personal information form comes out. It requires the test subject 
    for their personal information and those information will be saved as well. 
    
      3. When the 'start Testing' button is clicked, one trial starts.
      
      4. In the beginning of each trials, a 10-s countdown page appears. 
      
      5. After the countdown, the corresponding video is presented. In the mean 
    time, the recorder start recording the data.
    
      6. When the video is over, the recorder stops recording and a page with a SAM 
    form appears. This page requires the subject repoert their emotion status. After 
    clicking 'Submit' button, if there is still some trial left, return to 4, else 
    go to 7.
    
      7. A finish page will appear and 3-s countdown will start. When the countdown 
    is over, the GUI will close and the program will quit.
    
     This script gives an example of how to use this data collection framework.
"""

import sys
sys.path.append('..')
from data_collection_framework import data_collection
from data_collection_framework.util import face_record
from data_collection_framework.util import EEG_record


def the_start():
      print("start_GUI.py..__main__.start...")
      obj = data_collection.DataCollection()
      recorders = [
            face_record.FacialExpressionRecorder('faces'),
            EEG_record.EEGRecorder('EEG')
      ]
      print("recorders")
      obj.add_recorders(recorders)
      print("obj.start...")
      obj.start()
      print("start_GUI.py..__main__.end...")


if __name__ == '__main__':
      print("start_GUI.py..__main__.start...")
      obj = data_collection.DataCollection()

      # create recorders: EEG_recorder and face_recorder
      recorders = [
            face_record.FacialExpressionRecorder('faces'),
            EEG_record.EEGRecorder('EEG')
      ]
      print("recorders")
      obj.add_recorders(recorders)
      print("obj.start...")
      obj.start()
      print("start_GUI.py..__main__.end...")
