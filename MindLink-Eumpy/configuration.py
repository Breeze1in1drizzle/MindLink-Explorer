# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:49:04 2018

@author: Yongrui Huang
"""

import os
abs_path = os.path.abspath(__file__)

# this variable is the root path in Eumpy
ROOT_PATH = abs_path.replace('\\', '/')[:-16]

# this variable decide where the dataset is
DATASET_PATH = ROOT_PATH + 'dataset/'

# this variable decide where the model is saved
MODEL_PATH = ROOT_PATH + 'model/'

# this variable decide where the data collected by data collection framework is saved
COLLECT_DATA_PATH = DATASET_PATH + 'collected_dataset/'

# this variable describes the port the flask will take (in data collection framework)
PORT = 5000


