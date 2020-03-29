# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:32:55 2018

@author: Yongrui Huang
"""

from flask import render_template
print("from flask import render_template")
# import tool
# from . import tool
from real_time_detection.GUI import tool
# from real_time_detection.GUI.MLE_tool import mytool
print("from real_time_detection.GUI import tool")
# print("import tool")
import numpy as np
print("import numpy as np")
from flask import jsonify


# def bind_html(realtime_emotion_detection)
def bind_html(realtime_emotion_detection):
    '''
        bind all static html with app, use 'GET' mode return the html to the 
        front end
    '''
        
    app = realtime_emotion_detection.app

    @app.route('/')
    @app.route('/index')
    def index(name=None):
        return render_template('index.html', name=name)
    
    @app.route('/EEG')
    def EEG(name=None):
        return render_template('EEG.html', name=name)
    
    @app.route('/video')
    def vedio(name=None):
        return render_template('video.html', name=name)
# def bind_html(realtime_emotion_detection)
     

# def bind_post(realtime_emotion_detection)
def bind_post(realtime_emotion_detection):
    '''
        bind all post with app
    '''
    app = realtime_emotion_detection.app

    @app.route('/get_faces_data', methods=['GET', 'POST'])
    def get_faces_data():
        '''
        Returns:
            one face from stream, the image is saved as file in hard disk, and
            return the file name.
        '''
        print("bind.py.get_faces_data().start...")
        face_reader_obj = tool.face_reader_obj      # mytool
        # face_reader_obj = mytool.GET_face_reader_obj()
        file_name = face_reader_obj.get_one_face()  # mytool
        print("bind.py.get_faces_data().end...")
        return str(file_name)
        # pass

    @app.route('/get_EEGs_data', methods=['GET', 'POST'])
    def get_EEGs_data():
        '''
        Returns:
            A string contain messages of 1 second EEG data.
            The string is separated by symbol '|'.
            The first position is the timestamp represets it is how much second.
            The second to six positions are the EEG data contains 5 channel and
            their data is separated by block.
        '''
        print("bind.py.get_EEGs_data().start...")
        EEG_read_obj = tool.EEG_reader_obj          # mytool   *****
        # EEG_read_obj = mytool.GET_EEG_reader_obj()
        EEG_datas, timestamp, PSD_features = EEG_read_obj.get_EEG_data()#error
        '''
        error:
        line 76, in get_EEGs_data
        EEG_datas, timestamp, PSD_features = EEG_read_obj.get_EEG_data()
        '''
        
        if PSD_features is None:    # it means the input type is 'file'
            EEG_datas = EEG_datas*100000
            
            # preprocess data for better looking in front end
            for i in range(5):
                EEG_datas[i] = EEG_datas[i] - EEG_datas[i].mean() + 5*i
        
        items = [str(timestamp), ]
                
        for channel_data in EEG_datas:
            items.append('|')
            items.append(str(channel_data)[1:-1])
        result = ''.join(items)
        
        print(EEG_datas.shape)
        print("bind.py.get_EEGs_data().end...")
        return result
    
    @app.route('/get_emotion_data', methods = ['GET', 'POST'])
    def get_emotion_data():
        '''
        Returns:
            A string contain emotion data.
            The string is separated by symbol '|'.
            The first position is the timestamp represents it is how much second.
            The second position is the valence value.
            The third position is the scores of 32 emotion, separated by block.
            The fourth position is the strength of the strength of the discrete
            emotion.
        '''
        print("bind.py.get_emotion_data().start...")
        '''
        error:
        line 107, in get_emotion_data
        The third position is the scores of 32 emotion, separated by block.
        '''
        # mytool
        timestamp, valence, arousal, distcrete_emotion, emotion_strength = tool.emotion_reader_obj.get_emotion_data() #error
        # timestamp, valence, arousal, distcrete_emotion, emotion_strength = mytool.GET_emotion_reader_obj().get_emotion_data()
        '''
        line 107, in get_emotion_data
    timestamp, valence, arousal, distcrete_emotion, emotion_strength = tool.emotion_reader_obj.get_emotion_data()
        '''
        result = ''.join([str(timestamp), '|', str(valence), '|', str(arousal), '|', str(distcrete_emotion)[1:-1], '|', str(emotion_strength)])
        print("bind.py.get_emotion_data().end...")
        return result
    
    @app.route('/get_EEG_PSD', methods=['GET', 'POST'])
    def get_EEG_PSD():
        '''
        This method for emotiv device only
        Returns:
            A string contain emotion data.
            The string is separated by space.
            20 units are in the string. They represents 
            theta, alpha, beta, gamma in order
            IED_AF3, IED_AF4, IED_T7, IED_T8, IED_Pz in order
            For instance, the first 4 units is theta, alpha, beta, gamma
            for AF3 channel.
        '''
        print("bind.py.get_EEG_PSD().start...")
        emotiv_reader = tool.emotiv_reader#;        # mytool
        # emotiv_reader = mytool.GET_emotiv_reader()
        data = emotiv_reader.get_data()
        data = np.array(data)
        data /= 10
        data_mean = np.mean(data, axis = 0)
        data_mean = data_mean
        psd_data = []
        for i in range(25):
            if (i - 2) % 5 != 0:
                psd_data.append(data_mean[i])
        psd_data = np.array(psd_data)
#         print(psd_data)
        print("bind.py.get_EEG_PSD().end...")
        return str(psd_data)[1:-1]
    
    @app.route('/get_face_features', methods = ['GET', 'POST'])
    def get_face_features():
        '''
        This method for emotiv device only
        separated by '|' 
        The first element is the frame path while the rest 
        is the feature map.
        '''
        print("bind.py.get_face_features().start...")
        face_reader_obj = tool.face_reader_obj      # mytool
        # face_reader_obj = mytool.GET_face_reader_obj()
        items = face_reader_obj.read_face_feature()
        print("bind.py.get_face_features().end...")
        return '|'.join(items)

# def bind_post(realtime_emotion_detection)
