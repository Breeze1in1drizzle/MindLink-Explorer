# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:52:12 2018

@author: Yongrui Huang
"""

import win32api
import win32con
from flask import render_template


def bind_html(data_collection_obj):
    '''
        bind all static html with app, use 'GET' mode return the html to the 
        front end
    '''
    
    app = data_collection_obj.app
    
    @app.route('/')
    @app.route('/information')
    def information(name=None):
        return render_template('information.html', name=name)
    
    @app.route('/count_down.html')
    def count_down(name=None):
        return render_template('count_down.html', name=name)
    
    @app.route('/fill_label_SAM.html')
    def fill_label_SAM(name=None):
        return render_template('fill_label_SAM.html', name=name)

    @app.route('/finish.html')
    def finish(name=None):
        return render_template('finish.html', name=name)
    
    @app.route('/show_stimul_video.html')
    def show_stimul_video(name=None):
        return render_template('show_stimul_video.html', name=name)


from flask import request
from data_collection_framework.util import creat_subject
from data_collection_framework.util import GUI_control
import configuration


def bind_post(data_collection_obj):
    '''
        bind all post with app
    '''
    app = data_collection_obj.app

    @app.route('/full_screen', methods=['GET', 'POST'])
    def full_screen():
        '''
            When opening the bower, we push F11 to have the full screen. 
        '''
        win32api.keybd_event(122, 0, 0, 0)
        win32api.keybd_event(122, 0, win32con.KEYEVENTF_KEYUP, 0)
        return '1'
        
    @app.route('/post_information', methods=['GET', 'POST'])
    def post_information():
        '''
            It happens when the subject finish fulling the information form
        '''
        if request.method == 'POST':
            data = request.get_json(force=True)
            name = data['name']
            age = data['age']
            gender = data['gender']
            trial_num = data['trial_num']
            creat_subject.creat_subject(name, age, gender, trial_num)
            return '1'
        return '0'

    @app.route('/start_trial', methods=['GET', 'POST'])
    def start_trial():
        '''
            It happens when one trial start
        '''
        data = request.get_json(force=True)
        for recorder in data_collection_obj.recorders:
            recorder.start_one_trial(data['trial_id'])
        return '1'
    
    @app.route('/end_trial', methods=['GET', 'POST'])
    def end_trial():
        '''
            It happens when one trial end
        '''
        for recorder in data_collection_obj.recorders:
            recorder.end_one_trial()
         
        return '1'

    @app.route('/post_SAM_label', methods=['GET', 'POST'])
    def post_SAM_label():
        '''
            It happens when subjects finishs filling their SAM form.
        '''
        print("post_SAM_label.start...")
        data = request.get_json(force=True)
        trial_id = data['trial_id']
        valence = data['valence']
        arousal = data['arousal']
        creat_subject.save_SAM_label(trial_id, valence, arousal)
        print("post_SAM_label.end...")
        return '1'
    
    @app.route('/experiment_over', methods=['GET', 'POST'])
    def experiment_over():
        '''
            It happens when the whole experiment is over.
            Two thing is done by this method.
            1. close the browser.
            2. finish the whole program in cmd.
        '''

        GUI_control.close_browser()
        GUI_control.kill_process_by_port(str(configuration.PORT))
        return '1'
