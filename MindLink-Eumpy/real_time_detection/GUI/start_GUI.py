# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:10:53 2018

@author: Yongrui Huang
"""

import sys
sys.path.append('../../')
from flask import Flask

import _thread
import time
import configuration


# from real_time_detection.GUI import bind

def open_webbrower(thread_name, delay):
    '''
        Open the browser with another thread
        
        Arguments:
        
            thread_name: the name of this thread
            
            delay: how many second do the thread sleep before opening the browser
    '''
    
    time.sleep(delay)
    import webbrowser
    webbrowser.open('http://127.0.0.1:5000/')


# class RealtimeEmotionDetection()
class RealtimeEmotionDetection():
    
    '''
        This class is used for data collection
        
        Attributes:
        
            app: the flask app
            
            
    '''

    # __init__(self)
    def __init__(self):
        print("real_time_detection.GUI.start_GUI.py..RealtimeEmotionDetection().__init__(self).start...")
        '''
            initialize recorders and bind the app to backend
        '''

        # create a Flask obj
        self.app = Flask(__name__)

        print("self.app:")
        print(self.app)

        self.frames = []

        self.EEGs = []

        print("import bind")
        # import real_time_detection.GUI.bind as bind
        # from . import bind
        from real_time_detection.GUI import bind    # why should we import bind here but not the top of text?
        print("successfully import bind")

        # bind static html
        bind.bind_html(self)
        
        # bind post request
        bind.bind_post(self)

        print("real_time_detection.GUI.start_GUI.py..RealtimeEmotionDetection().__init__(self).end...")
        # return None
    # __init__(self)

    # start(self)
    def start(self):
        print("real_time_detection.GUI.start_GUI.py..RealtimeEmotionDetection().start(self).start...")
        '''
            start GUI, first we open the browser then we run the server
        '''
        
        try:
           _thread.start_new_thread(open_webbrower, ("open_webbrower", 0))
        except:
           pass
       
        self.app.run(debug=False, port=configuration.PORT)

        print("real_time_detection.GUI.start_GUI.py..RealtimeEmotionDetection().start(self).end...")
        return 0
    # start(self)
# class RealtimeEmotionDetection()


# __main__
if __name__ == '__main__':
    print("real_time_detection.py..start_GUI.start...")

    realtime_emotion_detection_obj = RealtimeEmotionDetection()
    print("realtime_emotion_detection_obj = RealtimeEmotionDetection() finish")
    realtime_emotion_detection_obj.start()

    print("real_time_detection.py..start_GUI.end...")
# __main__
