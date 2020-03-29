print("real_time_detection.GUI.MLE_tool.mytool.py")
import numpy as np


###############################################################
'''
return graph(tensorflow)
'''
print("tool.py...import tensorflow as tf")
def GET_graph():
    import tensorflow as tf
    return tf.compat.v1.get_default_graph()
# graph = tf.compat.v1.get_default_graph()


###############################################################
'''
return faceReader, EEGReader, emotionReader obj...
'''
print("face_reader_obj")
def GET_face_reader_obj():
    from real_time_detection.GUI.MLE_tool.FaceReader import FaceReader
    return FaceReader(input_type='')
# face_reader_obj = FaceReader(input_type='')
print("EEG_reader_obj")


def GET_EEG_reader_obj():
    from real_time_detection.GUI.MLE_tool.EEGReader import EEGReader
    return EEGReader(input_type='')
# EEG_reader_obj = EEGReader(input_type='')
print("emotion_reader_obj")


def GET_emotion_reader_obj():
    from real_time_detection.GUI.MLE_tool.EmotionReader import EmotionReader
    return EmotionReader(input_type='')
# emotion_reader_obj = EmotionReader(input_type='')
'''
return faceReader, EEGReader, emotionReader obj...
'''


###############################################################
'''
return emotiv reader(emotiv_reader.start())
'''
def GET_emotiv_reader():
    from real_time_detection.GUI.EmotivDeviceReader import EmotivDeviceReader
    emotiv_reader = EmotivDeviceReader()
    emotiv_reader.start()
    return emotiv_reader
'''
return emotiv reader(emotiv_reader.start())
'''


###############################################################


print("real_time_detection.GUI.MLE_tool.mytool.py.__main__")


if __name__ == '__main__':
    print("real_time_detection_tool.GUI.py.__main__.start...")
    face_reader_obj = GET_face_reader_obj()
    EEG_reader_obj = GET_EEG_reader_obj()
    emotion_reader_obj = GET_emotion_reader_obj()
    for i in range(5):
        for j in range(25):
            print("i:%d" % i)
            print("j:%d" % j)
            face_reader_obj.get_one_face()
            print(face_reader_obj.get_one_face())
        print(EEG_reader_obj.get_EEG_data())
        EEG_reader_obj.get_EEG_data()
        '''
        error:
        line 449, in <module>
    EEG_reader_obj.get_EEG_data()
        '''
        print(emotion_reader_obj.get_emotion_data())
    print("real_time_detection_tool.GUI.py.__main__.end...")

