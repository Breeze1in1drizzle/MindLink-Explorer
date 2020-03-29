print("real_time_detection.GUI.MLE_tool.test.py")
import numpy as np

from real_time_detection.GUI.MLE_tool.EmotionReader import EmotionReader
from real_time_detection.GUI.MLE_tool.EEGReader import EEGReader
from real_time_detection.GUI.MLE_tool.FaceReader import FaceReader

# from real_time_detection.GUI.EmotivDeviceReader import EmotivDeviceReader
# emotiv_reader = EmotivDeviceReader()
# emotiv_reader.start()

print("face_reader_obj")
face_reader_obj = FaceReader(input_type='')
print("EEG_reader_obj")
EEG_reader_obj = EEGReader(input_type='')
print("emotion_reader_obj")
emotion_reader_obj = EmotionReader(input_type='')


print("real_time_detection.GUI.MLE_tool.test.py.__main__")


if __name__ == '__main__':
    print("real_time_detection_tool.GUI.MLE_tool.test.py.__main__.start...")
    for i in range(5):
        for j in range(25):
            print("i:%d" % i)
            print("j:%d" % j)
            face_reader_obj.get_one_face()
            # print(face_reader_obj.get_one_face())
        # print(EEG_reader_obj.get_EEG_data())
        EEG_reader_obj.get_EEG_data()
        '''
        error:
        line 449, in <module>
    EEG_reader_obj.get_EEG_data()
        '''
        # print(emotion_reader_obj.get_emotion_data())
    print("real_time_detection_tool.GUI.MLE_tool.test.py.__main__.end...")

