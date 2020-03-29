import cv2
import os
import configuration
import queue
import random
import string

import tensorflow as tf

from real_time_detection.GUI import FaceFeatureReader


# class FaceReader
class FaceReader:
    '''
    This class is used to return the face data in real time.

    Attribute:
        cap: the capture stream
        faceCascade: model for detecting where the face is.
        file_name: the file name of the current frame in hard disk
        delete_queue: the queue is used to save all the delete file name
        faces: the faces for predicting the emotion, we used a set of face
        rather than one face.
    '''

    def __init__(self, input_type, file_path=None, face_feature_reader_obj=None):
        '''
        Arguments:
            input_type: 'file' indicates that the stream is from file. In other
            case, the stream will from the defalt camera.
        '''
        # graph needs to be modified
        tf.compat.v1.get_default_graph()
        self.face_feature_reader_obj = face_feature_reader_obj
        # self.graph = graph #tf.compat.v1.get_default_graph()  # face_feature_reader_obj
        # self.face_feature_reader_obj = FaceFeatureReader(self.graph)
        self.input_type = input_type
        if input_type == 'file':
            self.cap = cv2.VideoCapture(file_path)
        else:
            self.cap = cv2.VideoCapture(0)
            ret, frame = self.cap.read()

        cascPath = configuration.MODEL_PATH + "haarcascade_frontalface_alt.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.file_name = None
        self.delete_queue = queue.Queue()
        self.faces = []

    def delete_files(self):
        '''
        delete files for releasing the resourse.
        '''
        print("delete_files()")
        while self.delete_queue.qsize() > 10:
            file = self.delete_queue.get()
            if (os.path.exists(file)):
                os.remove(file)

    def get_one_face(self):
        '''
        Returns:
            one face from stream.
        '''
        print("get_one_face().start...")
        if self.input_type == 'file':
            cnt = 0
            while cnt < 15:
                self.cap.read()
                cnt += 1
        ret, frame = self.cap.read()
        print("ret, frame:")
        print(ret)
        print(frame)

        if ret is True:
            print("ret is True")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (x, y, w, h) = self.detect_face(gray)
            if (w != 0):
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                self.faces.append(face)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            if self.file_name is not None:
                print("self.file_name is not None")
                del_file_name = 'static/cache_image/%s.png' % self.file_name
                self.delete_queue.put(del_file_name)

                if self.delete_queue.qsize() > 50:
                    self.delete_files()

            self.file_name = ''.join(random.sample(string.ascii_letters + string.digits, 12))

            cv2.imwrite('static/cache_image/%s.png' % self.file_name, frame)
            print("get_one_face().end...")
            return self.file_name
        else:
            print("ERROR")
            print("get_one_face().end...")
            return 'ERROR'

    def detect_face(self, gray):
        '''
        find faces from a gray image.
        Arguments:
            gray: a gray image
        Returns:
            (x, y, w, h)
            x, y: the left-up points of the face
            w, h: the width and height of the face
        '''
        print("detect_face()")
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(32, 32)
        )
        print("faces:")
        print(faces)
        print("faces' len:")
        if len(faces) > 0:
            print(len(faces))
            (x, y, w, h) = faces[0]
            print("faces[0]:")
            print((x, y, w, h))
        else:
            print("0")
            (x, y, w, h) = (0, 0, 0, 0)

        return (x, y, w, h)

    def read_face_feature(self):
        '''
        Returns:
            items: a list, the first element is the frame path while the rest
            is the feature map.
        '''
        print("read_face_feature()")
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = self.detect_face(gray)
        if (w != 0):
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            self.face_feature_reader_obj.set_face(face)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        random_str = ''.join(random.sample(string.ascii_letters + string.digits, 12))
        frame_path = 'static/cache_image/%s.png' % random_str
        cv2.imwrite(frame_path, frame)

        feature_map_list = self.face_feature_reader_obj.read_feature_map()
        items = [frame_path, ]
        items += feature_map_list

        self.delete_queue.put(frame_path)
        if self.delete_queue.qsize() > 10:
            self.delete_files()
        return items
# class FaceReader