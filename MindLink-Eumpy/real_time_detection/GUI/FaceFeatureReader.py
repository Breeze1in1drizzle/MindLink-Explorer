# encoding: utf-8
'''
Created on Dec 23, 2018

@author: Yongrui Huang
'''

import keras
import keras.layers as L
import cv2
import random
import string
import os
from bokeh.themes import default    # In which cases should I use bokeh in Eumpy?   From Ruixin Lee
import queue


# class FaceFeatureReader(object)
class FaceFeatureReader(object):
    '''
    class docs
        This class is used to see features in CNN model for faces
    
    '''
    # __init__(self, graph)
    def __init__(self, graph):
        '''
        Constructor
        '''
        # now we don't have .h5 model, so we can't use it(self.model).
        # self.model = keras.models.load_model('D:/eclipse/PythonWorkspaces/Eumpy/model/CNN_expression_baseline.h5')
        # self.model = keras.models.load_model("D:/PythonWorkPlaceALL/Eumpy-master/algorithm_implement/train_baseline_model_inFer2013/CNN_expression_baseline.h5")
        # self.model = keras.models.load_model("D:/workSpace/python_workspace/MindLink-Explorer/algorithm_implement/train_baseline_model_inFer2013/CNN_expression_baseline.h5")
        # self.model = keras.models.load_model("D:/workSpace/python_workspace/MindLink-Explorer/model/CNN_expression_baseline.h5")
        self.model = keras.models.load_model("D:/workSpace/python_workspace/MindLink-Explorer/model/CNN_face_regression.h5")
        print("FaceFeatureReader.py....self.model...")
        self.face = None
        self.used_face = False
        self.res = []
        self.graph = graph
        self.first_layer = self.build_layer('conv2d_1')
        self.second_layer = self.build_layer('conv2d_2')
        self.third_layer = self.build_layer('conv2d_3')
        self.delete_queue = queue.Queue()
    # __init__(self, graph)

    # build_layer(self, layer_name)
    def build_layer(self, layer_name):
        '''
        build layer
        Arguments:
            layer_name: accept 3 parameter: 'conv2d_1', 'conv2d_2', 'conv2d_3',
                represent the first, the second and the last convolutional
                 layers respectively.
        '''
        with self.graph.as_default():
            layer = self.model.get_layer(layer_name).output
            # layer = L.Deconv2D(filters=32, kernel_size=(3, 3), padding = 'same')(layer)
            layer = L.Deconvolution2D(filters=32, kernel_size=(3, 3), padding='same')(layer)
            conv_layer_output = keras.models.Model(inputs=self.model.input, outputs=layer)
        return conv_layer_output
    # build_layer(self, layer_name)

    # delete_files(self)
    def delete_files(self):
        '''
        delete files for releasing the resourse.
        '''
        while self.delete_queue.qsize() > 540:
            file = self.delete_queue.get()
            if (os.path.exists(file)):
                os.remove(file)
    # delete_files(self)

    # revert_img(self, img)
    def revert_img(self, img):
        '''
        give more weight to the image pixel since they
        are normalized into -1~1
        '''
        img = (img)*255
        return img
    # revert_img(self, img)

    # set_face(self, face)
    def set_face(self, face):
        '''
        Arguments:
            
            faces: the faces to process.
        
        '''
        self.face = face
        self.used_face = False
    # set_face(self, face)

    # format_face(self, face)
    def format_face(self, face):
        return face.reshape(1, 48, 48, 1)
    # format_face(self, face)

    # read_layer(self, conv_layer_output, layer_name)
    def read_layer(self, conv_layer_output, layer_name):
        '''
        Arguments:
            conv_layer_output:
            
            layer_name: accept 3 parameter: 'conv2d_1', 'conv2d_2', 'conv2d_3',
                represent the first, the second and the last convolutional
                 layers respectively.
            faces: the faces to process.
            
        '''
        
        if self.face is None:
            return []
        
        with self.graph.as_default():
            imgs = conv_layer_output.predict(self.format_face(self.face))[0]
            
        res_list = []
        for i in range(30):
            img = imgs[:, :, i]
            img = self.revert_img(img)
            path = 'static/cache_image/%s' % layer_name+''.join(random.sample(string.ascii_letters + string.digits, 12)) + '.png'
            cv2.imwrite(path, img)
            res_list.append(path)
        return res_list
    # read_layer(self, conv_layer_output, layer_name
    
    def read_feature_map(self):
        '''
        read feature map
        Returns:
            a list contains file name saving feature map
        '''
        if self.used_face:
            return self.res
        self.used_face = True
        print(self.res)
        
        for file_name in self.res:
            self.delete_queue.put(file_name)
       
        if self.delete_queue.qsize() > 540:
            self.delete_files()     
        self.res = []
        self.res = self.read_layer(self.first_layer, 'conv2d_1_')\
                   + self.read_layer(self.second_layer, 'conv2d_2_') + self.read_layer(self.third_layer, 'conv2d_3_')
       
        return self.res
# class FaceFeatureReader(object)

# cascPath = "D:/eclipse/PythonWorkspaces/Eumpy/model/haarcascade_frontalface_alt.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)

# def detect_face(gray):
#         '''
#         find faces from a gray image.
#         Arguments:
#             gray: a gray image
#         Returns:
#             (x, y, w, h)
#             x, y: the left-up points of the face
#             w, h: the width and height of the face
#         '''
#         faces = faceCascade.detectMultiScale(
#             gray,
#             scaleFactor = 1.1,
#             minNeighbors = 5,
#             minSize=(32, 32)
#         )
#         if len(faces) > 0:
#             (x, y, w, h) = faces[0]
#         else:
#             (x, y, w, h) = (0, 0, 0, 0)
#             
#         return (x, y, w, h)      
#             
# if __name__ == '__main__':
#     obj = FaceFeatureReader()  
#     
#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         (x, y, w, h) = detect_face(gray)
#         
#         if (w != 0):   
#             face = gray[y:y+h, x:x+w]
#             face = cv2.resize(face, (48, 48))
#             obj.set_face(face)
#         frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness = 2)
#         cv2.imshow('11', frame)
#         cv2.waitKey(10)
#         res = obj.read_feature_map()
#         print (len(res))
#         print ('|'.join(res))
