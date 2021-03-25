'''
# https://blog.csdn.net/tinyzhao/article/details/52681250
根据以上链接进行改写的代码
出自 paper《Eulerian Video Magnification for Revealing Subtle Changes in the World》
'''


import cv2
import numpy as np
from scipy import fftpack


def load_video(video_filename):
    '''
    load video from file
    :param video_filename:
    :return:
    '''
    cap = cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_tensor = np.zeros((frame_count, height, width, 3), dtype='float')
    x = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            video_tensor[x] = frame
            x += 1
        else:
            break
    return video_tensor, fps


def build_gaussian_pyramid(src, level=3):
    '''
    Build Gaussian Pyramid
    :param src:
    :param level:
    :return:
    '''
    s = src.copy()
    pyramid = [s]
    for i in range(level):
        s = cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid


def temporal_ideal_filter(tensor, low, high, fps, axis=0):
    '''
    apply temporal ideal bandpass filter to gaussian video
    :param tensor:
    :param low:
    :param high:
    :param fps:
    :param axis:
    :return:
    '''
    fft = fftpack.fft(tensor, axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff = fftpack.ifft(fft, axis=axis)
    return np.abs(iff)


def reconstruct_video(amp_video, origin_video, levels=3):
    '''
    reconstruct video from original video and gaussian video
    :param amp_video:
    :param origin_video:
    :param levels:
    :return:
    '''
    final_video = np.zeros(origin_video.shape)
    for i in range(0, amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img = cv2.pyrUp(img)
        img = img + origin_video[i]
        final_video[i] = img
    return final_video


def save_video(video_tensor):
    '''
    save video to files
    :param video_tensor:
    :return:
    '''
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    [height, width] = video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("out.avi", fourcc, 30, (width, height), 1)
    for i in range(0, video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()


if __name__ == '__main__':
    # cv2.pyrDown()
    # cv2.pyrUp()
    pass
