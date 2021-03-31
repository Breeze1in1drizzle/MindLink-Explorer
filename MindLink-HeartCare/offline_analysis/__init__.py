# -*- coding: utf-8 -*-
import cv2
import time


def test_record_video(filepath='', second=60):
    '''
    :param filepath:
    :param second: 以s为单位的时间，默认60s
    :return:
    '''
    cap = cv2.VideoCapture(0)
    width = 640
    ret = cap.set(3, width)
    height = 480
    ret = cap.set(4, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out_12138.avi', fourcc, 20.0, (width, height))
    start_time = time.time()
    # end_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            frame = cv2.resize(frame, (640, 480))
            out.write(frame)
            cv2.imshow('frame', frame)
        else:
            break
        key = cv2.waitKey(1)
        end_time = time.time()

        if (key == ord('q')) or (start_time - end_time >= second):
            # 乘以1000后才是以ms毫秒为单位的时间，1s==1000ms，1min==60s==60*1000ms==60000ms
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def test1():
    filename = 'test.mp4'
    test_record_video(filepath=filename)


def test2():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    print 'size of frame: ', len(frame), ', ', len(frame[0])
    print type(frame)


def test3():
    T1 = time.time()
    # ______假设下面是程序部分______
    # for i in range(100 * 100):
    #     pass
    time.sleep(5)
    T2 = time.time()
    gap_time = (T2 - T1)
    print('程序运行时间:%s秒' % gap_time)
    # 程序运行时间:0.0毫秒


if __name__ == "__main__":
    test3()
