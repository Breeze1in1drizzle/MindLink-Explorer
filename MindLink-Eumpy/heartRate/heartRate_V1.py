#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# https://download.csdn.net/download/itnerd/12923042
# https://blog.csdn.net/itnerd/article/details/109078291

import cv2
import traceback
import threading
from queue import Queue
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

import time
import numpy as np


def hp(y, lamb=10):
    def D_matrix(N):
        D = np.zeros((N-1,N))
        D[:,1:] = np.eye(N-1)
        D[:,:-1] -= np.eye(N-1)
        """D1
        [[-1.  1.  0. ...  0.  0.  0.]
         [ 0. -1.  1. ...  0.  0.  0.]
         [ 0.  0. -1. ...  0.  0.  0.]
         ...
         [ 0.  0.  0. ...  1.  0.  0.]
         [ 0.  0.  0. ... -1.  1.  0.]
         [ 0.  0.  0. ...  0. -1.  1.]]
        """
        return D
    N = len(y)
    D1 = D_matrix(N)
    D2 = D_matrix(N-1)
    D = D2 @ D1
    g = np.linalg.inv((np.eye(N)+lamb*D.T@D))@ y
    return g


class Producer(threading.Thread):
    def __init__(self, data_queue, *args, **kwargs):        # *args 和 **kwargs 代表什么？如何使用？
        super(Producer, self).__init__(*args, **kwargs)
        self.data_queue = data_queue    # 队列，在下方 main 函数中可以看到实例
 
    def run(self):
        capture = cv2.VideoCapture(0)  # 0是代表摄像头编号，只有一个的话默认为0
        capture.set(cv2.CAP_PROP_FPS, 10)
        try:
            t0 = time.time()
            while (True):
                ref, frame = capture.read()
                frame = frame[:,::-1,:].copy()
                H, W, _ = frame.shape
                w, h = 40, 40
                x, y = W//2 -w//2, H//4-h//2
                area = frame[y:y + h, x:x + w, :]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                frame[:h,:w] = area
                
                t = time.time()-t0
                cv2.putText(frame, 't={:.3f}'.format(t), (10, H-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)
                cv2.imshow("face", frame)

                B = np.average(area[:,:,0])
                G = np.average(area[:,:,1])
                R = np.average(area[:,:,2])
                if self.data_queue.full():
                    self.data_queue.queue.popleft()
                self.data_queue.put((t,B,G,R))

                c = cv2.waitKey(10) & 0xff  # 等待30ms显示图像，若过程中按“Esc”退出
                if c == 27:
                    capture.release()
                    break
        except:
            traceback.print_exc()
        finally:
            capture.release()
            cv2.destroyAllWindows()
            if self.data_queue.full():
                self.data_queue.get()
            self.data_queue.put('Bye')
        print('Producer quit')


class Consumer(threading.Thread):
   
    def __init__(self, data_queue, *args, **kwargs):
        super(Consumer, self).__init__(*args, **kwargs)
        self.data_queue = data_queue
 
    def run(self):
        time.sleep(1)

        fig, axes = plt.subplots(3, 3)
        axes[0, 0].set_title('原始信号')
        axes[0, 1].set_title('HP滤波残差')
        axes[0, 2].set_title('FFT频谱')
        axes[0, 0].set_ylabel('Blue')
        axes[1, 0].set_ylabel('Green')
        axes[2, 0].set_ylabel('Red')
        axes[2, 0].set_xlabel('Time(s)')
        axes[2, 1].set_xlabel('Time(s)')
        axes[2, 2].set_xlabel('Frequency(Hz)')
        start = None
        lines = [None, None, None]
        glines = [None, None, None]
        rlines = [None, None, None]
        flines = [None, None, None]
        BGR = [None, None, None]
        g = [None, None, None]
        r = [None, None, None]
        f = [None, None, None]
        num_fft = 256

        while True:
            # time.sleep(0.2)
            if self.data_queue.qsize() > 2:
                if self.data_queue.queue[-1] == 'Bye':
                    break
                ts, BGR[0], BGR[1], BGR[2] = zip(*self.data_queue.queue)
                t = ts[-1] if len(ts) > 0 else 0

                for i in range(3):
                    g[i] = hp(BGR[i], 1000)
                    r[i] = BGR[i] - g[i]

                # FFT
                for i in range(3):
                    rr = r[i][-num_fft:]
                    f[i] = np.fft.fft(rr, num_fft)
                    f[i] = np.abs(f[i])[:num_fft//2]
                fs =len(rr)/ (ts[-1] - ts[-len(rr)])


                if start is None:
                    start = 1
                    lines[0] = axes[0,0].plot(ts, BGR[0], '-b')[0]
                    lines[1] = axes[1,0].plot(ts, BGR[1], '-g')[0]
                    lines[2] = axes[2,0].plot(ts, BGR[2], '-r')[0]
                    glines[0] = axes[0,0].plot(ts, g[0], '-k')[0]
                    glines[1] = axes[1,0].plot(ts, g[1], '-k')[0]
                    glines[2] = axes[2,0].plot(ts, g[2], '-k')[0]
                    rlines[0] = axes[0, 1].plot(ts, r[0], '-b')[0]
                    rlines[1] = axes[1, 1].plot(ts, r[1], '-g')[0]
                    rlines[2] = axes[2, 1].plot(ts, r[2], '-r')[0]
                    flines[0] = axes[0, 2].plot(np.arange(num_fft//2)*fs/num_fft, f[0], '-b', marker='*')[0]
                    flines[1] = axes[1, 2].plot(np.arange(num_fft//2)*fs/num_fft, f[1], '-g', marker='*')[0]
                    flines[2] = axes[2, 2].plot(np.arange(num_fft//2)*fs/num_fft, f[2], '-r', marker='*')[0]

                for i in range(3):
                    lines[i].set_xdata(ts)
                    lines[i].set_ydata(BGR[i])
                    glines[i].set_xdata(ts)
                    glines[i].set_ydata(g[i])
                    rlines[i].set_xdata(ts)
                    rlines[i].set_ydata(r[i])
                    flines[i].set_xdata(np.arange(num_fft//2)*fs/num_fft)
                    flines[i].set_ydata(f[i])

                for i in range(3):
                    axes[i, 0].set_xlim([t - 10, t + 1])
                    axes[i, 0].set_ylim([np.min(BGR[i][-num_fft:]), np.max(BGR[i][-num_fft:])])
                    axes[i, 1].set_xlim([t - 10, t + 1])
                    axes[i, 1].set_ylim([np.min(r[i][-num_fft:]), np.max(r[i][-num_fft:])])
                    axes[i, 2].set_xlim([0, fs//2])
                    axes[i, 2].set_ylim([np.min(f[i]), np.max(f[i])])

                plt.pause(0.1)
        print('Consumer quit')


# N = 300
# data_queue = Queue(N)
# print('N=300')
#
# p = Producer(data_queue)
# p.start()
# c = Consumer(data_queue)
# c.start()
# print('C.start()')
#
# p.join()
# print('p.join()')
# c.join()
# print('c.join()')
# print('Bye')

if __name__ == "__main__":
    N = 300
    data_queue = Queue(N)
    print('N=300')

    p = Producer(data_queue)
    p.start()
    c = Consumer(data_queue)
    c.start()
    print('C.start()')

    p.join()
    print('p.join()')
    c.join()
    print('c.join()')
    print('Bye')



# In[3]:


# In[ ]:




