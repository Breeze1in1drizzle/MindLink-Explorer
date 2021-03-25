'''
Python线程机制 Thread
target参数如何使用？
如何实现多线程？ thread.join()函数？

'''

import threading
import time


def run():
    time.sleep(2)
    print('当前线程的名字是： ', threading.current_thread().name)
    time.sleep(2)


if __name__ == '__main__':

    start_time = time.time()

    print('这是主线程：', threading.current_thread().name)
    thread_list = []

    # 这里是创建Thread 实例，传递给他一个函数
    for i in range(5):
        t = threading.Thread(target=run)
        thread_list.append(t)

    for t in thread_list:
        t.start()

    print('主线程结束！', threading.current_thread().name)
    print('一共用时：', time.time()-start_time)
