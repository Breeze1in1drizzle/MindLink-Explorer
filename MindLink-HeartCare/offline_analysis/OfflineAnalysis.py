# -*- coding: utf-8 -*-
'''
欧拉放大视频，然后计算视频钟的心率
'''
import EVM_V1 as EVM
import heartRateObserver as HRO


def offline_analysis_for_one_video(vidoe_path='', save_path=''):
    # 首先读取视频文件(*.mp4)
    evm = EVM.EVM_tools()
    # 然后进行欧拉放大，保存放大后的视频文件
    evm.magnify_color("baby.mp4", 0.4, 3)
    ###########################################
    #####上述欧拉放大步骤需要调整一下保存路径########
    ###########################################
    # 再读取新的文件，计算每一分钟图片的心率
    hro = HRO.heartRateObserver()       # 这个类没有文件的IO和处理功能
    hro.run()       # 这里需要把在线分析代码改为离线分析代码，同时还要增加保存数据的函数


if __name__ == "__main__":
    video_path = 'baby.mp4'
    offline_analysis_for_one_video(vidoe_path=video_path)
