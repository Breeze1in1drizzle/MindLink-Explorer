'''
@author: Ruixin Lee
@date: 2021/03/08
This file provides methods to preprocess eye movement signals form MAHNOB-HCI-TAGGING

该文件用于处理来自MAHNOB-HCI-TAGGING数据集的眼动信号。
首先将(*.tsv)文件转换为(*.csv)文件【这一步需要将实验不需要的描述性数据全部删除】
然后将(*.csv)文件中需要的列提出来成为新的子(*.csv)文件【这一步目的在于获取用于提取特征值的关键数据，难点在于选取哪些数据？有些数据无法对齐？有些数据无法处理？是否有脏数据？】
基于新的子(*.csv)文件，提取特征值，并存储到(*.npy)文件当中【这一步难点是如何进行视频分析（手写基础代码）？特征数据能否对齐？能够构造特征矩阵？这些特征数据能否与EEG的特征结合起来构造混合特征矩阵？】

主要考虑两种特征提取方法：
（1）用(*.csv)文件手写提取方法，计算得到特征值，并将特征值存储到(*.npy)文件当中
（2）构造(*.fif)文件以及MNE数据结构，利用MNE的方法来处理数据
'''


import pandas as pd
import numpy as np

basic_path = '../../data/mahnob_example/2/'
path_1 = '../../data/mahnob_example/2/P1-Rec1-Guide-Cut.tsv'


def one_tsv_2_csv(file_path=None, file_name=None):
    '''
    Description: A (*.tsv) file is going to be converted to (*.csv) one after this function works.
    这里涉及缺失值的处理方法
    原始数据有缺失值，例如 -1、null等等
    (1)整行空白的null，可以drop掉
    (2)如果有-1，则需要进行一定的数据处理

    :param file_path:文件路径
    :param file_name:文件名
    :return:None
    '''
    print(file_path+file_name)
    # 分隔符需要定义，（*.tsv)文件的分隔符是'\t'，而不是默认的' '
    df = pd.DataFrame(pd.read_csv(file_path+file_name, sep='\t', header=17))

    # print('df:')
    # print(df)

    # new_file_name ==>将尾缀的'.tsv'改为'.csv'进行保存
    new_file_name = file_name.replace('.tsv', '.csv')

    ##########################################################
    ##########################################################
    ##########################################################
    # 数据处理，并返回新的df-->new_df
    # new_df = myProcessor(DataFrame=df)    # 完成这个模块后，把这个模块改为函数
    # new_df = df.drop(columns=[
    #     'Event', 'EventKey', 'Data1', 'Data2', 'Descriptor', 'StimuliName', 'StimuliID',
    #     'MediaWidth', 'MediaHeight', 'MediaPosX', 'MediaPosY', 'MappedFixationPointX',
    #     'MappedFixationPointY', 'AoiIds', 'AoiNames', 'WebGroupImage', 'MappedGazeDataPointX',
    #     'MappedGazeDataPointY', 'AudioSampleNumber', 'Unnamed: 43', 'Unnamed: 44'
    # ])
    # print("new_df_df_df")
    new_df = df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']]
    # new_df = new_df.drop(df[df['PupilLeft']=='NaN'].index)

    # 去除有空白值的行
    new_df = new_df.dropna(subset=['PupilLeft'])


    # print(df[df['PupilLeft']==None].index)

    ##########################################################
    ##########################################################
    ##########################################################


    # print('new_df:')
    # print(new_df)

    # 在原路径进行保存
    new_df.to_csv(file_path + new_file_name, index=False)
    # print("end...............")
    return None


def drop_the_front_data(file_path, file_name):
    '''
    这段代码与one_tsv_2_csv函数的代码放在一起会有错误，会有index上的识别错误
    猜测是在为保存的时候，虽然有些行被drop掉了，但是index没有更新，需要保存的时候才进行更新
    于是在进行行操作的时候就会有错误
    :param file_path:
    :param file_name:
    :return:
    '''
    new_df = pd.DataFrame(pd.read_csv(file_path+file_name))
    # 去除前一小段不足1s的数据
    time_start = new_df.iat[0, 1]
    # print("time_start:\n", type(time_start), "\n", time_start)
    start_index = 0
    for i in range(len(new_df)):
        time = new_df.iat[i, 1]
        # print("time: ", time)
        str_a = time_start[:8]
        # print("a: ", str_a)
        str_b = time[:8]
        # print("b: ", str_b)
        if str_a == str_b:
            continue
        else:
            start_index = i
            break
    # print("start_index: ", start_index)
    new_df = new_df.drop(labels=range(0, start_index), axis=0)
    new_df.to_csv(file_path+file_name, index=False)


def myProcessor(DataFrame=None):
    '''
    :param DataFrame:
    :return:
    '''
    new_df = DataFrame
    '''
       中间这一堆是一系列的数据处理，把df丢入函数中，然后返回一个新的new_df
    '''
    return new_df


def read_and_show_csv(file_path=None, file_name=None):
    '''
    测试代码
    :param file_path:
    :param file_name:
    :return:
    '''
    csv_file = pd.read_csv(file_path+file_name)
    df_csv_file = pd.DataFrame(csv_file)
    print(df_csv_file)


def read_and_show_npy(path=None):
    matrix = np.load(path)
    print('matrix:\n', matrix)
    # txt_matrix = matrix.resize(1, -1)
    # print('txt_matrix:\n', txt_matrix)
    np.savetxt(r'D:\myworkspace\mypyworkspace\MindLink-Eumpy\data\EEG.txt', matrix, delimiter=',')


def pupil_diameter_calculation(file_path, file_name):
    '''
    计算瞳孔直径
    :param file_path:
    :param file_name:
    :return:
    '''
    df = pd.DataFrame(pd.read_csv(file_path+file_name))
    # print(df.iat[0, 3])
    # print(df.iat[0, 2])
    # print(df.iat[1, 3])
    # print(df.iat[1, 2])
    pupil_diameter = []
    for i in range(len(df)):
        pupil_diameter.append(abs(df.iat[i, 3] - df.iat[i, 2]))     # -1的时候就变成0（-1往往成对出现）
    df['PupilDiameter'] = pupil_diameter
    df.to_csv(file_path+file_name, index=False)


if __name__ == "__main__":
    # print("Hello, MAHNOB-HCI-TAGGING!")
    # 单独处理一个文件
    ####################
    ####################
    one_tsv_2_csv(file_path=basic_path, file_name='P1-Rec1-All-Data-New_Section_2.tsv')     # 格式转换+清楚了空白值
    ####################
    ####################
    # read_and_show_csv(file_path=basic_path, file_name='P1-Rec1-All-Data-New_Section_2.csv')
    # read_and_show_npy(path='../../data/EEG.npy')

    ####################
    ####################
    # 去除了最开头不足1s的数据，因为要满足119s的时长，这多出的不足1s的数据无法纳入时间范围，而且这些数据不够后面的稳定
    drop_the_front_data(file_path=basic_path, file_name='P1-Rec1-All-Data-New_Section_2.csv')
    ####################
    ####################
    pupil_diameter_calculation(file_path=basic_path, file_name='P1-Rec1-All-Data-New_Section_2.csv')
    print("end")
