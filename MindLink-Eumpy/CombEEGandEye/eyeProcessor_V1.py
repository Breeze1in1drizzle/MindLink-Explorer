'''
@author: Ruixin Lee
@date: 2021/03/08
This file provides methods to preprocess eye movement signals form MAHNOB-HCI-TAGGING

Version 1: raw data processing
'''


import pandas as pd
import numpy as np
import os


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
    print("one_tsv_2_csv.start......")
    # 分隔符需要定义，（*.tsv)文件的分隔符是'\t'，而不是默认的' '
    # df = pd.DataFrame(pd.read_csv(file_path + file_name, sep='\t', header=23))
    try:
        df = pd.DataFrame(pd.read_csv(file_path+file_name, sep='\t', header=23))
    except:
        print("read_csv error")
        return 0
    # print("df:\n", df)
    # print("df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']]\n",
    #       df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']])
    # new_file_name ==>将尾缀的'.tsv'改为'.csv'进行保存
    new_file_name = file_name.replace('.tsv', '.csv')
    try:
        new_df = df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']]
    except:
        print("new_df error")
        df = pd.DataFrame(pd.read_csv(file_path + file_name, sep='\t', header=17))
        new_df = df[['Timestamp', 'DateTimeStamp', 'PupilLeft', 'PupilRight']]
    # new_df = new_df.drop(df[df['PupilLeft']=='NaN'].index)

    # 去除有空白值的行
    new_df = new_df.dropna(subset=['PupilLeft'])

    # 在原路径进行保存
    new_df.to_csv(file_path + new_file_name, index=False)
    print("one_tsv_2_csv.end......")
    return 1


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
    # print(new_df)
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


def raw_data_preprocess(file_path, file_name):
    '''
    :param file_path:
    :param file_name:
    :return:
    '''
    print("raw_data_preprocess.start......")
    print(file_path+file_name)
    judge = one_tsv_2_csv(file_path=file_path, file_name=file_name)  # 格式转换+清楚了空白值
    if judge == 0:
        return 0        # 如果有问题，直接跳过该文件夹
    # 去除了最开头不足1s的数据，因为要满足119s的时长，这多出的不足1s的数据无法纳入时间范围，而且这些数据不够后面的稳定
    file_name = file_name.replace('.tsv', '.csv')
    print(file_name)
    drop_the_front_data(file_path=file_path, file_name=file_name)
    pupil_diameter_calculation(file_path=file_path, file_name=file_name)
    print("raw_data_preprocess.end......")


def manobProcess():
    # data_path = 'C:/Users/xiaojian/Desktop/dataProcessor'
    # data_path = 'C:/Users/ps/Desktop/dataProcessor/dataTest'  # 43
    data_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions'   # 88
    i = 0
    for root, dirs, files in os.walk(data_path):
        i += 1
        print("i: ", i)
        print("root: ", root)  # 当前目录路径
        print("dirs: ", dirs)  # 当前路径下所有子目录
        print("len of dirs: ", len(dirs))
        print("files: ", files)  # 当前路径下所有非目录子文件
        j = 0
        for file in files:
            j += 1
            print("j: ", j)
            # print("file name: ", file, "\n", type(file))    # <class 'str'>
            if (file.endswith('.tsv') and ('All-Data-New' in file)):    # 每个子文件夹有两个(*.tsv)文件
                # 获取编号，用于拼装文件夹路径
                # data_number = file.rsplit('_', 1)[1].split('.')[0]
                if len(root) >= 88:
                    data_number = root[88:]
                    # real dataset      # 这里要根据路径的字符数量来判断，具体问题要具体修改。如果看见这里报错，请数一数root的字符数
                else:
                    data_number = root[43:]
                    # test dataset      # 这里要根据路径的字符数量来判断，具体问题要具体修改。如果看见这里报错，请数一数root的字符数
                print("a data number: ", data_number)
                file_path = data_path + '/' + data_number + '/'
                print("a file_path: ", file_path)
                print("a file: ", file)
                raw_data_preprocess(file_path=file_path, file_name=file)
                break
            else:
                continue
                # break


def test():
    file_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/1042/' # 1042
    file_name = 'P9-Rec1-All-Data-New_Section_2.tsv'
    # df = pd.DataFrame(pd.read_csv(file_path+file_name, sep='\t', error_bad_lines=False))
    df = pd.DataFrame(pd.read_csv(file_name, sep='\t', header=23))
    # df = pd.DataFrame(pd.read_csv(file_path+file_name, sep='\t'))
    # pdFile = pd.read_csv(file_path+file_name, sep='\t', header=23)
    # df = pd.DataFrame(pd.read_csv(file_path+, sep='\t'))
    print("df:\n", df)
    # print(pdFile)


def num_of_files_calc():
    '''
    统计mahnob-hci数据集里面有问题的文件的数量
    :return:
    '''
    data_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions'
    i = 0
    j = 0
    without_csv_path = list()
    for root, dirs, files in os.walk(data_path):
        i += 1
        print("i: ", i)
        print("root: ", root)  # 当前目录路径
        print("dirs: ", dirs)  # 当前路径下所有子目录
        print("len of dirs: ", len(dirs))
        print("files: ", files)  # 当前路径下所有非目录子文件
        k = 0   # 当前文件夹中(*.csv)文件数量
        for file in files:
            if (file.endswith('.csv') and ('All-Data-New' in file)):
                j += 1
                k += 1
                print("file name: ", file)
            else:
                continue
        if k == 0:  # 无csv文件
            without_csv_path.append(root)
    print("i: ", i)
    print("j: ", j)
    print("without csv path")
    print(without_csv_path)


if __name__ == "__main__":
    # raw_data_preprocess(file_path='../data/mahnob_example/2/', file_name='P1-Rec1-All-Data-New_Section_2.tsv')
    ####################################
    ####################################
    ####################################
    manobProcess()
    ####################################
    ####################################
    ####################################
    # test()
    # str = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/1042'
    # sub_str = str[88:]#88
    # print(sub_str)
    # print(type(sub_str))
    # num_of_files_calc()
