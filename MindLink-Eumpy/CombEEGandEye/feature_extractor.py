'''

'''
import numpy as np
import pandas as pd
from EEG_feature_extract import *
from Eye_feature_extract import *
import glob


def extract_features_from_both_Eye_and_EEG():
    '''
    :return:
    '''
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*'
    dirPath = glob.iglob(dataset_path)  # 有了'/*'才可以遍历*里面的东西，没有就只能在文件夹外面

    # 计算已经遍历了多少个文件夹
    num_of_folder = 0

    # 特征计算成功与否
    EEG_result = True
    Eye_result = True

    # 测试用，出现(*.fif)文件或者(*.csv)文件就自增。当test>=2，停止循环。这一目的在于仅测试第一次遍历到的文件夹。
    # test = 0

    # 记录出现错误的文件夹
    EEG_error_list = []
    Eye_error_list = []

    # 特征文本(*.npy)文件
    EEG_npy = None
    Eye_npy = None
    Fusion_npy = None

    for big_file in dirPath:

        num_of_folder += 1
        print("\n\n\n\n\nnum_of_folder: ", num_of_folder)

        files = os.listdir(big_file)
        print("big_file: ", big_file)
        print("files: ", files)

        for file in files:

            if file.endswith(".fif"):
                file_path = os.path.join(big_file, file)  # 路径+文件名
                file_path = file_path.replace('\\', '/')
                save_path = big_file.replace('\\', '/') + '/'  # 只有路径，没有文件名字
                print("file_path:\n", file_path)
                print("big_file:\n", big_file)
                print("save_path:\n", save_path)
                EEG_result = raw_to_numpy_one_trial(
                    EEG_path=file_path, save_path=big_file.replace('\\', '/'))
                if EEG_result is not True:
                    print("EEG error error error !!! !!!\n", result)
                    error_list.append(EEG_result)
                    continue
                EEG_npy = np.load(save_path+'EEG.npy')

                # test += 1

            elif file.endswith(".csv"):
                file_path = os.path.join(big_file, file)  # 路径+文件名
                file_path = file_path.replace('\\', '/')
                save_path = big_file.replace('\\', '/') + '/'     # 只有路径，没有文件名字
                print("file_path:\n", file_path)
                print("big_file:\n", big_file)
                print("save_path:\n", save_path)
                # 计算眼动特征
                Eye_result = Eye_extract_average_psd_from_a_trial(
                    file_path=file_path, save_path=save_path,
                    average_second=1, overlap=0.5)
                if Eye_result is not True:
                    print("Eye error error error !!! !!!\n", result)
                    Eye_error_list.append(Eye_result)
                    continue
                Eye_npy = np.load(save_path+'Eye.npy')

                # test += 1

        if (EEG_result and Eye_result) == True:
            min_len = min(len(Eye_npy), len(EEG_npy))
            print("min_len: ", min_len)
            new_Eye_npy = Eye_npy[0:min_len]
            new_EEG_npy = EEG_npy[0:min_len]
            print(new_EEG_npy.shape)
            print(new_Eye_npy.shape)
            Fusion_npy = np.concatenate((new_EEG_npy, new_Eye_npy), axis=1)
            np.save(save_path+'Fusion.npy', Fusion_npy)
            print("Fusion.save......\n\n\n\n\n")
        # if test >= 2:
        #     break   # test break
    # print(test)
    print("error EEG:\n", EEG_error_list)
    print("error Eye\n", Eye_error_list)


# def FLF_for_one_trail(EEG_npy, Eye_npy):
#     return np.array()


def test():
    '''
    Feature-Level Fusion
    特征层融合，矩阵融合
    傻瓜式融合方法（naive fusion method）
    （1）首先对比两个2D矩阵的size
    （2）将
    :return:
    '''
    x = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 0],
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 0]
    ])
    y = np.array([
        [1, 2, 3, 4, 5],          # 可以当作每个小的[]中间是一个竖着的向量
        [6, 7, 8, 4, 5]
    ])
    t = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    # print(np.dot(t, y))
    print(len(t))       # 竖着的   我们要左右拼接，所以计算行数，行数大的，就要减去过多的行，这里需要矩阵剪裁
    print(len(t[0]))    # 横着的
    print(t.shape)      # (4, 2)    (竖着，横着)-->(行，列)-->(*, 85)-->(*, 2)
    z = np.concatenate((x, y), axis=0)      # x==0就是两个矩阵上下摆放拼接，x==1就是左右拼接
    print(z)

    # 矩阵剪裁
    long_matrix = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    short_matrix = long_matrix[0:4]     # [x:y]，生成的数组的index为  x<=index<=y-1
    print("short_matrix:\n", short_matrix)


if __name__ == "__main__":
    extract_features_from_both_Eye_and_EEG()
    # FLF_for_one_trail()
    # test()
