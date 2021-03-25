'''
@date: 2021.03.10
@author: Ruixin Lee
用(*.fif)格式的样本文件来进行一系列的处理（例如：提取特征值并保存为(*.npy)文件格式）
从而计算特征矩阵的一些特点（例如，尺寸大小等）
为后续特征层融合做准备
'''


def myFunction():
    import numpy as np
    x = np.load('../EEG_1.npy')
    print(x.shape)


if __name__ == "__main__":
    # folder_path = '../../data/mahnob_example/EEG_files/'
    # original_filename = 'origin.fif'
    # label_filename = 'label.csv'
    myFunction()
