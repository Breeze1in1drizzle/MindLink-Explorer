'''
@author: Ruixin Lee
@Date: 2021.03.16
这个文件用于处理(*.mat)文件
'''
from scipy.io import loadmat
import scipy
import pandas as pd


def function_v1(filepath=''):
    eye_raw_mat_file = loadmat(file_path)
    print("eye_raw_mat_file:\n", type(eye_raw_mat_file))
    '''
    eye_raw_mat_file: <class 'dict'>    # 字典类
    http://c.biancheng.net/view/2212.html     # Python对字典类数据的操作
    '''
    # print(eye_raw_mat_file)
    # eye_raw_pandas_file = pd.DataFrame(data=eye_raw_mat_file, index=[0])
    # ValueError: Data must be 1-dimensional    # 数据是多维度的，不是1维，所以无法转换为pandas的DataFrame？？？
    # new_filepath = filepath.replace('.mat', '.csv')
    # eye_raw_pandas_file.to_csv(new_filepath, index=False)


if __name__ == "__main__":
    file_path = "C:/Users/ps/Desktop/test_eye_raw.mat"
    function_v1(filepath=file_path)
