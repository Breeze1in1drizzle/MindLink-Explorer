'''
用这个文件可以批处理(*.npy)文件，目前功能单一
主要步骤：
step 1: 查看(*.npy)文件的size
step 2: 把(*.npy)文件中的矩阵转换成一张张图片（.jpg）
step 3: 新建一个文件夹，把一个(*.npy)文件中提取的图片放进去
step 4: 将一张张图片，按照(*.npy)原本的格式，再存入(*.npy)文件中（注意这里使用另一个版本的Numpy将图片转为(*.npy)，但是代码基本相同）
step 5: 将上述的方法写成函数，且能够针对文件夹进行使用
'''


import numpy as np      # version: 1.16.5
from PIL import Image
import os
import glob
from pathlib import Path


def npy2jpg(file_folder=None):
    '''
    首先知道了imgs是Python中的三维数组，size: (198, 48, 48)
    :param fil_floder: (*.npy)文件的文件夹路径
    :return:
    '''
    # 在file_floder这个文件夹中创建一个新文件夹

    # 将一个faces.npy文件里面的图片转化为(.jpg)格式
    file_path = file_folder + 'faces.npy'
    imgs = np.load(file_path)
    print(imgs.dtype)
    img0 = imgs[0]
    im = Image.fromarray(img0)
    im.save(file_folder + '1.jpg')
    img1 = Image.open(file_folder + '1.jpg')
    img_all = []
    img_all.append(img1)
    img_all.append(img1)
    print("success")
    img_arr = np.array(img_all)
    print(img_arr.dtype)


def npy2txt(file_folder=None, save_path=None):
    file_path = file_folder + 'faces.npy'
    imgs = np.load(file_path)       # 这里改了Numpy的底层实现代码，allow_pickle=False改为allow_pickle=True
    for i in range(len(imgs)):  # 同一个被试subject的同一个试次trail的所有的人脸图像
        img = imgs[i]
        file_name = 'faces' + str(i+1) + '.txt'
        np.savetxt(save_path + file_name, img, fmt='%d', delimiter=',')
    # img1 = np.loadtxt(file_folder + 'faces1.txt', delimiter=',')
    # img2 = img1.astype(np.uint8)


def npy2txt_in_batch(subject_file_folder=''):
    dirPath = glob.iglob(subject_file_folder)
    path = subject_file_folder.replace('/*', '')
    i = 0
    for big_file in dirPath:
        i += 1
        print("i: ", i)
        # files = os.listdir(big_file)
        this_big_file = big_file.replace(path, '')
        this_big_file = this_big_file.replace('\\', '')
        # print("big_file: ", this_big_file)   # 路径
        folder_path = os.path.join(big_file)  # 路径，与big_file一样
        folder_path = folder_path.replace('\\', '/')
        folder_path += '/'
        # print("folder_path: ", folder_path)
        save_path = folder_path + 'faces/'
        # this_path = Path(save_path)
        # print("save_path: ", save_path, "\n\n\n")
        # flag = this_path.is_dir()
        # if flag is 0:     # 不存在则创建文件夹
        os.mkdir(folder_path + 'faces')      # 创建一个新文件夹
        npy2txt(file_folder=folder_path, save_path=save_path)
    print("success!!!\n")


def npy2txt_in_batch_2222222(subject_file_folder=''):
    dirPath = glob.iglob(subject_file_folder)
    path = subject_file_folder.replace('/*', '')
    # dirPath = os.listdir(path)
    dirList = []
    for big_file in dirPath:

        # files = os.listdir(big_file)
        this_big_file = big_file.replace(path, '')
        this_big_file = this_big_file.replace('\\', '')
        # print("big_file: ", this_big_file)   # 路径
        folder_path = os.path.join(big_file)  # 路径，与big_file一样
        folder_path = folder_path.replace('\\', '/')
        folder_path += '/'
        # print("folder_path: ", folder_path)
        save_path = folder_path + 'faces/'
        # this_path = Path(save_path)
        # print("save_path: ", save_path, "\n\n\n")
        # flag = this_path.is_dir()
        # if flag is 0:     # 不存在则创建文件夹
        if os.path.isdir(folder_path):
            print("faces!!!")
            # os.mkdir(folder_path + 'faces')      # 创建一个新文件夹
            npy2txt(file_folder=folder_path, save_path=save_path)
            # npy2txt(file_folder=folder_path, save_path=save_path)
        elif os.path.isfile(folder_path):
            print("pass")
        else:
            print("folder_path: ", folder_path)
    print("success!!!\n")


def txt2npy_in_batch(subject_file_folder=''):
    return 0


if __name__ == "__main__":
    import configuration
    dataset_file_folder = configuration.DATASET_PATH + 'ONLINE/'
    print(np.__version__)
    '''
    for i in range(12, 21):
        print("subject", i, ": ")
        subject_file_folder = dataset_file_folder + str(i) + '/*'
        # npy2txt_in_batch(subject_file_folder=subject_file_folder)
        npy2txt_in_batch_2222222(subject_file_folder=subject_file_folder)
    '''
    # subject_file_folder
    # print(subject_file_folder)
    # npy2txt_in_batch(subject_file_folder=subject_file_folder)
