'''
改文件用于统计文件夹个数-->以求得各个被试的数据量
'''
import sys
sys.path.append('../../')
import glob
import os
import numpy as np


def leave_one_subject_out_validation():
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/'
    # model = LstmModelRegression()

    dirPath = glob.iglob(dataset_path)

    num_of_folder = 0
    num_of_fif = 0
    num_of_success_fif = 0

    # leave one subject out validation
    for leave_one_subject_id in range(0, 39):  # 0->39-1==38   # 遍历到的号码就用作
        print("leave_one_subject_id: ", leave_one_subject_id)
        if leave_one_subject_id == 32:
            print("yes")
            continue
        print("dirPath", dirPath)
        for big_file in dirPath:
            print("123")
            print("big_file:\n", big_file)
            num_of_folder += 1
            print("num_of_folder: ", num_of_folder)
            files = os.listdir(big_file)
            for file in files:
                print("file: ", file)
                subject_id = int((int(file)) / 100)
                print("subject_id: ", subject_id)
                if subject_id == leave_one_subject_id:
                    # model.add_test_data()     # 验证数据集添加
                    print("model.add_test_data()")
                else:
                    # model.add_training_data()    # 训练数据集添加
                    print("model.add_training_data()")
        # model.train()
        print("model.train()")
        # model.save()
        print("model.save()")
        # statistic saving......    # 保存本次模型训练的准确率、RMSE等数据
        print("statistic saving.....")
        # model.evaluation()
        print("model.evaluation()")
        print("\n\n\n\n\n")


def calc_subfolder():
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/'
    print("len of dataset_path: ", len(dataset_path))
    dirPath = glob.iglob(dataset_path)
    print("dirPath:\n", dirPath)

    num_of_folder_plus_one = 0      # 最开始遍历大文件夹的时候，大文件夹也算入了进去，所以计算出来是所有子文件夹数量再加一
    num_of_fif = 0
    num_of_success_fif = 0

    count = np.zeros(39)    # 文件夹id: 0->38       # 每个被试i有多少个文件夹

    for big_file in dirPath:
        print("big_file:\n", big_file)
        num_of_folder_plus_one += 1
        print("num_of_folder_plus_one: ", num_of_folder_plus_one)
        files = os.listdir(big_file)
        print("files:\n", files)
        for file in files:
            # if num_of_folder_plus_one == 0:     # 这个时候在遍历大文件夹
            #     break
            print("file: ", file)
            subject_id = int((int(file)) / 100)
            print("subject_id: ", subject_id)
            count[subject_id] += 1      # 对应的subject_id文件夹数目+1
            # if subject_id
    print("num_of_folder: ", num_of_folder_plus_one - 1)
    print("num_of_fif: ", num_of_fif)
    print("num_of_success_fif", num_of_success_fif)
    print("count:\n", count)
    '''
    [20. 20. 17.  4. 16. 20. 20.  9. 11. 20. 14. 14.  6. 20. 20. 19.  5. 16.
    16. 16.  9. 11. 20. 20. 14.  6. 20. 20. 19.  5. 16. 20.  0.  9. 11. 20.
    20. 14.  6.]
    '''
    print(count.sum())  # 563个文件夹      # 第一维是轴0
    sum = 0
    for i in range(0, 30):
        sum += count[i]
        print(i)
    print(sum)


def loadXML():
    import xml.dom.minidom
    dom = xml.dom.minidom.parse('xx.xml')


if __name__ == "__main__":
    # calc_subfolder()
    # leave_one_subject_out_validation()
    loadXML()
