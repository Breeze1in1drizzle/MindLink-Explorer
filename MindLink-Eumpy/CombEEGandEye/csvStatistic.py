import pandas as pd
import glob
import os


def statistic_cal_FLF():
    data_path = 'C:/Users/ps/Desktop/experiments/leave-one-subject-out_validation/FLF-mahnob/*'
    dirPath = glob.iglob(data_path)  # 有了'/*'才可以遍历*里面的东西，没有就只能在文件夹外面
    # path = dataset_path.replace('/*', '')

    statistics = pd.DataFrame()     # 用dict构造dataframe

    num_of_folder = 0
    for big_file in dirPath:
        num_of_folder += 1
        print("num_of_folder: ", num_of_folder)
        files = os.listdir(big_file)
        # print("big_file: ", big_file)
        # print("files: ", files)
        for file in files:
            subject_id = int(file[8:0])
            if file.endswith("statistic.csv"):
                statistic = pd.DataFrame(pd.read_csv())
                #


if __name__ == "__main__":
    statistic_cal_FLF()