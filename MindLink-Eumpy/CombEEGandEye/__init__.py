'''

'''


import glob
import os


def leave_one_subject_out_validation():
    dataset_path = 'C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*'
    print('model = LstmModelRegression()')

    dirPath = glob.iglob(dataset_path)  # 有了'/*'才可以遍历*里面的东西，没有就只能在文件夹外面
    # path = dataset_path.replace('/*', '')

    num_of_folder = 0
    # num_of_fif = 0
    # num_of_success_fif = 0
    for i in range(0, 39):
        # i 作为leave-one-subject-out validation的测试集的那个被试的id
        print("i: ", i)
        if i == 32:
            continue
        for big_file in dirPath:
            num_of_folder += 1
            print("num_of_folder: ", num_of_folder)
            files = os.listdir(big_file)
            print("big_file: ", big_file)
            print("files: ", files)
            for file in files:
                print("file: ", file)
                if file.endswith("EEG.npy"):
                    # print("big_file: ", big_file)
                    folder_num = big_file[len(dataset_path) - 1:]  # 获取相关数字
                    subject_id = int(int(folder_num) / 100)
                    if subject_id == i:
                        print("subject ", subject_id, ", add_test_data")
                        # model.add_test_data(big_file)
                    else:
                        print("subject ", subject_id, ", add_train_data")
                        # model.add_train_data(big_file)
        print("train.....")
        # model.train()
        print("evaluae.....")
        # model.evalute()


if __name__ == "__main__":
    # leave_one_subject_out_validation()
    import pandas as pd
    dict_statistic = {
        "acc_valence": "acc_valence",
        "f1_valence": "f1_valence",
        "acc_arousal": "acc_arousal",
        "f1_arousal": "f1_arousal",
        "train_valence_sample_proportion": "train_valence_sample_proportion",
        "train_arousal_sample_proportion": "train_arousal_sample_proportion",
        "test_valence_sample_proportion": "test_valence_sample_proportion",
        "test_arousal_sample_proportion": "test_arousal_sample_proportion",
        "train_sample_shape": "train_sample_shape",
        "test_sample_shape": "test_sample_shape"
    }
    data_df = pd.DataFrame(data=dict_statistic, index=[0])
    print(data_df)
    data_df.to_csv('test.csv', index=False)
