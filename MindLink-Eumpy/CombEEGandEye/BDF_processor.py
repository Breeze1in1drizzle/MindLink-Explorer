import mne
import pandas as pd
import os
import glob
# import configuration


def DEAP_bdf_2_fif():
    # dataset_path = configuration.DATASET_PATH + "28-32/*"
    dataset_path = ''
    # dataset_path = "../dataset3/*"
    # want_meg = False
    # want_eeg = True
    # want_stim = False
    include = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    bad_channels = [
        'Fp1', 'FC1', 'C3', 'CP1', 'CP5',
        'Pz', 'PO3', 'Oz', 'PO4', 'CP6',
        'CP2', 'C4', 'FC2', 'Fp2', 'Fz',
        'Cz', 'EXG1', 'EXG2', 'EXG3', 'EXG4',
        'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1',
        'GSR2', 'Erg1', 'Erg2', 'Resp',
        'Temp', 'P3', 'P4', 'highpass'
    ]
    # 获取多个文件夹的路径，并返回一个可迭代对象
    dirPath = glob.iglob(dataset_path)
    path = dataset_path.replace('/*', '')
    # 将可迭代的对象进行循环获取，赋值给 big_file
    i = 0
    for big_file in dirPath:
        # 获取每个文件夹下的文件名并赋值给 file
        files = os.listdir(big_file)
        this_big_file = big_file.replace(path, '')
        this_big_file = this_big_file.replace('\\', '')
        print("\n\n\n", i, "\n")
        i += 1
        print("big_file: ", this_big_file)   # 路径
        # 将获取的所有文件名进行循环判断
        for file in files:
            # 判断扩展名只为.bdf       ########不包含.origin.txt
            if file.endswith(".bdf"):  # and not file.endswith(".origin.txt"):
                '''
                    进行文件完整路径拼接并读取数据
                    file为文件名
                    os.path.join(big_file, file)为路径
                '''
                # print("file: ", file)       # 文件名
                T_file = file.replace('.bdf', '.fif')
                T_file = T_file.replace('\\', '/')
                print("T_file: ", T_file)
                file_path = os.path.join(big_file, file)  # 路径+文件名
                print(file_path)
                file_path = file_path.replace('\\', '/')
                folder_path = os.path.join(big_file)  # 路径，与big_file一样
                folder_path = folder_path.replace('\\', '/')

                raw_bdf_data = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')
                # raw_bdf_data.to_data_frame()
                raw_bdf_data.info['bads'] += bad_channels
                picks = mne.pick_types(
                    raw_bdf_data.info, meg=False, eeg=True, stim=False,
                    include=include, exclude='bads'
                )
                save_path = folder_path + '/' + T_file
                print("save_path: ", save_path)
                raw_bdf_data.save(save_path, picks=picks, overwrite=True)
                # raw = mne.io.read_raw_fif(save_path, preload=True, verbose='ERROR')
                # raw = raw.to_data_frame()
                # df = pd.DataFrame(raw)
                # T_file = T_file.replace('.fif', '.csv')
                # df.to_csv(folder_path + '/' + T_file, index=False)


# def MAHNOB_HCI_bdf_2_fif():
#     # want_meg = False
#     # want_eeg = True
#     # want_stim = False
#     dataset_path = configuration.DATASET_PATH + "MAHNOB_HCI/*"
#     include = ['AF3', 'F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'AF4', 'F4', 'F8', 'FC6', 'T8', 'P8', 'O2']
#     bad_channels = [
#         'FC1', 'C3', 'CP5', 'CP1', 'PO3', 'Oz', 'Pz',
#         'Fp2', 'Fz', 'FC2', 'Cz', 'C4', 'CP6', 'CP2',
#         'PO4', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5',
#         'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1',
#         'Erg2', 'Resp', 'Temp', 'Status', 'P3', 'P4', 'Fp1'
#     ]
#     # 获取多个文件夹的路径，并返回一个可迭代对象
#     dirPath = glob.iglob(dataset_path)
#     path = dataset_path.replace('/*', '')
#     # 将可迭代的对象进行循环获取，赋值给 big_file
#     for big_file in dirPath:
#         # 获取每个文件夹下的文件名并赋值给 file
#         files = os.listdir(big_file)
#         this_big_file = big_file.replace(path, '')
#         this_big_file = this_big_file.replace('\\', '')
#         print("\n\n\n")
#         print("big_file: ", this_big_file)   # 路径
#         # 将获取的所有文件名进行循环判断
#         for file in files:
#             # 判断扩展名只为.bdf       ########不包含.origin.txt
#             if file.endswith(".bdf"):  # and not file.endswith(".origin.txt"):
#                 '''
#                     进行文件完整路径拼接并读取数据
#                     file为文件名
#                     os.path.join(big_file, file)为路径
#                 '''
#                 # print("file: ", file)       # 文件名
#                 filearray = file.split("_")
#                 T_file = filearray[0]+filearray[1]+"_"+filearray[3]+".fif"
#                 # T_file = file.replace('.bdf', '.fif')
#                 T_file = T_file.replace('\\', '/')
#                 # print("T_file: ", T_file)
#                 file_path = os.path.join(big_file, file)  # 路径+文件名
#                 file_path = file_path.replace('\\', '/')
#                 folder_path = os.path.join(big_file)  # 路径，与big_file一样
#                 folder_path = folder_path.replace(big_file,"")
#                 folder_path = folder_path.replace("MAHNOB_HCI","newMAHNOB_HCI/"+filearray[0]+filearray[1])
#                 folder_path = folder_path.replace('\\', '/')
#
#                 raw_bdf_data = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')
#                 # raw_bdf_data.to_data_frame()
#                 raw_bdf_data.info['bads'] += bad_channels
#                 picks = mne.pick_types(
#                     raw_bdf_data.info, meg=False, eeg=True, stim=False,
#                     include=include, exclude='bads'
#                 )
#                 save_path = folder_path + '/' + T_file
#                 print("save_path: ", save_path)
#                 raw_bdf_data.save(save_path, picks=picks, overwrite=True)
#                 raw = mne.io.read_raw_fif(save_path, preload=True, verbose='ERROR')
#                 raw = raw.to_data_frame()
#                 df = pd.DataFrame(raw)
#                 T_file = T_file.replace('.fif', '.csv')
#                 df.to_csv(folder_path + '/' + T_file, index=False)


class BDF_2_FIF(object):
    def __init__(self, path=None):
        self.x = 1

    def MAHNOB_HCI_bdf_2_fif(self, file_path=None, folder_path=None, T_file='you.fif'):
        raw_bdf_data = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')
        want_meg = False
        want_eeg = True
        want_stim = False
        include = ['AF3', 'F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'AF4', 'F4', 'F8', 'FC6', 'T8', 'P8', 'O2']
        raw_bdf_data.info['bads'] += [
            'FC1', 'C3', 'CP5', 'CP1', 'PO3', 'Oz', 'Pz',
            'Fp2', 'Fz', 'FC2', 'Cz', 'C4', 'CP6', 'CP2',
            'PO4', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5',
            'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1',
            'Erg2', 'Resp', 'Temp', 'Status', 'P3', 'P4', 'Fp1'
        ]
        picks = mne.pick_types(
            raw_bdf_data.info, meg=want_meg, eeg=want_eeg, stim=want_stim,
            include=include, exclude='bads'
        )
        save_path = folder_path + T_file
        raw_bdf_data.save(save_path, picks=picks, overwrite=True)


# useful
def MAHNOB_HCI_TEST2(file_path="../dataset/Part_1_S_Trial1_emotion.bdf", folder_path="../dataset/", T_file='you.fif'):
    # print("ma: ", os.path)
    raw_bdf_data = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')
    # df_data = raw_bdf_data.to_data_frame()
    # mne.write_events("../dataset/Part_1_S_Trial1_emotion_eve.fif", df_data)
    want_meg = False
    want_eeg = True
    want_stim = False
    include = ['AF3', 'F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'AF4', 'F4', 'F8', 'FC6', 'T8', 'P8', 'O2']
    raw_bdf_data.info['bads'] += [
        'FC1', 'C3', 'CP5', 'CP1', 'PO3', 'Oz', 'Pz',
        'Fp2', 'Fz', 'FC2',	'Cz', 'C4',	'CP6', 'CP2',
        'PO4', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5',
        'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1',
        'Erg2', 'Resp', 'Temp', 'Status',  'P3', 'P4', 'Fp1'
    ]
    picks = mne.pick_types(
        raw_bdf_data.info, meg=want_meg, eeg=want_eeg, stim=want_stim,
        include=include, exclude='bads'
    )
    save_path = folder_path + T_file
    raw_bdf_data.save(save_path, picks=picks, overwrite=True)

    print(".fif completed!")
    #
    # path2 = "../dataset/raw_22_54_raw.fif"
    # raw_fif_data = mne.io.read_raw_fif(path2, preload=True, verbose='ERROR')
    # df = pd.DataFrame(raw_fif_data.to_data_frame())
    # df.to_csv("../dataset/raw_22_54_raw.csv", index=False)

    print("MAHNOB_HCI_TEST2.end...")
# useful

def MAHNOB_HCI_test():
    # AF3	F3	F7	FC5	T7	P7	O1	AF4	F4	F8	FC6	T8	P8	O2
    # MAHNOB_HCI = ['AF3', 'F3', 'F7', 'FC5', 'T7', 'P7', 'O1', 'AF4', 'F4', 'F8', 'FC6', 'T8', 'P8', 'O2']
    # bad = [
    # 'FC1', 'C3', 'CP5', 'CP1', 'PO3',	'Oz', 'Pz',
    # 'Fp2', 'Fz', 'FC2',	'Cz', 'C4',	'CP6', 'CP2',
    # 'PO4', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5',
    # 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1',
    # 'Erg2', 'Resp', 'Temp', 'Status'
    # ]
    path = "../dataset/Part_1_S_Trial1_emotion.bdf"
    raw_bdf_data = mne.io.read_raw_bdf(path, preload=True, verbose='ERROR')
    df = pd.DataFrame(raw_bdf_data.to_data_frame())
    df.to_csv("../dataset/Part_1_S_Trial1_emotion.csv", index=False)
    print("MAHNOB_HCI_test().end...")


def DEAP_TEST2(path="../dataset/s01.bdf"):
    raw_bdf_data = mne.io.read_raw_bdf(path, preload=True, verbose='ERROR')
    # df_data = raw_bdf_data.to_data_frame()
    # mne.write_events("../dataset/Part_1_S_Trial1_emotion_eve.fif", df_data)
    want_meg = False
    want_eeg = True
    want_stim = False
    include = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    raw_bdf_data.info['bads'] += [
        'Fp1', 'FC1', 'C3', 'CP1', 'CP5',
        'Pz', 'PO3', 'Oz', 'PO4', 'CP6',
        'CP2', 'C4', 'FC2', 'Fp2', 'Fz',
        'Cz', 'EXG1', 'EXG2', 'EXG3', 'EXG4',
        'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1',
        'GSR2', 'Erg1', 'Erg2', 'Resp',
        'Plet', 'Temp', 'Status', 'P3', 'P4'
    ]
    picks = mne.pick_types(
        raw_bdf_data.info, meg=want_meg, eeg=want_eeg, stim=want_stim,
        include=include, exclude='bads'
    )
    raw_bdf_data.save("../dataset/raw_23_49_raw.fif", picks=picks, overwrite=True)

    print(".fif completed!")

    path2 = "../dataset/raw_23_49_raw.fif"
    raw_fif_data = mne.io.read_raw_fif(path2, preload=True, verbose='ERROR')
    df = pd.DataFrame(raw_fif_data.to_data_frame())
    df.to_csv("../dataset/raw_23_49_raw.csv", index=False)

    print("main.end...")


def DEAP_test():
    # AF3	F7	F3	FC5	T7	P7	O1	O2	P8	T8	FC6	F4	F8	AF4
    # DEAP = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    path = "../dataset/s01.bdf"
    raw_bdf_data = mne.io.read_raw_bdf(path, preload=True, verbose='ERROR')
    df = pd.DataFrame(raw_bdf_data.to_data_frame())
    df.to_csv("../dataset/s01.csv", index=False)
    print("DEAP_test().end...")


def test():
    # MAHNOB_HCI_TEST2()
    # DEAP_TEST2()

    # 导入需要用的包
    # 获取多个文件夹的路径，并返回一个可迭代对象
    path = '../dataset2'
    dirPath = glob.iglob('../dataset2/*')
    # 将可迭代的对象进行循环获取，赋值给 big_file
    for big_file in dirPath:
        # 获取每个文件夹下的文件名并赋值给 file
        files = os.listdir(big_file)
        this_big_file = big_file.replace(path, '')
        this_big_file = this_big_file.replace('\\', '')
        print("this_big_file: ", this_big_file)   # 路径
        # 将获取的所有文件名进行循环判断
        for file in files:
            # 判断扩展名只为.bdf       ########不包含.origin.txt
            if file.endswith(".csv"):  # and not file.endswith(".origin.txt"):
                '''
                    进行文件完整路径拼接并读取数据
                    file为文件名
                    os.path.join(big_file, file)为路径
                '''
                # print("file: ", file)       # 文件名
                T_file = file.replace('.csv', '.fif')
                T_file = T_file.replace('\\', '/')
                # print("T_file: ", T_file)
                file_path = os.path.join(big_file, file)  # 路径+文件名
                file_path = file_path.replace('\\', '/')
                folder_path = os.path.join(big_file)  # 路径，与big_file一样
                folder_path = folder_path.replace('\\', '/')
                print("folder_path: ", folder_path)
                print("file_path: ", file_path)


def MAHNOB_HCI_bdf_2_fif():
    # want_meg = False
    # want_eeg = True
    # want_stim = False
    # dataset_path = configuration.DATASET_PATH + "MAHNOB_HCI/*"
    dataset_path = "C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*"
    include = ['Fp1', 'T7', 'CP1', 'Oz', 'Fp2', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'PO4']
    bad_channels = [
        'FC1', 'C3', 'CP5', 'PO3', 'Pz',
        'Fz', 'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5',
        'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1',
        'Erg2', 'Resp', 'Temp', 'Status', 'P3', 'P4',
        'AF3', 'F3', 'F7', 'FC5', 'P7', 'O1', 'AF4', 'F4', 'P8', 'O2'
    ]
    # 获取多个文件夹的路径，并返回一个可迭代对象
    dirPath = glob.iglob(dataset_path)
    path = dataset_path.replace('/*', '')
    # 将可迭代的对象进行循环获取，赋值给 big_file
    for big_file in dirPath:
        # 获取每个文件夹下的文件名并赋值给 file
        files = os.listdir(big_file)
        this_big_file = big_file.replace(path, '')
        this_big_file = this_big_file.replace('\\', '')
        print("\n\n\n")
        print("big_file: ", this_big_file)   # 路径
        # 将获取的所有文件名进行循环判断
        for file in files:
            # 判断扩展名只为.bdf       ########不包含.origin.txt
            if file.endswith(".bdf"):  # and not file.endswith(".origin.txt"):
                '''
                    进行文件完整路径拼接并读取数据
                    file为文件名
                    os.path.join(big_file, file)为路径
                '''
                # print("file: ", file)       # 文件名
                filearray = file.split("_")  #.bdf文件名称根据_分割
                T_file = filearray[0]+filearray[1]+"_"+filearray[3]+".fif"#拼接成新的文件名
                # T_file = file.replace('.bdf', '.fif')
                T_file = T_file.replace('\\', '/')#将\\替换为/
                print("T_file: ", T_file)
                # print("BIG_file: ", big_file)
                file_path = os.path.join(big_file, file)  # 路径+文件名
                # print("file_path1: ", file_path)
                # file_path = file_path.replace('\\', '/')
                # print("file_path2: ", file_path)
                folder_path = os.path.join(big_file)  # 路径，与big_file一样
                #print("folder_path1: ", folder_path)
                folder_path = folder_path.replace(this_big_file, "")
                # print("folder_path2: ", folder_path)
                folder_path = folder_path.replace("MAHNOB_HCI","newMAHNOB_HCI/"+filearray[0]+filearray[1])
                # print("folder_path3: ", folder_path)
                folder_path = folder_path.replace('\\', '/')#拼接成新的文件路径
                print("folder_path4: ", folder_path)
                raw_bdf_data = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')#打开bdf文件
                # raw_bdf_data.to_data_frame()
                raw_bdf_data.info['bads'] += bad_channels
                picks = mne.pick_types(
                    raw_bdf_data.info, meg=False, eeg=True, stim=False,
                    include=include, exclude='bads')
                save_path = folder_path + T_file
                print("save_path: ", save_path)
                # raw_bdf_data.save(save_path, picks=picks, overwrite=True)
                raw = mne.io.read_raw_fif(save_path, preload=True, verbose='ERROR')
                raw = raw.to_data_frame()
                df = pd.DataFrame(raw)
                T_file = T_file.replace('.fif', '.csv')
                number0 = T_file.split(".")
                number1 = number0[0].split("Trial")
                length0 = len(number1)
                detail = number1[length0 - 1]
                # print(T_file)
                # print(detail)
                # print(folder_path +detail+'/'+ T_file,)
                df.to_csv(folder_path +detail+'/'+ "csv.csv", index=False)



def MAHNOB_HCI_bdf_2_fif_20210312():
    # want_meg = False
    # want_eeg = True
    # want_stim = False
    # dataset_path = configuration.DATASET_PATH + "MAHNOB_HCI/*"
    dataset_path = "C:/Users/ps/Desktop/datasets/hci-tagging-database_download_2020-09-25_01_09_28/Sessions/*"
    include = ['Fp1', 'T7', 'CP1', 'Oz', 'Fp2', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'PO4']
    bad_channels = [
        'FC1', 'C3', 'CP5', 'PO3', 'Pz',
        'Fz', 'EXG1', 'EXG2', 'EXG3', 'EXG4',
        'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1',
        'GSR2', 'Erg1', 'Erg2', 'Resp', 'Temp',
        'Status', 'P3', 'P4', 'AF3', 'F3', 'F7',
        'FC5', 'P7', 'O1', 'AF4', 'F4', 'P8', 'O2']
    # 获取多个文件夹的路径，并返回一个可迭代对象
    dirPath = glob.iglob(dataset_path)
    path = dataset_path.replace('/*', '')
    # 将可迭代的对象进行循环获取，赋值给 big_file
    i = 0
    j = 0
    for big_file in dirPath:
        # 获取每个文件夹下的文件名并赋值给 file
        files = os.listdir(big_file)
        this_big_file = big_file.replace(path, '')
        this_big_file = this_big_file.replace('\\', '')
        print("\n\n\n")
        print("this_big_file: ", this_big_file)   # 路径
        print("big_file: ", big_file)
        print("files: ", files)
        # 将获取的所有文件名进行循环判断
        i += 1
        print("i: ", i)

        for file in files:
            print("file: ", file)
            # 判断扩展名只为.bdf       ########不包含.origin.txt
            if file.endswith(".bdf"):  # and not file.endswith(".origin.txt"):
                '''
                    进行文件完整路径拼接并读取数据
                    file为文件名
                    os.path.join(big_file, file)为路径
                '''
                j += 1
                print("file: ", file)

                # print("file: ", file)       # 文件名
                filearray = file.split("_")  #.bdf文件名称根据_分割
                T_file = filearray[0]+filearray[1]+"_"+filearray[3]+".fif"#拼接成新的文件名
                T_file = T_file.replace('\\', '/')#将\\替换为/

                print("T_file: ", T_file)

                file_path = os.path.join(big_file, file)  # 路径+文件名
                folder_path = os.path.join(big_file)  # 路径，与big_file一样
                folder_path = folder_path.replace(this_big_file, "")

                # print("folder_path2: ", folder_path)

                folder_path = folder_path.replace("MAHNOB_HCI","newMAHNOB_HCI/"+filearray[0]+filearray[1])

                # print("folder_path3: ", folder_path)

                folder_path = folder_path.replace('\\', '/')#拼接成新的文件路径

                print("folder_path4: ", folder_path)

                raw_bdf_data = mne.io.read_raw_bdf(file_path, preload=True, verbose='ERROR')#打开bdf文件

                # raw_bdf_data.to_data_frame()

                raw_bdf_data.info['bads'] += bad_channels
                picks = mne.pick_types(
                    raw_bdf_data.info, meg=False, eeg=True, stim=False,
                    include=include, exclude='bads')

                big_file_prime = big_file.replace('\\', '/')
                save_path_prime = big_file_prime + '/' + T_file
                print("save_path_prime: ", save_path_prime)

                save_path = folder_path + T_file
                # save_path = files + T_file
                print("folder_path + big_file: ", folder_path + big_file)

                print("save_path: ", save_path)

                raw_bdf_data.save(save_path_prime, picks=picks, overwrite=True)
                # raw = mne.io.read_raw_fif(save_path, preload=True, verbose='ERROR')
                # raw = raw.to_data_frame()

                # df = pd.DataFrame(raw)
                '''
                T_file = T_file.replace('.fif', '.csv')
                number0 = T_file.split(".")
                number1 = number0[0].split("Trial")
                length0 = len(number1)
                detail = number1[length0 - 1]
                
                # print(T_file)
                # print(detail)
                # print(folder_path +detail+'/'+ T_file,)

                # df.to_csv(folder_path +detail+'/'+ "csv.csv", index=False)
                # print(df)
                '''
    print("j: ", j)


if __name__ == "__main__":

    # test()
    # MAHNOB_HCI_bdf_2_fif()
    MAHNOB_HCI_bdf_2_fif_20210312()
    # DEAP_bdf_2_fif()
    print("\n\n\nmain.end...")
