'''
Created on Nov 3 2017
Modified on Nov 13 2019
Modified on March 25 2020
@author1: Yongrui Huang
@author2: Ruixin Lee
'''

import pandas as pd
import numpy as np
import os
import configuration


def makedir():
    '''
        Create a subject's directory in target path from configuration.py
    '''
    base_path = configuration.COLLECT_DATA_PATH
    i = 0
    while True:
        i += 1
        path = '%s%d/' % (base_path, i,)
        
        is_exists=os.path.exists(path)
        if not is_exists:
            os.makedirs(path)
            break
    return i, path


def make_trial_dir(path, trial_num):
    '''
        Create trial's directory for subjects
        
        Arguments:
        
            path: the target path
            
            trial_num: how much trail this subject will do.
    '''
    for i in range(1, int(trial_num)+1):
        os.makedirs(path + "/trial_" + str(i))


def creat_subject(name, age, gender, trial_num):
    '''
        Create a subject csv file for each subject when they fill their information
       
        Arguments:
        
            name: the subject's name in English, a string
            
            age: the subject's age, a integer

            gender: the subject's gender, female or male
            
            trial_num: the number of trials in which every subject needs to participate,
                        ranges from 1 to 40
    '''
    subject_id, path = makedir()
    label_frame = pd.DataFrame(
        columns=['trail_id', 'valence', 'arousal']
    )
    label_frame.to_csv(path+'label.csv', index=False)
    # information_frame = pd.DataFrame(
    #     np.array(
    #         [
    #             [subject_id, name, age, gender],
    #
    #         ]
    #     ),
    #     columns=['subject_id', 'name', 'age', 'gender']
    # )
    print("subject_id:", subject_id, ", name:", name, " age:", age, " gender:", gender)
    information_frame = pd.DataFrame(pd.read_csv("../dataset/collected_dataset/information.csv"))
    information_frame = information_frame.append(
        [
            {
                'subject_id': subject_id,
                'name': name, 'age': age, 'gender': gender
            }
        ],
        ignore_index=True
    )
    print("information: \n", information_frame)
    save_info_path = "../dataset/collected_dataset/information.csv"
    information_frame.to_csv(save_info_path, index=False)
    # information_frame.to_csv(path+'/information.csv', index=False)
    make_trial_dir(path, trial_num)
    # subject_id = int(subject_id)
    # np.save("../subject_id.npy", subject_id)
    # np.save("subject_id.npy", subject_id)


# save_SAM_label(trial_id, valence, arousal)
def save_SAM_label(trial_id, valence, arousal):
    '''
        Create a .csv file to storage the ground truth labels of the subject
        
        Arguments:
        
            trial_id: which trial it is.
            
            valence: the ground truth labels in valence space, ranges from 1 to 9 [1, 9]
            
            arousal: the ground truth labels in arousal space, ranges from 1 to 9 [1, 9]
    '''
    base_path = configuration.COLLECT_DATA_PATH
    # subject_id = np.load('subject_id.npy')
    infomation_path = "../dataset/collected_dataset/information.csv"
    infomation_frame = pd.DataFrame(pd.read_csv(infomation_path))
    if len(infomation_frame) == 0:
        subject_index = 0
    else:
        subject_index = len(infomation_frame)-1
    print(subject_index)
    subject_id = infomation_frame.iat[subject_index, 0]
    print("subject id: ", subject_id)
    # subject_id = np.load('../subject_id.npy')     Is this path true?
    # The 'subject_id.npy' is in the path "Eumpy-master\data_colletcion_framework\subject_id.npy".
    # Is 'subject_id.npy' the true path?    From Ruixin Lee

    # path = base_path + str(subject_id) + '/trial_' + str(trial_id) + '/label.csv'
    # save_path =\
    #     "../dataset/collected_dataset/" + str(subject_id) + "/trial_" + str(trial_id) + "/label.csv"
    # SAM_dataframe = pd.DataFrame(
    #     np.array(
    #         [
    #             [trial_id, valence, arousal],
    #
    #         ]
    #     ),
    #     columns=['valence', 'arousal']
    # )

    label_load_path = "../dataset/collected_dataset/" + str(subject_id) + "/label.csv"
    SAM_dataframe = pd.DataFrame(pd.read_csv(label_load_path))

    # append a new row: trail_id, valence, arousal
    SAM_dataframe = SAM_dataframe.append([{'trail_id': trial_id, 'valence': valence, 'arousal': arousal}], ignore_index=True)


    print("SAM_dataframe: \n", SAM_dataframe)
    SAM_dataframe.to_csv(label_load_path, index=False)
    df = pd.DataFrame(pd.read_csv(label_load_path))
    str_load_ = "D:/workSpace/python_workspace/Eumpy-master/dataset/label.csv"
    df.to_csv(str_load_, index=False)
    print("df:\n", df)
# save_SAM_label(trial_id, valence, arousal)


# __main__
if __name__ == '__main__':
    pass
# __main__
