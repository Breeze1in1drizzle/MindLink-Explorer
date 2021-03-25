import glob
import os
import shutil

import configuration


def MAHNOB_HCI():
    dataset_path = configuration.DATASET_PATH + "MAHNOB_HCI/*"
    save_path = configuration.DATASET_PATH + "newMAHNOB_HCI/Part"
    dirPath = glob.iglob(dataset_path)
    path = dataset_path.replace('/*', '')
    for big_file in dirPath:
        files = os.listdir(big_file)
        this_big_file = big_file.replace(path, '')
        this_big_file = this_big_file.replace('\\', '')
        print("big_file:", this_big_file)
        for file in files:
            if file.endswith(".fif"):
                file_array = file.split("_")
                new_file_name = file_array[0]+"_"+file_array[1]+file_array[3]+".fif"

        shutil.move(big_file+"/"+file, "C:\\Users\\xiaojian\\Desktop\\1")
                # print(file.split("_")[2])
                # file_name = file.


def newDirectory():
    save_path = configuration.DATASET_PATH + "newMAHNOB_HCI/Part"
    for i in range(1,30):
        t=str(i)
        os.makedirs(save_path+t)


if __name__ == "__main__":

    print("start")

    MAHNOB_HCI()

    print("end")
