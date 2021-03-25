import numpy as np





if __name__ == "__main__":
    x = np.load('EEG.npy')
    print(x.shape)
    y = np.load('EEG_1.npy')
    print(y.shape)
