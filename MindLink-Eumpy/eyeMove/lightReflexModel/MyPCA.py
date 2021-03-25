'''
@author: Ruixin Lee
@date: 2021.03.11
This file is for eliminate the luminance influences of light to the pupil diameter.
Version 1
首先，PCA需要将所有subjects的所有samples都结合到一个矩阵Y当中。这一步需要等数据处理完毕，再从数据中提取出矩阵。
然后，通过PCA，消除光对瞳孔直径的影响。
'''


import pandas as pd
import numpy as np


def createLightReflexModel():
    pass


def myPCA(X, k):  # k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data


if __name__ == "__main__":
    file_name = 'P1-Rec1-All-Data-New_Section_2.tsv'
    file_path = '../../data/mahnob_example/2/'
    df = pd.DataFrame(pd.read_csv(file_path+file_name))

    # PCA例子
    X = np.array([
        [-1, 1, 2, 4, 8, 9],
        [-2, -1, 3, 6, 2, 1],
        [-3, -2, 9, 4, 3, 2],
        [1, 1, 0, 3, 2, 9],
        [2, 1, 2, 2, 1, 7],
        [3, 2, 5, 6, 4, 3]
    ])
    # X = np.array([1, 2, 3, 4, 5, 6, 7])
    print(myPCA(X, 1))






