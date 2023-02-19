import logging
from mat4py import loadmat
import numpy as np


def CVIQD_mat2txt(path):
    """
    :return two txt file about dis_label and ref_label
    """
    data = loadmat(path)
    data = np.array(data['CVIQ'])
    half_idx = data.shape[0] // 2
    dis_f = open('CVIQD_dis_label.txt', 'w')
    ref_f = open('CVIQD_ref_label.txt', 'w')
    count_ref = 0
    count_dis = 0
    for i in range(half_idx):
        for j in range(data.shape[1]):
            if int(data[i][j][:3]) % 34 == 0:
                ref_f.write(data[i][j] + ' ' + str(data[i + half_idx][j]) + '\n')
                count_ref += 1
            else:
                dis_f.write(data[i][j] + ' ' + str(data[i + half_idx][j]) + '\n')
                count_dis += 1
    
    ref_f.close()
    dis_f.close()
    print("========== Transform successfully! ==========")
    print("Include {} ref images and {} dis images".format(count_ref, count_dis))






if __name__ == '__main__':
    CVIQD_mat2txt('./CVIQ.mat')

