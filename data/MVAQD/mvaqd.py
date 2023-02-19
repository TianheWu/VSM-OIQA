import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils_tools.erp2rec import ERP2REC

import random
# import joblib


class MVAQD(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, viewport_size,
                 viewport_nums, decision_method, fov, start_points):
        super(MVAQD, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.viewport_size = viewport_size
        self.viewport_nums = viewport_nums
        self.decision_method = decision_method
        self.fov = fov
        self.domain_transform = ERP2REC()
        self.start_points = start_points
        # self.name_track_dict = joblib.load('./data/MVAQD/name_track_dict.pkl')
        
        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                ref, dis, score = line.split()
                if dis in list_name:
                    dis_files_data.append(dis)
                    score = float(score)
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data, range_val = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)

        self.range_val = range_val

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)

        # track = self.name_track_dict[d_img_name]
        # d1 = []
        # for i in range(len(track)):
        #     viewport = self.domain_transform.toREC(
        #         frame=d_img,
        #         center_point=np.array([track[i][0], track[i][1]]),
        #         FOV=self.fov,
        #         width=self.viewport_size[0],
        #         height=self.viewport_size[1]
        #     )
        #     d1.append(viewport)

        d1 = self.select_viewports(d_img, start=self.start_points, method=self.decision_method)
        for i in range(d1.shape[0]):
            for j in range(d1.shape[1]):
                # d1[i][j] = cv2.resize(d1[i][j], (224, 224))
                d1[i][j] = cv2.cvtColor(d1[i][j], cv2.COLOR_BGR2RGB)
                d1[i][j] = np.array(d1[i][j]).astype('float32') / 255
        d1 = np.transpose(d1, (0, 1, 4, 2, 3))

        score = self.data_dict['score_list'][idx]
        d1 = np.array(d1)
        sample = {
            'd_img_org': d1,
            'score': score,
            'name': d_img_name,
            'range_val': self.range_val,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range, range

    def cal_entropy(self, sig):
        len = sig.size
        sig_set = list(set(sig))
        p_list = [np.size(sig[sig == i]) / len for i in sig_set]
        entropy = np.sum([p * np.log2(1.0 / p) for p in p_list])
        return entropy
    
    def get_img_entropy(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        sum_entropy = self.cal_entropy(gray_img.flatten())
        return sum_entropy
    
    def cal_next_patch_coordinate(self, viewport, cur_coordinate=(0, 0), method='ent'):
        r, c = cur_coordinate[0], cur_coordinate[1]
        idx2next_coordinate = {0:(-1, -1), 1:(c - 60, r + 60), 2:(c + 0, r + 60), 3:(c + 60, r + 60),
                        4:(c - 60, r + 0), 6:(c + 60, r + 0),
                        7:(c - 60, r - 60), 8:(c + 0, r - 60), 9:(c + 60, r - 60)}
        H, W, C = viewport.shape
        patch_size = H // 4
        kernel_size = patch_size * 2
        idx = 1
        max_val = 0
        max_val_idx = 0
        for i in range(0, W - patch_size, patch_size):
            for j in range(0, H - patch_size, patch_size):
                if idx == 5:
                    idx += 1
                    continue
                _c, _r = idx2next_coordinate[idx][0], idx2next_coordinate[idx][1]
                if _r > 90 or _r < -90:
                    idx += 1
                    continue

                if _c > 180:
                    _c = -180 + (_c - 180)
                elif _c < -180:
                    _c = 180 - (-180 - _c)

                if str(_r) + str(_c) in self.coord_tabel.keys():
                    idx += 1
                    continue
                    
                window = viewport[i:i + kernel_size, j:j + kernel_size, :]
                if method == 'ent':
                    val = self.get_img_entropy(window)
                elif method == 'std':
                    val = self.get_img_std(window)
                if val > max_val:
                    max_val = val
                    max_val_idx = idx
                idx += 1
        next_coordinate = idx2next_coordinate[max_val_idx]
        next_c, next_r = next_coordinate[0], next_coordinate[1]
        if next_c != -1:
            if next_c > 180:
                next_c = -180 + (next_c - 180)
            elif next_c < -180:
                next_c = 180 - (-180 - next_c)
        return (next_r, next_c)

    def dfs_get_viewport(self, img, cur_coordinate=(0, 0), method='ent'):
        r, c = cur_coordinate[0], cur_coordinate[1]
        viewport = self.domain_transform.toREC(
            frame=img,
            center_point=np.array([c, r]),
            FOV=self.fov,
            width=self.viewport_size[0],
            height=self.viewport_size[1]
        )
        self.coord_tabel[str(r) + str(c)] = 1
        self.viewports_list.append(viewport)

        if len(self.viewports_list) == self.viewport_nums:
            return

        for i in range(8):
            if len(self.viewports_list) == self.viewport_nums:
                return
            next_coordinate = self.cal_next_patch_coordinate(viewport, (r, c), method=method)
            next_r, next_c = next_coordinate[0], next_coordinate[1]
            if next_r == -1 and next_c == -1:
                return
            else:
                # print("next_r: {}, next_c: {}".format(next_r, next_c))
                self.dfs_get_viewport(img, cur_coordinate=(next_r, next_c), method=method)
        return
    
    def select_viewports(self, img, start=(0, 0), method='ent'):
        self.seq_list = []
        for i in range(len(self.start_points)):
            self.coord_tabel = {}
            self.viewports_list = []
            self.dfs_get_viewport(img, cur_coordinate=self.start_points[i], method=method)
            self.seq_list.append(np.array(self.viewports_list, dtype=np.float32))
        return np.array(self.seq_list)