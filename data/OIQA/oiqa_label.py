import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.erp2rec import ERP2REC
import random


def renumber_table(table):
    vis_array = np.zeros(table.shape[0] * table.shape[1])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if table[i][j] <= 30:
                vis_array[int(table[i][j] - 1)] = 1
    for z in range(vis_array.shape[0]):
        if vis_array[z] != 1:
            min_distance_val = z + 1
            while min_distance_val not in table:
                min_distance_val += 1
            for i in range(table.shape[0]):
                for j in range(table.shape[1]):
                    if table[i][j] == min_distance_val:
                        cur_val = table[i][j]
                        table[i][j] = z + 1
                        vis_array[z] = 1
                        if int(cur_val - 1) < 30:
                            vis_array[int(cur_val - 1)] = 0
    return table


def sort_viewports_with_table(table, viewports_list):
    sorted_viewports = []
    idx2viewports = {}
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            idx2viewports[table[i][j]] = viewports_list[i * table.shape[1] + j]
    for i in sorted(idx2viewports):
        sorted_viewports.append(idx2viewports[i])
    return sorted_viewports


class OIQA(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, viewport_size,
                 viewport_num, grid_size, decision_method):
        super(OIQA, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform
        self.viewport_size = viewport_size
        self.viewport_num = viewport_num
        self.grid_size = grid_size
        self.decision_method = decision_method
        self.domain_transform = ERP2REC()

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                if dis in list_name:
                    score = float(score)
                    if os.path.exists(os.path.join(self.dis_path, dis) + ".jpg"):
                        dis_files_data.append(dis + ".jpg")
                    elif os.path.exists(os.path.join(self.dis_path, dis) + ".png"):
                        dis_files_data.append(dis + ".png")
                    else:
                        print("There is no data {}".format(dis))
                    score_data.append(score)

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img = cv2.imread(os.path.join(self.dis_path, d_img_name), cv2.IMREAD_COLOR)
        d1 = self.select_viewports(d_img, start=(0, 0), method=self.decision_method)
        for i in range(len(d1)):
            d1[i] = cv2.cvtColor(d1[i], cv2.COLOR_BGR2RGB)
            d1[i] = np.array(d1[i]).astype('float32') / 255
            d1[i] = np.transpose(d1[i], (2, 0, 1))

        self.vis_table = renumber_table(self.vis_table)
        score = self.data_dict['score_list'][idx]
        d1 = np.array(d1)
        sample = {
            'd_img_org': d1,
            'score': score,
            'table': self.vis_table,
            'name': d_img_name
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range
    
    def select_viewports_kedema(self, x):
        viewports_list = []
        yaw_list = [0, -24, -48, -72, -90, -72, -48, -24, 0, 24, 48, 72, 90, 72, 48, 24, 0]
        for i in range(len(yaw_list)):
            viewport = self.domain_transform.toREC(
                frame=x,
                center_point=np.array([yaw_list[i], 0]),
                width=self.viewport_size[0],
                height=self.viewport_size[1]
            )
            viewports_list.append(viewport)
        return viewports_list
    
    def cal_std(self, sig):
        return np.std(sig)
    
    def get_img_std(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        img_std = self.cal_std(gray_img.flatten())
        return img_std
    
    def cal_entropy(self, sig):
        """ Calculate the entropy
        Args:
            sig (ndarray): the one-dim array flatten from a window
        """
        len = sig.size
        sig_set = list(set(sig))
        p_list = [np.size(sig[sig == i]) / len for i in sig_set]
        entropy = np.sum([p * np.log2(1.0 / p) for p in p_list])
        return entropy
    
    def get_img_entropy(self, img):
        """ Calculate the entropy of an image
        Args:
            img (BGR type): the image read from cv2
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        sum_entropy = self.cal_entropy(gray_img.flatten())
        return sum_entropy
    
    def cal_next_patch_coordinate(self, viewport, cur_coordinate=(0, 0), method='ent'):
        """ Calculate the next coordinate viewport according to the entropy
        Args:
            x (cv2): the current viewport
            vis_table (numpy): the table record which coordinate has been visited
            cur_coordinate (tuple): current coordinate. Defaults to (0, 0).

        Returns:
            [tuple]: the next viewport coordinate
        """
        r, c = cur_coordinate[0], cur_coordinate[1]
        idx2next_coordinate = {0:(-1, -1), 1:(c - 60, r + 45), 2:(c + 0, r + 45), 3:(c + 60, r + 45),
                        4:(c - 60, r + 0), 6:(c + 60, r + 0),
                        7:(c - 60, r - 45), 8:(c + 0, r - 45), 9:(c + 60, r - 45)}
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
                _c = self.modify_c[_c]
                x, y = self.r2x[_r], self.c2y[_c]
                if self.vis_table[x][y] != 0:
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
            next_c = self.modify_c[next_c]
        return (next_r, next_c)

    def dfs_get_viewport(self, img, cur_coordinate=(0, 0), method='ent'):
        r, c = cur_coordinate[0], cur_coordinate[1]
        viewport = self.domain_transform.toREC(
            frame=img,
            center_point=np.array([c, r]),
            width=self.viewport_size[0],
            height=self.viewport_size[1]
        )
        x, y = self.r2x[r], self.c2y[c]
        self.vis_table[x][y] = self.track_idx
        self.track_idx += 1
        # print(vis_table)
        for i in range(8):
            if self.back_trace == True:
                r, c = cur_coordinate[0], cur_coordinate[1]
                x, y = self.r2x[r], self.c2y[c]
                self.vis_table[x][y] = self.track_idx
                self.track_idx += 1
                self.back_trace = False
                # print(self.vis_table)

            next_coordinate = self.cal_next_patch_coordinate(viewport, (r, c), method=method)
            next_r, next_c = next_coordinate[0], next_coordinate[1]
            if next_r == -1 and next_c == -1:
                if 0 in self.vis_table:
                    self.back_trace = True
                return
            else:
                # print("next_r: {}, next_c: {}".format(next_r, next_c))
                self.dfs_get_viewport(img, cur_coordinate=(next_r, next_c), method=method)
        return
    
    def select_viewports(self, img, start=(0, 0), method='ent'):
        self.back_trace = False
        self.track_idx = 1
        self.vis_table = np.zeros((self.grid_size[0], self.grid_size[1]))
        self.r2x = {90:0, 45:1, 0:2, -45:3, -90:4}
        self.c2y = {-120:0, -60:1, 0:2, 60:3, 120:4, 180:5}
        self.modify_c = {-180:180, -120:-120, -60:-60, 0:0, 60:60, 120:120, 180:180, 240:-120}
        viewports_list = self.get_lon_lat_viewport(img)
        self.dfs_get_viewport(img, cur_coordinate=start, method=method)
        sorted_viewports = sort_viewports_with_table(table=self.vis_table, viewports_list=viewports_list)
        return sorted_viewports
    
    def get_lon_lat_viewport(self, img):
        viewports_list = []
        for key_r, val_r in self.r2x.items():
            for key_c, val_c in self.c2y.items():
                viewport = self.domain_transform.toREC(
                    frame=img,
                    center_point=np.array([key_c, key_r]),
                    width=self.viewport_size[0],
                    height=self.viewport_size[1]
                )
                viewports_list.append(viewport)
        return viewports_list





