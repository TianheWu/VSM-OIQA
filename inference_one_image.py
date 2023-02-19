import os
import torch
import numpy as np
import torch.nn as nn
import random
import cv2
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils.erp2rec import ERP2REC
from models.oiqa_model import creat_model
from config import Config
from sklearn.manifold import TSNE
from einops import rearrange


os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def change_ERP_up_down_serious(x, n):
    H, W, C = x.shape
    ret = np.zeros_like(x)
    row_list = []
    pre = 0
    step = H // n
    for i in range(n):
        x_tmp = x[pre:pre + step, :, :]
        pre = pre + step
        row_list.append(x_tmp)
    random.shuffle(row_list)
    pre = 0
    for i in range(n):
        ret[pre:pre + step, :, :] = row_list[i]
        pre = pre + step
    return ret

def change_ERP_left_right_serious(x, n):
    H, W, C = x.shape
    ret = np.zeros_like(x)
    row_list = []
    pre = 0
    step = W // n
    for i in range(n):
        x_tmp = x[:, pre:pre + step, :]
        pre = pre + step
        row_list.append(x_tmp)
    random.shuffle(row_list)
    pre = 0
    for i in range(n):
        ret[:, pre:pre + step, :] = row_list[i]
        pre = pre + step
    return ret


def change_ERP_up_down(x):
    H, W, C = x.shape
    ret = np.zeros_like(x)
    x1 = x[:H // 2, :, :]
    x2 = x[H // 2:, :, :]
    ret[H // 2:, :, :] = x1
    ret[:H // 2, :, :] = x2
    return ret

def change_ERP_left_right(x):
    H, W, C = x.shape
    ret = np.zeros_like(x)
    x1 = x[:, :W // 2, :]
    x2 = x[:, W // 2:, :]
    ret[:, W // 2:, :] = x1
    ret[:, :W // 2, :] = x2
    return ret


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


class DataOI(torch.utils.data.Dataset):
    def __init__(self, dis_path, viewport_size, grid_size, decision_method):
        super(DataOI, self).__init__()
        self.domain_transform = ERP2REC()
        img = cv2.imread(dis_path, cv2.IMREAD_COLOR)

        self.grid_size = grid_size
        self.viewport_size = viewport_size
        d = self.select_viewports(img, start=(0, 0), method=decision_method)
        for i in range(len(d)):
            d[i] = cv2.cvtColor(d[i], cv2.COLOR_BGR2RGB)
            d[i] = np.array(d[i]).astype('float32') / 255
            d[i] = np.transpose(d[i], (2, 0, 1))

        self.vis_table = renumber_table(self.vis_table)
        d = np.array(d)
        d = (d - 0.5) / 0.5
        self.d_img = torch.from_numpy(d).type(torch.FloatTensor)
        self.vis_table = torch.from_numpy(self.vis_table).type(torch.FloatTensor)

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range
    
    def cal_std(self, sig):
        return np.std(sig)
    
    def get_img_std(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        img_std = self.cal_std(gray_img.flatten())
        return img_std

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

        for i in range(8):
            if self.back_trace == True:
                r, c = cur_coordinate[0], cur_coordinate[1]
                x, y = self.r2x[r], self.c2y[c]
                self.vis_table[x][y] = self.track_idx
                self.track_idx += 1
                self.back_trace = False

            next_coordinate = self.cal_next_patch_coordinate(viewport, (r, c), method=method)
            next_r, next_c = next_coordinate[0], next_coordinate[1]
            if next_r == -1 and next_c == -1:
                if 0 in self.vis_table:
                    self.back_trace = True
                return
            else:
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


def t_sne(data, label, title, save_path):
    # t-sne处理
    print('starting T-SNE process')
    data = TSNE(n_components=2, perplexity=2, learning_rate='auto', init='pca').fit_transform(data)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
    df.insert(loc=1, column='label', value=label)
    print('Finished')

    sns.scatterplot(x='x', y='y', hue='label', s=3, palette="Set2", data=df)
    plt.title(title)
    plt.savefig(save_path, dpi=400)


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "data_path": "./test_images/4.png",

        # model
        "decision_method": "ent",
        "viewport_size": (224, 224),
        "grid_size": (5, 6),
        "input_dim": 3,
        "embed_dim": 1024,
        "num_outputs": 1,
        "num_heads": 16,
        "depth": 4,
        
        # checkpoint
        "pretrained_weight_path": "./output/models/vsm_mvaqd_s20/epoch256.pt",
        "save_img_path": "./feature_map/"
    })

    if not os.path.exists(config.save_img_path):
        os.makedirs(config.save_img_path)

    Data_oi = DataOI(dis_path=config.data_path, viewport_size=config.viewport_size, grid_size=config.grid_size,
        decision_method=config.decision_method)

    net = creat_model(config=config, model_weight_path=config.pretrained_weight_path, pretrained=True)
    net = nn.DataParallel(net).cuda()

    with torch.no_grad():
        net.eval()
        pred_d, x_before_gru, x_after_gru = net(Data_oi.d_img.unsqueeze(0), Data_oi.vis_table.unsqueeze(0))
        x_before_gru = x_before_gru.squeeze(0)
        x_after_gru = x_after_gru.squeeze(0)

        label = np.expand_dims(np.arange(0, 30), axis=1)

        save_path1 = config.save_img_path + "x_before_gru.jpg"
        save_path2 = config.save_img_path + "x_after_gru.jpg"

        t_sne(data=x_before_gru.cpu().numpy(), label=label, title="before gru", save_path=save_path1)
        t_sne(data=x_after_gru.cpu().numpy(), label=label, title="before gru", save_path=save_path2)

        print("OI score: ", pred_d)

