import os
import torch
import numpy as np
import logging
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.oiqa_model import creat_model
from config import Config
from utils.process import Normalize, ToTensor
from utils.process import split_dataset_cviqd, split_dataset_iqaodi, split_dataset_oiqa, split_dataset_mvaqd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


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
        "dataset_name": "mvaqd",
        "oiqa_train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/OIQA/distorted_images/",
        "cviqd_train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/CVIQ_database/CVIQ/",
        "iqaodi_train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/IQA-ODI/all_ref_test_img/",
        "mvaqd_train_dis_path": "/mnt/data_16TB/wth22/IQA_dataset/MVAQD-dataset/",
        "oiqa_dis_label": "./data/OIQA/OIQA_dis_label.txt",
        "oiqa_ref_label": "./data/OIQA/OIQA_ref_label.txt",
        "iqaodi_dis_label": "./data/IQA_ODI/ref_dis_label.txt",
        "cviqd_dis_label": "./data/cviqd/CVIQD_dis_label.txt",
        "cviqd_ref_label": "./data/cviqd/CVIQD_ref_label.txt",
        "mvaqd_dis_label": "./data/MVAQD/MVAQD_dis_label.txt",

        # optimization
        "batch_size": 4,
        "num_workers": 8,
        "split_seed": 20,

        # model
        "viewport_num": 30,
        "viewport_size": (224, 224),
        "grid_size": (5, 6),
        "input_dim": 3,
        "embed_dim": 1024,
        "num_outputs": 1,
        "num_heads": 16,
        "depth": 4,
        
        # checkpoint
        "model_name": "cviqd-train-test-random-15",
        "pretrained_weight_path": "./output/models/vsm_mvaqd_s20/epoch256.pt",
        "log_path": "./output/log/temp/",
        "log_file": ".txt"
    })
    config.log_file = config.model_name + config.log_file

    set_logging(config)
    logging.info(config)

    if config.dataset_name == 'cviqd':
        from data.cviqd.cviqd_label import CVIQD
        train_name, val_name = split_dataset_cviqd(config.cviqd_ref_label, config.cviqd_dis_label, split_seed=config.split_seed)
        dis_train_path = config.cviqd_train_dis_path
        dis_val_path = config.cviqd_train_dis_path
        label_train_path = config.cviqd_dis_label
        label_val_path = config.cviqd_dis_label
        Dataset = CVIQD
    elif config.dataset_name == 'iqaodi':
        from data.IQA_ODI.iqa_odi_label import IQAODI
        train_name, val_name = split_dataset_iqaodi(config.iqaodi_dis_label, split_seed=config.split_seed)
        dis_train_path = config.iqaodi_train_dis_path
        dis_val_path = config.iqaodi_train_dis_path
        label_train_path = config.iqaodi_dis_label
        label_val_path = config.iqaodi_dis_label
        Dataset = IQAODI
    elif config.dataset_name == 'oiqa':
        from data.OIQA.oiqa_label import OIQA
        train_name, val_name = split_dataset_oiqa(config.oiqa_ref_label, config.oiqa_dis_label, split_seed=config.split_seed)
        dis_train_path = config.oiqa_train_dis_path
        dis_val_path = config.oiqa_train_dis_path
        label_train_path = config.oiqa_dis_label
        label_val_path = config.oiqa_dis_label
        Dataset = OIQA
    elif config.dataset_name == 'mvaqd':
        from data.MVAQD.mvaqd import MVAQD
        train_name, val_name = split_dataset_mvaqd(config.mvaqd_dis_label, split_seed=config.split_seed)
        dis_train_path = config.mvaqd_train_dis_path
        dis_val_path = config.mvaqd_train_dis_path
        label_train_path = config.mvaqd_dis_label
        label_val_path = config.mvaqd_dis_label
        Dataset = MVAQD
    else:
        raise ValueError("No dataset, you need to add this new dataset.")
    
    print(val_name)
    # val_name = train_name + val_name
    # data load
    val_dataset = Dataset(
        dis_path=dis_val_path,
        txt_file_name=label_val_path,
        list_name=val_name,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
        viewport_size=config.viewport_size,
        viewport_num=config.viewport_num,
        grid_size=config.grid_size,
        data_type='valid'
    )

    logging.info('number of val scenes: {}'.format(len(val_dataset)))
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )

    net = creat_model(config=config, model_weight_path=config.pretrained_weight_path, pretrained=True)
    net = nn.DataParallel(net).cuda()
    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        losses = []
        net.eval()
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(val_loader):
            pred = 0
            d = data['d_img_org'].cuda()
            labels = data['score']
            name = data['name']
            table = data['table'].cuda()
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
            pred_d = net(d, table)

            # compute loss
            loss = criterion(torch.squeeze(pred_d), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred_d.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rmse = np.sqrt(mean_squared_error(np.squeeze(labels_epoch), np.squeeze(pred_epoch)))

        logging.info('loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== RMSE:{:.4}'.format(np.mean(losses), rho_s, rho_p, rmse))