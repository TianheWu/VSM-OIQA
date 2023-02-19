import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from torchvision import transforms
from torch.utils.data import DataLoader
from models.oiqa_model import creat_model
from config import Config
from utils_tools.process import ToTensor, RandHorizontalFlip, Normalize
from utils_tools.process import split_dataset_cviqd, split_dataset_iqaodi, split_dataset_oiqa, split_dataset_mvaqd
from utils_tools.process import split_dataset_ke2020, split_dataset_Fang2022

from torch.utils.tensorboard import SummaryWriter 
from load_train import train_oiqa, eval_oiqa


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
        "mvaqd_train_dis_path": "/mnt/cpath2/lf2/OIQA_dataset/MVAQD/MVAQD-dataset/",

        "ke2020_dataset_path": "/mnt/data_16TB/wth22/IQA_dataset/Ke2020_Dis/",
        "ke2020_user_data_path": "/mnt/data_16TB/wth22/IQA_dataset/ke2020_userData/",
        "ke2020_label_path": "/home/wth22/OIQA/data/ke2020_data/ke2020_label.xlsx",

        "fang2022_dataset_path": "/mnt/data_16TB/wth22/IQA_dataset/Fang2022_dis/",
        "fang2022_user_data_path": "/mnt/data_16TB/wth22/IQA_dataset/Fang2022_userData_bad/",
        "fang2022_label_path": "./data/fang2022_data/fang2022_label.xls",

        "oiqa_dis_label": "./data/OIQA/OIQA_dis_label.txt",
        "oiqa_ref_label": "./data/OIQA/OIQA_ref_label.txt",
        "iqaodi_dis_label": "./data/IQA_ODI/ref_dis_label.txt",
        "cviqd_dis_label": "./data/cviqd/CVIQD_dis_label.txt",
        "cviqd_ref_label": "./data/cviqd/CVIQD_ref_label.txt",
        "mvaqd_dis_label": "./data/MVAQD/MVAQD_dis_label.txt",

        # optimization
        "batch_size": 4,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 500,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_workers": 8,
        "split_seed": 0,

        # model
        "embed_dim": 1024,
        "seq_layers": 3,
        "conv_input_dim": 128,
        "viewport_nums": 6,
        "seq_dim": 256,
        "seq_hidden_dim":256,

        # data
        "decision_method": "ent",
        "start_points": [[-90, 0], [0, 0], [90, 0]],
        "viewport_size": (224, 224),
        "fov": [90, 90],
        "pretrained": False,
        "pretrained_weight_path": "./output/models/pretrained_2diqa_kadid10k/epoch77.pt",

        # load & save checkpoint
        "model_name": "VSM_s0",
        "snap_path": "./output/models/Refine/",               # directory for saving checkpoint
        "log_path": "./output/log/Refine/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/"
    })

    if not os.path.exists(config.snap_path):
        os.makedirs(config.snap_path)

    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

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
    elif config.dataset_name == 'ke2020':
        from data.ke2020_data.ke2020 import Ke2020
        train_name, val_name = split_dataset_ke2020(config.ke2020_dataset_path, split_seed=config.split_seed)
        dis_train_path = config.ke2020_dataset_path
        dis_val_path = config.ke2020_dataset_path
        label_train_path = config.ke2020_label_path
        label_val_path = config.ke2020_label_path
        user_data_path = config.ke2020_user_data_path
        Dataset = Ke2020
    elif config.dataset_name == 'fang2022':
        from data.fang2022_data.fang2022 import Fang2022
        train_name, val_name = split_dataset_Fang2022(config.fang2022_dataset_path, split_seed=config.split_seed)
        dis_train_path = config.fang2022_dataset_path
        dis_val_path = config.fang2022_dataset_path
        label_train_path = config.fang2022_label_path
        label_val_path = config.fang2022_label_path
        user_data_path = config.fang2022_user_data_path
        Dataset = Fang2022
    else:
        raise ValueError("No dataset, you need to add this new dataset.")


    if config.dataset_name != 'fang2022' or 'ke2020':
        # data load
        train_dataset = Dataset(
            dis_path=dis_train_path,
            txt_file_name=label_train_path,
            list_name=train_name,
            transform=transforms.Compose([Normalize(0.5, 0.5), RandHorizontalFlip(), ToTensor()]),
            viewport_size=config.viewport_size,
            viewport_nums=config.viewport_nums,
            decision_method=config.decision_method,
            fov=config.fov,
            start_points=config.start_points
        )
        val_dataset = Dataset(
            dis_path=dis_val_path,
            txt_file_name=label_val_path,
            list_name=val_name,
            transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            viewport_size=config.viewport_size,
            viewport_nums=config.viewport_nums,
            decision_method=config.decision_method,
            fov=config.fov,
            start_points=config.start_points
        )
    elif config.dataset_name == 'fang2022':
        train_dataset = Dataset(
            dis_path=dis_train_path,
            label_path=label_train_path,
            user_data_path=user_data_path,
            list_name=train_name,
            transform=transforms.Compose([Normalize(0.5, 0.5), RandHorizontalFlip(), ToTensor()]),
            viewport_size=config.viewport_size,
            viewport_nums=config.viewport_nums,
            grid_size=config.grid_size,
            decision_method=config.decision_method,
            fov=config.fov
        )
        val_dataset = Dataset(
            dis_path=dis_val_path,
            label_path=label_val_path,
            user_data_path=user_data_path,
            list_name=val_name,
            transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
            viewport_size=config.viewport_size,
            viewport_nums=config.viewport_nums,
            grid_size=config.grid_size,
            decision_method=config.decision_method,
            fov=config.fov
        )


    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))
    logging.info('train : val ratio is: {:.4}'.format(len(train_dataset) / len(val_dataset)))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )

    net = creat_model(config=config, model_weight_path=config.pretrained_weight_path, pretrained=config.pretrained)
    net = nn.DataParallel(net).cuda()

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0
    best_rmse = 999
    for epoch in range(0, config.n_epoch):
        # visual(net, val_loader)
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p, rmse = train_oiqa(epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)
        writer.add_scalar("RMSE", rmse, epoch)

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running val {} in epoch {}'.format(config.dataset_name, epoch + 1))
            loss, rho_s, rho_p, rmse = eval_oiqa(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

            if rho_s + rho_p > main_score:
                main_score = rho_s + rho_p
                logging.info('======================================================================================')
                logging.info('============================== best main score is {} ================================='.format(main_score))
                logging.info('======================================================================================')
        
                best_srocc = rho_s
                best_plcc = rho_p
                best_rmse = rmse
                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.snap_path, model_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}, RMSE:{}'.format(epoch + 1, best_srocc, best_plcc, best_rmse))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))