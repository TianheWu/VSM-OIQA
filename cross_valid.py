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
from utils.process import ToTensor, RandHorizontalFlip, Normalize
from utils.process import split_dataset_cviqd, split_dataset_iqaodi, split_dataset_oiqa, split_dataset_mvaqd

from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
from load_train import train_oiqa, eval_oiqa


os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'


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

def visual(net, visual_loader):
    for data in tqdm(visual_loader):
        x_d = data['d_img_org'].cuda()
        name = data['name']
        pred_d = net(x_d, name)


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
        "train_dataset_name": "mvaqd",
        "val_dataset_name": "cviqd",
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
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 500,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_workers": 8,
        "split_seed": 8,

        # model
        "decision_method": "ent",
        "viewport_num": 30,
        "viewport_size": (224, 224),
        "grid_size": (5, 6),
        "input_dim": 3,
        "embed_dim": 1024,
        "num_outputs": 1,
        "num_heads": 16,
        "depth": 4,
        "pretrained": False,
        "pretrained_weight_path": "./output/models/pretrained_2diqa_kadid10k/epoch77.pt",

        # load & save checkpoint
        "model_name": "mvaqd_cviqd_all",
        "output_path": "./output",
        "snap_path": "./output/models/",               # directory for saving checkpoint
        "log_path": "./output/log/Cross-Validation/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard/"
    })

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)
    logging.info('Train Dataset [{}], Val Dataset [{}]'.format(config.train_dataset_name, config.val_dataset_name))


    if config.train_dataset_name == 'cviqd' or config.val_dataset_name == 'cviqd':
        from data.cviqd.cviqd_label import CVIQD
        if config.train_dataset_name == 'cviqd':
            train_name1, val_name1 = split_dataset_cviqd(config.cviqd_ref_label, config.cviqd_dis_label, split_seed=config.split_seed)
            dis_train_path1 = config.cviqd_train_dis_path
            dis_val_path1 = config.cviqd_train_dis_path
            label_train_path1 = config.cviqd_dis_label
            label_val_path1 = config.cviqd_dis_label
            Dataset_train = CVIQD

        if config.val_dataset_name == 'cviqd':
            train_name2, val_name2 = split_dataset_cviqd(config.cviqd_ref_label, config.cviqd_dis_label, split_seed=config.split_seed)
            dis_train_path2 = config.cviqd_train_dis_path
            dis_val_path2 = config.cviqd_train_dis_path
            label_train_path2 = config.cviqd_dis_label
            label_val_path2 = config.cviqd_dis_label
            Dataset_val = CVIQD

    if config.train_dataset_name == 'iqaodi' or config.val_dataset_name == 'iqaodi':
        from data.IQA_ODI.iqa_odi_label import IQAODI
        if config.train_dataset_name == 'iqaodi':
            train_name1, val_name1 = split_dataset_iqaodi(config.iqaodi_dis_label, split_seed=config.split_seed)
            dis_train_path1 = config.iqaodi_train_dis_path
            dis_val_path1 = config.iqaodi_train_dis_path
            label_train_path1 = config.iqaodi_dis_label
            label_val_path1 = config.iqaodi_dis_label
            Dataset_train = IQAODI

        if config.val_dataset_name == 'iqaodi':
            train_name2, val_name2 = split_dataset_iqaodi(config.iqaodi_dis_label, split_seed=config.split_seed)
            dis_train_path2 = config.iqaodi_train_dis_path
            dis_val_path2 = config.iqaodi_train_dis_path
            label_train_path2 = config.iqaodi_dis_label
            label_val_path2 = config.iqaodi_dis_label
            Dataset_val = IQAODI

    if config.train_dataset_name == 'oiqa' or config.val_dataset_name == 'oiqa':
        from data.OIQA.oiqa_label import OIQA
        if config.train_dataset_name == 'oiqa':
            train_name1, val_name1 = split_dataset_oiqa(config.oiqa_ref_label, config.oiqa_dis_label, split_seed=config.split_seed)
            dis_train_path1 = config.oiqa_train_dis_path
            dis_val_path1 = config.oiqa_train_dis_path
            label_train_path1 = config.oiqa_dis_label
            label_val_path1 = config.oiqa_dis_label
            Dataset_train = OIQA

        if config.val_dataset_name == 'oiqa':
            train_name2, val_name2 = split_dataset_oiqa(config.oiqa_ref_label, config.oiqa_dis_label, split_seed=config.split_seed)
            dis_train_path2 = config.oiqa_train_dis_path
            dis_val_path2 = config.oiqa_train_dis_path
            label_train_path2 = config.oiqa_dis_label
            label_val_path2 = config.oiqa_dis_label
            Dataset_val = OIQA

    if config.train_dataset_name == 'mvaqd' or config.val_dataset_name == 'mvaqd':
        from data.MVAQD.mvaqd import MVAQD
        if config.train_dataset_name == 'mvaqd':
            train_name1, val_name1 = split_dataset_mvaqd(config.mvaqd_dis_label, split_seed=config.split_seed)
            dis_train_path1 = config.mvaqd_train_dis_path
            dis_val_path1 = config.mvaqd_train_dis_path
            label_train_path1 = config.mvaqd_dis_label
            label_val_path1 = config.mvaqd_dis_label
            Dataset_train = MVAQD

        if config.val_dataset_name == 'mvaqd':
            train_name2, val_name2 = split_dataset_mvaqd(config.mvaqd_dis_label, split_seed=config.split_seed)
            dis_train_path2 = config.mvaqd_train_dis_path
            dis_val_path2 = config.mvaqd_train_dis_path
            label_train_path2 = config.mvaqd_dis_label
            label_val_path2 = config.mvaqd_dis_label
            Dataset_val = MVAQD

    # data load
    train_dataset = Dataset_train(
        dis_path=dis_train_path1,
        txt_file_name=label_train_path1,
        list_name=train_name1,
        transform=transforms.Compose([Normalize(0.5, 0.5), RandHorizontalFlip(), ToTensor()]),
        viewport_size=config.viewport_size,
        viewport_num=config.viewport_num,
        grid_size=config.grid_size,
        decision_method=config.decision_method
    )
    val_name2 = train_name2 + val_name2
    val_dataset = Dataset_val(
        dis_path=dis_val_path2,
        txt_file_name=label_val_path2,
        list_name=val_name2,
        transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
        viewport_size=config.viewport_size,
        viewport_num=config.viewport_num,
        grid_size=config.grid_size,
        decision_method=config.decision_method
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

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
            logging.info('Running val {} in epoch {}'.format(config.val_dataset_name, epoch + 1))
            loss, rho_s, rho_p, rmse = eval_oiqa(config, epoch, net, criterion, val_loader)
            logging.info('Eval done...')

            if abs(rho_s) + abs(rho_p) > main_score:
                main_score = abs(rho_s) + abs(rho_p)
                logging.info('======================================================================================')
                logging.info('============================== best main score is {} ================================='.format(main_score))
                logging.info('======================================================================================')
        
                best_srocc = abs(rho_s)
                best_plcc = abs(rho_p)
                best_rmse = rmse
                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.snap_path, model_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}, RMSE:{}'.format(epoch + 1, best_srocc, best_plcc, best_rmse))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))