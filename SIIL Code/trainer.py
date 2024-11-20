import datetime

import numpy as np
import torch

import os

from torch import optim

from get_data import get_data_train
from utilize import print_object, char_color

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings("ignore")

from get_config import MyConfig
from models.siil import SIIL

def adjust_learning_rate(config, optimizer, epoch):
    lr = config.IR * (config.IR_DEAY ** (epoch // config.IR_DEAY_EPOCH))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    myconfig = MyConfig()
    print_object(myconfig)
    print('--------------  Load_Model  --------------')

    # network
    mynetwork = SIIL(myconfig).to(my_device)
    hist_loss = np.zeros(shape=(myconfig.EPOCH))

    # data
    train_loader = get_data_train(myconfig)

    # train
    scalar = torch.cuda.amp.GradScaler()
    optimizer = optim.AdamW(mynetwork.parameters(), lr=myconfig.IR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    for epoch in range(myconfig.EPOCH):
        print(char_color('Epoch:{}/{}'.format(epoch, myconfig.EPOCH)))

        tic = datetime.datetime.now()
        mynetwork.train()
        epoch_loss = 0
        for idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if epoch < mynetwork.frozen_epoch:
                    for p in mynetwork.SIIL_core.parameters(): p.requires_grad = False
                    for p in mynetwork.DE.parameters(): p.requires_grad = False
                output_dict = mynetwork(data, epoch, idx)
                scalar.scale(output_dict['final_loss']).backward()
                scalar.step(optimizer)
                scalar.update()
            epoch_loss += output_dict['final_loss'].item()
        epoch_loss = float(epoch_loss / len(train_loader))
        adjust_learning_rate(myconfig, optimizer, epoch)

        toc = datetime.datetime.now()
        h, remainder = divmod((toc - tic).seconds, 3600)
        m, s = divmod(remainder, 60)
        print("per epoch cost Time %02d h:%02d m:%02d s" % (h, m, s))

        hist_loss[epoch] = epoch_loss
        torch.save(mynetwork, myconfig.latest_model)
