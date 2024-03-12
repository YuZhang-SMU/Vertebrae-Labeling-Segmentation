import os

import torch

from get_config import MyConfig
from models.siil import SIIL

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    myconfig = MyConfig()

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIIL(myconfig).to(my_device)
    img = torch.randn(2, 1, 128, 64, 64).to(my_device)
    msk = torch.randn(2, 1, 128, 64, 64).to(my_device)
    print(model(img, msk))