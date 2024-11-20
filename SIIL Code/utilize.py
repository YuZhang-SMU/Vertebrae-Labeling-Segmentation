import torch
from math import ceil, floor
import numpy as np

def to_one_hot_tensor(seg, all_seg_labels):
    B, C, D, W, H = seg.shape
    result = torch.zeros(size=(B, len(all_seg_labels), D, W, H), device=seg.device).to(torch.float32)
    for i, l in enumerate(all_seg_labels):
        result[:, i, :, :, :][seg.squeeze(1) == l] = 1
    return result

def print_object(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

def char_color(s,front=50,word=32):
    """
    # 改变字符串颜色的函数
    :param s:
    :param front:
    :param word:
    :return:
    """
    new_char = "\033[0;"+str(int(word))+";"+str(int(front))+"m"+s+"\033[0m"
    return new_char