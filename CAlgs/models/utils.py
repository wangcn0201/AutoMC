import torch.nn as nn
from models.activations import LMA
from .conv_LFB import conv_LFB
from .utils import *


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, special=None):
    if special == None:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    elif special[0] == 'LFB':
        group, times = special[1:]
        return conv_LFB(in_channels, out_channels, group, times, stride=stride, bias=True)


def select_act(activation, num_bins=None):
    if activation == 'relu':
        act = nn.ReLU(inplace=True)
    elif activation == 'lma':
        act = LMA(num_bins=num_bins)
    return act

def extend_cfg(cfg):
    tmp = []
    def f(l, level=0):
        for item in l:
            if isinstance(item, list):
                f(item, level + 1)
            else:
                tmp.append(item)
    f(cfg)
    cfg = tmp
    return cfg
