import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np
from .utils import *


class wide_basic(nn.Module):
    def __init__(self, cfg, dropout_rate, first, stride=1, activation='relu', numBins=8, special=None):
        super(wide_basic, self).__init__()

        assert(len(cfg) == 3)

        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.act1 = select_act(activation, numBins)
        self.conv1 = conv(cfg[0], cfg[1], kernel_size=3, padding=1, bias=True, special=special)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.act2 = select_act(activation, numBins)
        self.conv2 = conv(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=True, special=special)

        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cfg[0], cfg[2], kernel_size=1, stride=stride, bias=True),
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cfg[0], cfg[2], kernel_size=1, stride=1, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.act1(self.bn1(x))))
        out = self.conv2(self.act2(self.bn2(out)))
        out += self.shortcut(x)
        return out


class Wide_ResNet(nn.Module):
    def __init__(self, cfg, n, dropout_rate=0.3, num_classes=10, activation='relu', numBins=8, special=None):
        super(Wide_ResNet, self).__init__()
        self.cfg = cfg
        self.n = n
        self.args = {
            'num_classes': num_classes,
            'activation': activation,
            'numBins': numBins,
            'special': special
        }

        self.conv1 = conv(3, cfg[0][0], kernel_size=3)
        self.layer1 = self._wide_layer({'last_output': cfg[0][0], 'cfg': cfg[1]}, n, dropout_rate, stride=1, activation=activation, numBins=numBins, special=special)
        self.layer2 = self._wide_layer({'last_output': cfg[1][-1][-1], 'cfg': cfg[2]}, n, dropout_rate, stride=2, activation=activation, numBins=numBins, special=special)
        self.layer3 = self._wide_layer({'last_output': cfg[2][-1][-1], 'cfg': cfg[3]}, n, dropout_rate, stride=2, activation=activation, numBins=numBins, special=special)
        self.bn1 = nn.BatchNorm2d(cfg[3][-1][-1], momentum=0.9)
        self.act = select_act(activation, numBins)
        self.linear = nn.Linear(cfg[3][-1][-1], num_classes)

    def _wide_layer(self, new_cfg, num_blocks, dropout_rate, stride, activation='relu', numBins=8, special=None):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        layers.append(wide_basic([new_cfg['last_output']] + new_cfg['cfg'][0], dropout_rate, True, stride, activation, numBins, special))
        for i in range(1, len(strides)):
            layers.append(wide_basic([new_cfg['cfg'][i - 1][-1]] + new_cfg['cfg'][i], dropout_rate, False, strides[i], activation, numBins, special))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.act(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def get_cfg(n, widen_factor, cfg, rate):
    if cfg == None:
        new_16 = int(16 * widen_factor * rate)
        new_32 = int(32 * widen_factor * rate)
        new_64 = int(64 * widen_factor * rate)
        cfg = [[int(16 * rate)], [[new_16, new_16]] * n,
               [[new_32, new_32]] * n, [[new_64, new_64]] * n]
        return cfg
    elif isinstance(cfg[0], list):
        cfg = extend_cfg(cfg)

    for i in range(len(cfg)):
        cfg[i] = int(cfg[i] * rate)
    p, res = 1, [[cfg[0]]]
    num = 2
    for i in range(3):
        now = []
        for j in range(n):
            _ = []
            for k in range(num):
                _.append(cfg[p])
                p += 1
            now.append(_)
        res.append(now)
    cfg = res
    return cfg


def wide_resnet16(widen_factor=1, num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''
    widen_factor is disabled if cfg is not None.
    '''
    n = (16 - 4) // 6
    cfg = get_cfg(n, widen_factor, cfg, rate)
    model = Wide_ResNet(cfg, n, num_classes=num_classes, activation=activation, numBins=numBins, special=special)
    return model


def wide_resnet22(widen_factor=1, num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''
    widen_factor is disabled if cfg is not None.
    '''
    n = (22 - 4) // 6
    cfg = get_cfg(n, widen_factor, cfg, rate)
    model = Wide_ResNet(cfg, n, num_classes=num_classes, activation=activation, numBins=numBins, special=special)
    return model


def wide_resnet28(widen_factor=1, num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''
    widen_factor is disabled if cfg is not None.
    '''
    n = (28 - 4) // 6
    cfg = get_cfg(n, widen_factor, cfg, rate)
    model = Wide_ResNet(cfg, n, num_classes=num_classes, activation=activation, numBins=numBins, special=special)
    return model

def wide_resnet40(widen_factor=1, num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''
    widen_factor is disabled if cfg is not None.
    '''
    n = (40 - 4) // 6
    cfg = get_cfg(n, widen_factor, cfg, rate)
    model = Wide_ResNet(cfg, n, num_classes=num_classes, activation=activation, numBins=numBins, special=special)
    return model