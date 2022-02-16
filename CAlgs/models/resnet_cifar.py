import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import *



class BasicBlock(nn.Module):

    def __init__(self, cfg, first, stride=1, activation='relu', numBins=8, special=None):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.act1 = select_act(activation, numBins)
        self.conv1 = conv(cfg[0], cfg[1], 3, stride=stride, padding=1, bias=False, special=special)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.act2 = select_act(activation, numBins)
        self.conv2 = conv(cfg[1], cfg[2], 3, stride=1, padding=1, bias=False, special=special)

        if first:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(cfg[0]),
                nn.Conv2d(cfg[0], cfg[2], kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(cfg[0]),
                nn.Conv2d(cfg[0], cfg[2], kernel_size=1, stride=1, bias=False)
            )

    def forward(self, x):
        out = self.conv1(self.act1(self.bn1(x)))
        out = self.conv2(self.act2(self.bn2(out)))
        out += self.shortcut(x)
        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, n, num_classes, cfg, activation='relu', numBins=8, special=None):
        super(ResNet_Cifar, self).__init__()
        self.cfg = cfg
        self.args = {
            'num_classes': num_classes,
            'activation': activation,
            'numBins': numBins,
            'special': special
            }
        
        self.conv1 = conv(
            3, cfg[0][0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, n, {'last_output':cfg[0][0], 'cfg':cfg[1]}, activation=activation, numBins=numBins, special=special)
        self.layer2 = self._make_layer(block, n, {'last_output':cfg[1][-1][-1], 'cfg':cfg[2]}, stride=2, activation=activation, numBins=numBins, special=special)
        self.layer3 = self._make_layer(block, n, {'last_output':cfg[2][-1][-1], 'cfg':cfg[3]}, stride=2, activation=activation, numBins=numBins, special=special)
        self.bn = nn.BatchNorm2d(cfg[3][-1][-1])
        self.act = select_act(activation, numBins)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(cfg[3][-1][-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, blocks, new_cfg, stride=1, activation='relu', numBins=8, special=None):
        layers = []
        layers.append(block([new_cfg['last_output']] + new_cfg['cfg'][0], True, stride, activation=activation, numBins=numBins, special=special))
        for i in range(1, blocks):
            layers.append(block([new_cfg['cfg'][i - 1][-1]] + new_cfg['cfg'][i], False, activation=activation, numBins=numBins, special=special))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_cfg(block, n, cfg, rate):
    if cfg == None:
        new_16 = int(16 * rate)
        new_32 = int(32 * rate)
        new_64 = int(64 * rate)
        new_128 = int(128 * rate)
        new_256 = int(256 * rate)
        if block == BasicBlock:
            cfg = [[new_16], [[new_16, new_16]] * n, [[new_32, new_32]] * n, [[new_64, new_64]] * n]
        else:
            cfg = [[new_16], [[new_16, new_16, new_64]] * n, [[new_32, new_32, new_128]] * n, [[new_64, new_64, new_256]] * n]
        return cfg
    elif isinstance(cfg[0], list):
        cfg = extend_cfg(cfg)

    for i in range(len(cfg)):
        cfg[i] = int(cfg[i] * rate)
    p, res = 1, [[cfg[0]]]
    num = 2 if block == BasicBlock else 3
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


def resnet20(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(BasicBlock, 3, cfg, rate)
    model = ResNet_Cifar(BasicBlock, 3, num_classes, cfg, activation, numBins, special)
    return model


def resnet32(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(BasicBlock, 5, cfg, rate)
    model = ResNet_Cifar(BasicBlock, 5, num_classes, cfg, activation, numBins, special)    
    return model


def resnet44(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(BasicBlock, 7, cfg, rate)
    model = ResNet_Cifar(BasicBlock, 7, num_classes, cfg, activation, numBins, special)   
    return model


def resnet56(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(BasicBlock, 9, cfg, rate)
    model = ResNet_Cifar(BasicBlock, 9, num_classes, cfg, activation, numBins, special)    
    return model


def resnet110(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(BasicBlock, 18, cfg, rate)
    model = ResNet_Cifar(BasicBlock, 18, num_classes, cfg, activation, numBins, special)
    return model

def resnet164(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(BasicBlock, 27, cfg, rate)
    model = ResNet_Cifar(BasicBlock, 27, num_classes, cfg, activation, numBins, special)
    return model