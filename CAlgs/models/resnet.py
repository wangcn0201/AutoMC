import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


class BasicBlock(nn.Module):

    def __init__(self, cfg, first, stride=1, activation='relu', numBins=8, special=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(
            cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1, bias=False, special=special)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.act1 = select_act(activation, numBins)
        self.conv2 = conv(cfg[1], cfg[2], kernel_size=3, stride=1, padding=1, bias=False, special=special)
        self.bn2 = nn.BatchNorm2d(cfg[2])

        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cfg[0], cfg[2], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cfg[2])
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cfg[0], cfg[2], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(cfg[2])
            )
        self.act2 = select_act(activation, numBins)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, cfg, first, stride=1, activation='relu', numBins=8, special=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv(cfg[0], cfg[1], kernel_size=1, bias=False, special=special)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.act1 = select_act(activation, numBins)
        self.conv2 = conv(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False, special=special)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.act2 = select_act(activation, numBins)
        self.conv3 = conv(cfg[2], cfg[3], kernel_size=1, bias=False, special=special)
        self.bn3 = nn.BatchNorm2d(cfg[3])

        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cfg[0], cfg[3], kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cfg[3])
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cfg[0], cfg[3], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(cfg[3])
            )
        self.act3 = select_act(activation, numBins)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, cfg, activation='relu', numBins=8, special=None):
        super(ResNet, self).__init__()
        self.cfg = cfg
        self.args = {
            'num_classes': num_classes,
            'activation': activation,
            'numBins': numBins,
            'special': special
            }

        self.conv1 = conv(
            3, cfg[0][0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0][0])
        self.act = select_act(activation, numBins)
        self.layer1 = self._make_layer(block, {'last_output': cfg[0][0], 'cfg': cfg[1]}, num_blocks[0], stride=1, activation=activation, numBins=numBins, special=special)
        self.layer2 = self._make_layer(block, {'last_output': cfg[1][-1][-1], 'cfg': cfg[2]},
                                       num_blocks[1], stride=2, activation=activation, numBins=numBins, special=special)
        self.layer3 = self._make_layer(block, {'last_output': cfg[2][-1][-1], 'cfg': cfg[3]},
                                       num_blocks[2], stride=2, activation=activation, numBins=numBins, special=special)
        self.layer4 = self._make_layer(block, {'last_output': cfg[3][-1][-1], 'cfg': cfg[4]},
                                       num_blocks[3], stride=2, activation=activation, numBins=numBins, special=special)
        self.linear = nn.Linear(cfg[4][-1][-1], num_classes)

    def _make_layer(self, block, new_cfg, num_blocks, stride, activation='relu', numBins=8, special=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        layers.append(block([new_cfg['last_output']] + new_cfg['cfg'][0], True, stride, activation, numBins, special))
        for i in range(1, len(strides)):
            layers.append(block([new_cfg['cfg'][i - 1][-1]] + new_cfg['cfg'][i], False, strides[i], activation, numBins, special))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_cfg(block, num_blocks, cfg, rate):
    if cfg == None:
        new_64 = int(64 * rate)
        new_128 = int(128 * rate)
        new_256 = int(256 * rate)
        new_512 = int(512 * rate)
        new_1024 = int(1024 * rate)
        new_2048 = int(2048 * rate)
        if block == BasicBlock:
            cfg = [[new_64], [[new_64, new_64]] * num_blocks[0], [[new_128, new_128]] * num_blocks[1],
                   [[new_256, new_256]] * num_blocks[2], [[new_512, new_512]] * num_blocks[3]]
        else:
            cfg = [[new_64], [[new_64, new_64, new_256]] * num_blocks[0], [[new_128, new_128, new_512]] * num_blocks[1],
                   [[new_256, new_256, new_1024]] * num_blocks[2], [[new_512, new_512, new_2048]] * num_blocks[3]]
        return cfg
    elif isinstance(cfg[0], list):
        cfg = extend_cfg(cfg)

    for i in range(len(cfg)):
        cfg[i] = int(cfg[i] * rate)
    p, res = 1, [[cfg[0]]]
    num = 2 if block == BasicBlock else 3
    for i in range(4):
        now = []
        for j in range(num_blocks[i]):
            _ = []
            for k in range(num):
                _.append(cfg[p])
                p += 1
            now.append(_)
        res.append(now)
    cfg = res
    return cfg


def resnet18(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(BasicBlock, [2, 2, 2, 2], cfg, rate)
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                   num_classes, cfg, activation, numBins, special)
    return model


def resnet34(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(BasicBlock, [3, 4, 6, 3], cfg, rate)
    model = ResNet(BasicBlock, [3, 4, 6, 3],
                   num_classes, cfg, activation, numBins, special)
    return model


def resnet50(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(Bottleneck, [3, 4, 6, 3], cfg, rate)
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   num_classes, cfg, activation, numBins, special)
    return model


def resnet101(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(Bottleneck, [3, 4, 23, 3], cfg, rate)
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   num_classes, cfg, activation, numBins, special)
    return model


def resnet152(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    cfg = get_cfg(Bottleneck, [3, 8, 36, 3], cfg, rate)
    model = ResNet(Bottleneck, [3, 8, 36, 3],
                   num_classes, cfg, activation, numBins, special)
    return model
