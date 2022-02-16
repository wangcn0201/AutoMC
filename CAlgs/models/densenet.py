import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


class BasicBlock(nn.Module):
    def __init__(self, cfg, dropRate, activation='relu', numBins=8, special=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(cfg[0], cfg[1], kernel_size=3, padding=1, bias=False, special=special)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.act1 = select_act(activation, numBins)
        self.dropout = nn.Dropout(p=dropRate)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, cfg, activation='relu', numBins=8, special=None):
        super(Transition, self).__init__()
        self.conv1 = conv(cfg[0], cfg[1], kernel_size=1, bias=False, special=special)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.act1 = select_act(activation, numBins)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):

    def __init__(self, cfg, n, dropRate=0.3, num_classes=10, activation='relu', numBins=8, special=None):
        super(DenseNet, self).__init__()

        self.cfg = cfg
        self.n = n
        self.args = {
            'num_classes': num_classes,
            'activation': activation,
            'numBins': numBins,
            'special': special
        }

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.conv1 = nn.Conv2d(3, cfg[0][0], kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cfg[0][0])
        self.dense1 = self._make_denseblock(n, {'last_output': cfg[0][0], 'cfg': cfg[1]}, dropRate, activation=activation, numBins=numBins, special=special)
        self.trans1 = self._make_transition({'last_output': calc_sum(cfg, 2), 'cfg': cfg[2]}, activation=activation, numBins=numBins, special=special)
        self.dense2 = self._make_denseblock(n, {'last_output': cfg[2][0], 'cfg': cfg[3]}, dropRate, activation=activation, numBins=numBins, special=special)
        self.trans2 = self._make_transition({'last_output': calc_sum(cfg, 4), 'cfg': cfg[4]}, activation=activation, numBins=numBins, special=special)
        self.dense3 = self._make_denseblock(n, {'last_output': cfg[4][0], 'cfg': cfg[5]}, dropRate, activation=activation, numBins=numBins, special=special)
        self.act = select_act(activation, numBins)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(calc_sum(cfg, 6), num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_denseblock(self, num_blocks, new_cfg, dropRate, activation='relu', numBins=8, special=None):
        layers = []
        channel_num = new_cfg['last_output']
        for i in range(num_blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(BasicBlock([channel_num, new_cfg['cfg'][i]], dropRate, activation=activation, numBins=numBins, special=special))
            channel_num += new_cfg['cfg'][i]

        return nn.Sequential(*layers)

    def _make_transition(self, new_cfg, activation='relu', numBins=8, special=None):
        return Transition([new_cfg['last_output'], new_cfg['cfg'][0]], activation=activation, numBins=numBins, special=special)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        x = self.act(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def calc_sum(cfg, pos):
    res = 0
    for i in range(pos - 2, pos):
        for j in range(len(cfg[i])):
            res += cfg[i][j]
    return res

def get_cfg(n, growthRate, cfg, rate):
    if cfg == None:
        growthRate = int(growthRate * rate)
        cfg = [[int(growthRate * 2)], [growthRate] * n, [growthRate * (n + 2)],
               [growthRate] * n, [growthRate * (n + n + 2)], [growthRate] * n]
        return cfg
    elif isinstance(cfg[0], list):
        cfg = extend_cfg(cfg)

    for i in range(len(cfg)):
        cfg[i] = int(cfg[i] * rate)
    p, res = 1, [[cfg[0]]]
    for i in range(3):
        now = []
        for j in range(n):
            now.append(cfg[p])
            p += 1
        res.append(now)
        if i < 2:
            res.append([cfg[p]])
            p += 1
    cfg = res
    return cfg


def densenet16(growthRate=12, num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''
    growthRate is disabled if cfg is not None.
    '''
    n = (16 - 4) // 3
    cfg = get_cfg(n, growthRate, cfg, rate)
    model = DenseNet(cfg, n, num_classes=num_classes, activation=activation, numBins=numBins, special=special)
    return model

def densenet40(growthRate=12, num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''
    growthRate is disabled if cfg is not None.
    '''
    n = (40 - 4) // 3
    cfg = get_cfg(n, growthRate, cfg, rate)
    model = DenseNet(cfg, n, num_classes=num_classes, activation=activation, numBins=numBins, special=special)
    return model

if __name__ == '__main__':
    model = DenseNet()
    print(model)
