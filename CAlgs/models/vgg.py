import torch.nn as nn
import math
from .utils import *


class VGG(nn.Module):

    def __init__(self, cfg, num_classes, activation='relu', numBins=8, special=None):
        super(VGG, self).__init__()
        self.cfg = cfg
        self.args = {
            'num_classes': num_classes,
            'activation': activation,
            'numBins': numBins,
            'special': special
            }

        self.features = self.make_layers(cfg, select_act(activation, numBins), special)
        self.classifier = nn.Linear(cfg[-2], num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def make_layers(self, cfg, activation, special):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = conv(in_channels, v, kernel_size=3, padding=1, special=special)
                layers += [conv2d, nn.BatchNorm2d(v), activation]
                in_channels = v
        return nn.Sequential(*layers)


cfgs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def get_vgg_info(arch_name):
    if '11' in arch_name:
        return cfgs['11']
    elif '13' in arch_name:
        return cfgs['13']
    elif '16' in arch_name:
        return cfgs['16']
    elif '19' in arch_name:
        return cfgs['19']


def get_cfg(num, cfg, rate):
    now_cfg, index = cfg, 0
    cfg = cfgs[num]
    if now_cfg == None:
        now_cfg = cfg
    for i in range(len(cfg)):
        if isinstance(cfg[i], int):
            cfg[i] = int(now_cfg[index] * rate)
            index += 1
        elif 'M' in now_cfg:
            index += 1
    return cfg


def vgg11(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''VGG 11-layer model with batch normalization'''
    cfg = get_cfg('11', cfg, rate)
    model = VGG(cfg, num_classes, activation=activation, numBins=numBins, special=special)
    return model


def vgg13(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''VGG 13-layer model with batch normalization'''
    cfg = get_cfg('13', cfg, rate)
    model = VGG(cfg, num_classes, activation=activation, numBins=numBins, special=special)
    return model


def vgg16(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''VGG 16-layer model with batch normalization'''
    cfg = get_cfg('16', cfg, rate)
    model = VGG(cfg, num_classes, activation=activation, numBins=numBins, special=special)
    return model


def vgg19(num_classes=10, cfg=None, rate=1, activation='relu', numBins=8, special=None):
    '''VGG 19-layer model with batch normalization'''
    cfg = get_cfg('19', cfg, rate)
    model = VGG(cfg, num_classes, activation=activation, numBins=numBins, special=special)
    return model
