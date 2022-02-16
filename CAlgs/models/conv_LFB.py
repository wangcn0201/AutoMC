import torch
import torch.nn as nn
import torch.nn.functional as F


class conv(nn.Module):
    def __init__(self, in_channels, kernel_size, group=1, times=1, stride=1, bias=True):
        super(conv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.group = group
        self.n_basis = int(in_channels // group * times)
        self.basis_size = in_channels // group
        self.basis_weight = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(self.n_basis, self.basis_size, 3, 3)))
        self.basis_bias = nn.Parameter(torch.zeros(self.n_basis)) if bias else None

    def forward(self, x):
        if self.group == 1:
            x = F.conv2d(input=x, weight=self.basis_weight, bias=self.basis_bias,
                         stride=self.stride, padding=self.kernel_size//2)
        else:
            x = torch.cat([F.conv2d(input=xi, weight=self.basis_weight, bias=self.basis_bias, stride=self.stride, padding=self.kernel_size//2) for xi in torch.split(x, self.basis_size, dim=1)], dim=1)
        return x

    def __repr__(self):
        s = 'Conv(in_channels={}, basis_size={}, group={}, n_basis={}, kernel_size={}, out_channel={})'.format(
            self.in_channels, self.basis_size, self.group, self.n_basis, self.kernel_size, self.group * self.n_basis)
        return s


class conv_LFB(nn.Module):
    def __init__(self, in_channels, out_channels, group, times, stride=1, bias=True):
        super(conv_LFB, self).__init__()
        assert in_channels % group == 0, "in_channels {} must be able to divide group {}.".format(in_channels, group)
        modules = [conv(in_channels, kernel_size=3, group=group, times=times, stride=stride, bias=bias)]
        modules.append(nn.Conv2d(int(times * in_channels // group * group), out_channels, kernel_size=1, stride=1, bias=bias))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)
