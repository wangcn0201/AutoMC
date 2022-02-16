import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LMA(nn.Module):
    def __init__(self, mu=0.99, num_bins=8):
        super(LMA, self).__init__()
        self.mu = mu
        self.num_bins = num_bins
        self.mv_mean = torch.tensor(0.0)#.to(device)
        self.mv_std = torch.tensor(0.0)#.to(device)
        self.init_alpha = torch.tensor([0.0 for k in range(num_bins//2)]+[1.0 for k in range(num_bins - num_bins//2)]).cuda()
        self.alphas = Parameter(torch.tensor([0.0 for k in range(num_bins)]))
        self.betas = Parameter(torch.tensor([0.0 for k in range(num_bins)]))
                
    def forward(self, x):
        mean = x.mean().detach()
        std = x.std().detach()
        new_mean = self.mv_mean * self.mu + mean * (1-self.mu)
        new_std = self.mv_std * self.mu + std * (1-self.mu)
        self.mv_mean = new_mean.clone()
        self.mv_std = new_std.clone()
        if not self.training:
            mean = self.mv_mean.detach()
            std = self.mv_std.detach()
        step = 6*std / self.num_bins
        idx = torch.clamp(((x-mean+3*std)/step).long(), 0, self.num_bins-1).view(-1)
        alphas = torch.index_select(self.alphas + self.init_alpha, dim=0, index=idx).view(x.size())
        betas = torch.index_select(self.betas, dim=0, index=idx).view(x.size())
        return alphas * x + betas

