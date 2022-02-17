from scipy import stats
import models
import torch
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
from .utils import *
from train import *
import matplotlib.pyplot as plt


def get_coef(T, modules, rate, strategy, logger):
    if logger:
        logger.info("getting coef")
        logger.info("coef target rate: " + str(rate))
    T_min, T_max = T
    T_minus = T_max - T_min
    now = 0.5
    coefs = []
    illegal = []
    while now <= 11:
        coefs.append(now)
        now += 0.0005
    if logger:
        logger.info("length of coefs: " + str(len(coefs)))
    sum, num, now_rate = [], [], []
    for i in range(len(coefs)):
        sum.append(0.0)
        num.append(0)
    with tqdm(total=len(modules)) as t:
        for index in range(len(modules)):
            if 'criterion' in modules[index][1]:
                m = modules[index][0]
                out_channels = m.weight.data.shape[0]
                weight_copy = m.weight.data.clone().cpu().numpy()
                a = weight_copy.flatten()
                if strategy == 'P1':
                    W, p = stats.shapiro(a)
                elif strategy == 'P2':
                    W, p = stats.jarque_bera(a)
                elif strategy == 'P3':
                    W = stats.kstat(a, 4) * stats.kstat(a, 3)
                for i in range(len(coefs)):
                    coef = coefs[i]
                    temp = (W - T_min) / (coef * T_minus)
                    prune_prob_stage = np.ceil(10 * temp) / 10
                    num_keep = int(out_channels * (1 - prune_prob_stage))
                    if prune_prob_stage >= 1 or prune_prob_stage < 0:
                        illegal.append(i)
                    sum[i] += num_keep
                    num[i] += out_channels
            t.update()
    now_rate = [sum[i] / num[i] for i in range(len(coefs))]
    illegal = list(set(illegal))

    coef_index = None
    for i in range(len(now_rate)):
        if i not in illegal:
            if (coef_index == None) or (abs(now_rate[coef_index] - rate) > abs(now_rate[i] - rate)):
                coef_index = i
    '''
    fig = plt.figure(figsize=(15,5))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.plot(coefs, now_rate)
    plt.legend()
    plt.show()
    '''
    return coefs[coef_index], now_rate[coef_index]


def get_T(modules, strategy):
    T_min, T_max = 2, -1
    for index in range(len(modules)):
        if 'criterion' in modules[index][1]:
            m = modules[index][0]
            a = m.weight.data.clone().cpu().numpy().flatten()

            if strategy == 'P1':
                W, p = stats.shapiro(a)
            elif strategy == 'P2':
                W, p = stats.jarque_bera(a)
            elif strategy == 'P3':
                W = stats.kstat(a, 4) * stats.kstat(a, 3)

            if W > T_max:
                T_max = W
            if W < T_min:
                T_min = W
    return T_min, T_max


def get_prune_prob_stage(T, a, strategy, coef):
    T_min, T_max = T
    T_minus = T_max - T_min
    if strategy == 'P1':
        W, p = stats.shapiro(a)
    elif strategy == 'P2':
        W, p = stats.jarque_bera(a)
    elif strategy == 'P3':
        W = stats.kstat(a, 4) * stats.kstat(a, 3)
    temp = (W - T_min) / (coef * T_minus)
    prune_prob_stage = np.ceil(10 * temp) / 10
    return prune_prob_stage


def get_arg_max(m, metric):
    if metric == 'l1norm':
        weight_copy = m.weight.data.abs().clone().cpu().numpy()
        filter_values = np.sum(weight_copy, axis=(1, 2, 3))
    else:
        weight_copy = m.weight.data.clone().cpu().numpy()
        filter_values = np.zeros(weight_copy.shape[0])

    for i in range(weight_copy.shape[0]):
        temp = weight_copy[i, :, :, :]
        if metric == 'k3':
            filter_values[i] = stats.kstat(temp, 3)
        elif metric == 'k4':
            filter_values[i] = stats.kstat(temp, 4)
        elif metric == 'k34':
            filter_values[i] = stats.kstat(
                temp, 3) * stats.kstat(temp, 4)
        else:
            temp = weight_copy[i, :, :, :].flatten()
            if metric == 'skew':
                filter_values[i] = stats.skew(temp)
            elif metric == 'kur':
                filter_values[i] = stats.kurtosis(temp)
            elif metric == 'skew_kur':
                filter_values[i] = stats.skew(
                    temp) * stats.kurtosis(temp)

    if metric == 'random':
        arg_max = list(range(weight_copy.shape[0]))
        np.random.shuffle(arg_max)
    else:
        arg_max = np.argsort(filter_values)
    return arg_max


def test_orig_for_step3(model, val_loader, dataset, logger):
    if logger:
        logger.info("testing original model")
    model.eval()

    output = np.zeros((50010, models.get_num_classes(dataset)))
    k = 0

    with torch.no_grad():
        for datat, target in val_loader:
            datat, target = datat.cuda(), target.cuda()
            datat, target = Variable(datat), Variable(target)
            b1 = model(datat)
            output[k:k+datat.shape[0], :] = b1.data.cpu().numpy()
            k = k+datat.shape[0]

    return output


def train_for_step3(model, optimizer, train_loader, output_orig, mse_factor, logger):
    model.train()

    summ = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    k = 0
    cuda = torch.cuda.is_available()

    with tqdm(total=len(train_loader)) as t:
        for i, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            output = model(data)

            output_temp = torch.from_numpy(
                output_orig[k:k + data.shape[0], :]).float().cuda()

            loss = F.cross_entropy(output, target).cuda(
            ) + mse_factor * F.mse_loss(output, output_temp).cuda()

            losses.update(loss.item(), data.size(0))

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                # compute all metrics on this batch
                summary_batch = {}
                summary_batch['acc_top1'] = top1.avg.cpu()
                summary_batch['acc_top5'] = top5.avg.cpu()
                summary_batch['loss'] = losses.avg
                summ.append(summary_batch)

            # update the average loss
            t.set_postfix(loss='{:05.3f}'.format(losses.avg))
            t.update()

    if logger:
        logger.info("- Train metrics (average): " + matrics_to_string(summ))
