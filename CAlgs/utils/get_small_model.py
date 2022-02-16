import torch
import torch.nn as nn
import models
import numpy as np
from .utils import *


def get_modules(model, arch_name, label_layer):
    if label_layer == 'conv':
        label_layer = nn.Conv2d
    elif label_layer == 'bn':
        label_layer = nn.BatchNorm2d
    else:
        raise ValueError('Wrong label layer {}!'.format(label_layer))
    modules = []
    if 'wide_resnet' in arch_name:
        for module in model.named_modules():
            name = module[0]
            module = module[1]
            if 'shortcut' in name and name[-1].isdigit():
                modules.append((module, ['shortcut'], name))
            elif layer_parameter_num(module):
                label = []
                if isinstance(module, nn.Conv2d):
                    label.append('prune')
                if isinstance(module, label_layer):
                    label.append('criterion')
                modules.append((module, label, name))
    elif 'resnet' in arch_name:
        active = False
        for module in model.named_modules():
            name = module[0]
            module = module[1]
            if isinstance(module, nn.Sequential) and not active:
                active = True
            if 'shortcut' in name and name[-1].isdigit():
                modules.append((module, ['shortcut'], name))
            elif layer_parameter_num(module):
                label = []
                if isinstance(module, nn.Conv2d) and active:
                    label.append('prune')
                if isinstance(module, label_layer) and active:
                    label.append('criterion')
                modules.append((module, label, name))
    elif 'vgg' in arch_name:
        for module in model.named_modules():
            name = module[0]
            module = module[1]
            if layer_parameter_num(module):
                label = []
                if isinstance(module, nn.Conv2d):
                    label.append('prune')
                if isinstance(module, label_layer):
                    label.append('criterion')
                modules.append((module, label, name))
    elif 'densenet' in arch_name:
        for module in model.named_modules():
            name = module[0]
            module = module[1]
            if layer_parameter_num(module):
                label = []
                if isinstance(module, nn.Conv2d):
                    label.append('prune')
                if isinstance(module, label_layer):
                    label.append('criterion')
                modules.append((module, label, name))
    
    return modules


def check_channel(tensor):
    size_0 = tensor.size()[0]
    tensor_resize = tensor.view(size_0, -1)
    channel_nonzero = np.zeros(size_0)

    for x in range(0, size_0):
        channel_nonzero[x] = np.count_nonzero(tensor_resize[x].cpu().detach().numpy()) != 0

    indices_nonzero = torch.LongTensor((channel_nonzero != 0).nonzero()[0])
    indices_zero = torch.LongTensor((channel_nonzero == 0).nonzero()[0])

    return indices_zero, indices_nonzero


def get_channel_index(modules, model_name, prune_type, args):
    if prune_type == 'conv':
        prune_type = nn.Conv2d
    elif prune_type == 'bn':
        prune_type = nn.BatchNorm2d
    kept_index = []
    kept_num = []
    # pruned_index = {}
    for index in range(len(modules)):
        if 'criterion' in modules[index][1] or (len(modules[index][1]) == 0 and isinstance(modules[index][0], prune_type)):
            indices_zero, indices_nonzero = check_channel(modules[index][0].weight)
            '''
            if isinstance(modules[index][0], nn.BatchNorm2d):
                print(item.data)
            else:
                print(type(modules[index][0]))
            '''
            # pruned_index[index] = indices_zero
            kept_index.append(indices_nonzero)
            kept_num.append(indices_nonzero.shape[0])
    # print(kept_num)
    small_model = models.__dict__[model_name](num_classes=args['num_classes'], cfg=kept_num, activation=args['activation'], numBins=args['numBins'], special=args['special'])
    return kept_index, small_model


def set_params_conv(module, small_module, in_channels, out_channels):
    conv = True
    for item, new_item in zip(module.parameters(), small_module.parameters()):
        if conv:
            _ = torch.index_select(item.data, 1, in_channels)
            new_item.data = torch.index_select(_, 0, out_channels)
        else:
            new_item.data = torch.index_select(item.data, 0, out_channels)
        conv = False


def set_params_bn(module, small_module, channels):
    for item, new_item in zip(module.parameters(), small_module.parameters()):
        new_item.data = torch.index_select(item.data, 0, channels)


def set_params_linear(module, small_module, channels):
    first = True
    for item, new_item in zip(module.parameters(), small_module.parameters()):
        if first:
            new_item.data = torch.index_select(item.data, 1, channels)
            first = False
        else:
            new_item.data = item.data


def set_model_params_vgg(arch_name, kept_index, small_model, modules, prune_type):
    small_modules = get_modules(small_model, arch_name, prune_type)
    last_channel_index, p = torch.tensor((0, 1, 2), dtype=torch.int32), 0

    for index in range(len(modules)):
        if isinstance(modules[index][0], nn.Conv2d):
            current_channel_index = kept_index[p]
            p += 1
            set_params_conv(modules[index][0], small_modules[index][0], last_channel_index, current_channel_index)
            last_channel_index = current_channel_index
        elif isinstance(modules[index][0], nn.BatchNorm2d):
            set_params_bn(modules[index][0], small_modules[index][0], last_channel_index)
        else:
            set_params_linear(modules[index][0], small_modules[index][0], last_channel_index)

    assert(p == len(kept_index))


def set_model_params_resnet_and_wide_resnet(arch_name, kept_index, small_model, modules, prune_type):
    small_modules = get_modules(small_model, arch_name, prune_type)
    label = 'bn1' if 'wide_resnet' in arch_name else 'conv1'
    last_channel_index, p = torch.tensor((0, 1, 2), dtype=torch.int32), 0

    for index in range(len(modules)):
        if label in modules[index][2]:
            basic_start_index = last_channel_index
            
        if isinstance(modules[index][0], nn.Conv2d) and 'shortcut' not in modules[index][1]:
            current_channel_index = kept_index[p]
            p += 1
            set_params_conv(modules[index][0], small_modules[index][0], last_channel_index, current_channel_index)
            last_channel_index = current_channel_index
        elif isinstance(modules[index][0], nn.Conv2d) and 'shortcut' in modules[index][1]:
            set_params_conv(modules[index][0], small_modules[index][0], basic_start_index, last_channel_index)
        elif isinstance(modules[index][0], nn.BatchNorm2d):
            set_params_bn(modules[index][0], small_modules[index][0], last_channel_index)
        else:
            set_params_linear(modules[index][0], small_modules[index][0], last_channel_index)

    assert(p == len(kept_index))

def set_model_params_resnet_cifar(arch_name, kept_index, small_model, modules, prune_type):
    small_modules = get_modules(small_model, arch_name, prune_type)
    label = 'bn1'
    last_channel_index, p = torch.tensor((0, 1, 2), dtype=torch.int32), 0


    for index in range(len(modules)):
        if label in modules[index][2]:
            basic_start_index = last_channel_index
            
        if isinstance(modules[index][0], nn.Conv2d) and 'shortcut' not in modules[index][1]:
            current_channel_index = kept_index[p]
            p += 1
            set_params_conv(modules[index][0], small_modules[index][0], last_channel_index, current_channel_index)
            last_channel_index = current_channel_index
        elif isinstance(modules[index][0], nn.Conv2d) and 'shortcut' in modules[index][1]:
            set_params_conv(modules[index][0], small_modules[index][0], basic_start_index, last_channel_index)
        elif isinstance(modules[index][0], nn.BatchNorm2d) and 'shortcut' not in modules[index][1]:
            set_params_bn(modules[index][0], small_modules[index][0], last_channel_index)
        elif isinstance(modules[index][0], nn.BatchNorm2d) and 'shortcut' in modules[index][1]:
            set_params_bn(modules[index][0], small_modules[index][0], basic_start_index)
        else:
            set_params_linear(modules[index][0], small_modules[index][0], last_channel_index)

    assert(p == len(kept_index))


def get_kept_index(kept_index, pre_channels):
    res = []
    for i in range(len(kept_index)):
        res.append(kept_index[i] + pre_channels)
    return torch.tensor(res, dtype=torch.int32)

def set_model_params_densenet(arch_name, kept_index, small_model, modules, prune_type):
    small_modules = get_modules(small_model, arch_name, prune_type)

    '''
    print('>>>>>>>>>>>>>>>>>> model:')
    for index in range(len(modules)):
        print(index, modules[index])

    print('>>>>>>>>>>>>>>>>>> small_model:')
    for index in range(len(small_modules)):
        print(index, small_modules[index])
    print(small_model)
    '''

    last_channel_index, p = torch.tensor((0, 1, 2), dtype=torch.int32), 0
    current_channel_index = torch.tensor([], dtype=torch.int32)
    pre_channels = 0

    for index in range(len(modules)):
        if isinstance(modules[index][0], nn.Conv2d) and 'dense' in modules[index][2]:
            current_channel_index = kept_index[p]
            p += 1
            set_params_conv(modules[index][0], small_modules[index][0], last_channel_index, current_channel_index)
            last_channel_index = torch.cat((last_channel_index, get_kept_index(current_channel_index, pre_channels)), 0)
            pre_channels += modules[index][0].weight.shape[0]
        elif isinstance(modules[index][0], nn.Conv2d):
            current_channel_index = kept_index[p]
            p += 1
            set_params_conv(modules[index][0], small_modules[index][0], last_channel_index, current_channel_index)
            last_channel_index = current_channel_index
            pre_channels = modules[index][0].weight.shape[0]
        elif isinstance(modules[index][0], nn.BatchNorm2d):
            set_params_bn(modules[index][0], small_modules[index][0], kept_index[p - 1])
        else:
            set_params_linear(modules[index][0], small_modules[index][0], last_channel_index)

    assert(p == len(kept_index))


def get_small_model(model, arch_name, prune_type, small_model_with_param=True):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    modules = get_modules(model, arch_name, prune_type)

    kept_index, small_model = get_channel_index(modules, arch_name, prune_type, model.args)

    model.to('cpu')
    small_model.to('cpu')

    if not small_model_with_param:
        return small_model

    # print('kept_index:\n', kept_index)
    if arch_name in ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']:
        set_model_params_resnet_cifar(arch_name, kept_index, small_model, modules, prune_type)
    elif 'wide_resnet' in arch_name or 'resnet' in arch_name:
        set_model_params_resnet_and_wide_resnet(arch_name, kept_index, small_model, modules, prune_type)
    elif 'vgg' in arch_name:
        set_model_params_vgg(arch_name, kept_index, small_model, modules, prune_type)
    elif 'densenet' in arch_name:
        set_model_params_densenet(arch_name, kept_index, small_model, modules, prune_type)

    return small_model