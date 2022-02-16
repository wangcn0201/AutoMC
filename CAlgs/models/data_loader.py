import os
import sys
import time
import shutil
import pickle
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_data(data_name='cifar10', data_dir='./data', batch_size=128, workers=4, arch_name=None, return_data=False):

    if 'mini_' in data_name:
        if arch_name == None:
            raise ValueError('arch_name should not be None!')
        mini_data_dir = get_mini_data_dir(data_name[5:], arch_name, rate=0.1)
        return load_mini_data(mini_data_dir, batch_size=batch_size, workers=workers, return_data=return_data)

    if data_name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif data_name == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif data_name == 'svhn':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    else:
        assert False, "Unknow dataset : {}".format(data_name)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if data_name == 'cifar10':
        train_data = dset.CIFAR10(
            data_dir, train=True, transform=train_transform, download=True)
        val_data = dset.CIFAR10(data_dir, train=False,
                                transform=test_transform, download=True)
        num_classes = 10
    elif data_name == 'cifar100':
        train_data = dset.CIFAR100(
            data_dir, train=True, transform=train_transform, download=True)
        val_data = dset.CIFAR100(
            data_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif data_name == 'svhn':
        train_data = dset.SVHN(data_dir, split='train',
                               transform=train_transform, download=True)
        val_data = dset.SVHN(data_dir, split='test',
                             transform=test_transform, download=True)
        num_classes = 10
    elif data_name == 'stl10':
        train_data = dset.STL10(data_dir, split='train',
                                transform=train_transform, download=True)
        val_data = dset.STL10(data_dir, split='test',
                              transform=test_transform, download=True)
        num_classes = 10
    else:
        assert False, 'Do not support dataset : {}'.format(data_name)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=True)

    if return_data:
        return train_data, val_data, train_loader, val_loader
    else:
        return train_loader, val_loader

def load_mini_data(data_dir, batch_size=64, workers=4, return_data=False):

    f_train_data = open(data_dir + 'mini_train_data', 'rb')
    mini_train_data = pickle.load(f_train_data)
    mini_train_loader = torch.utils.data.DataLoader(mini_train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    f_val_data = open(data_dir + 'mini_val_data', 'rb')
    mini_val_data = pickle.load(f_val_data)
    mini_val_loader = torch.utils.data.DataLoader(mini_val_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    if return_data:
        return mini_train_data, mini_val_data, mini_train_loader, mini_val_loader
    else:
        return mini_train_loader, mini_val_loader

def get_mini_data_dir(data_name, arch_name, rate=0.1):
    base_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    data_dir = base_dir+'/data/mini_dataset/{}/{}/{}/'.format(data_name, arch_name, rate)
    return data_dir

def get_num_classes(data_name):
    if data_name == 'cifar10' or data_name == 'mini_cifar10':
        return 10
    elif data_name == 'cifar100' or data_name == 'mini_cifar100':
        return 100
    elif data_name == 'svhn' or data_name == 'mini_svhn':
        return 10
    elif data_name == 'stl10' or data_name == 'mini_stl10':
        return 10
