'''
resnet56+mini_cifar10/mini_cifar100+50轮

vgg16+mini_cifar10/mini_cifar100+50轮
'''

CUDA_VISIBLE_DEVICES=1 nohup python scripts/pretrain.py mini_cifar10 vgg16 50 > ./logs/log_mini_cifar10_vgg16_50.log &
CUDA_VISIBLE_DEVICES=4 nohup python scripts/pretrain.py mini_cifar10 resnet56 50 > ./logs/log_mini_cifar10_resnet56_50.log &
CUDA_VISIBLE_DEVICES=5 nohup python scripts/pretrain.py mini_cifar100 vgg16 50 > ./logs/log_mini_cifar100_vgg16_50.log &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/pretrain.py mini_cifar100 resnet56 50 > ./logs/log_mini_cifar100_resnet56_50.log &
