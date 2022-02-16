'''
resnet56+cifar10/cifar100+200轮
resnet20+cifar10/cifar100+200轮
resnet164+cifar10/cifar100+200轮
重构mini_cifar10/mini_cifar100 for resnet56
resnet56+mini_cifar10/mini_cifar100+50轮

vgg16+cifar10/cifar100+200轮
vgg13+cifar10/cifar100+200轮
vgg19+cifar10/cifar100+200轮
重构mini_cifar10/mini_cifar100 for vgg16
vgg16+mini_cifar10/mini_cifar100+50轮
'''

CUDA_VISIBLE_DEVICES=0 nohup python scripts/pretrain.py cifar10 vgg13 200 > ./logs/log_cifar10_vgg13_200.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/pretrain.py cifar10 vgg16 200 > ./logs/log_cifar10_vgg16_200.log &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/pretrain.py cifar10 vgg19 200 > ./logs/log_cifar10_vgg19_200.log &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/pretrain.py cifar10 resnet20 200 > ./logs/log_cifar10_resnet20_200.log &
CUDA_VISIBLE_DEVICES=4 nohup python scripts/pretrain.py cifar10 resnet56 200 > ./logs/log_cifar10_resnet56_200.log &
CUDA_VISIBLE_DEVICES=5 nohup python scripts/pretrain.py cifar10 resnet164 200 > ./logs/log_cifar10_resnet164_200.log &

CUDA_VISIBLE_DEVICES=6 nohup python scripts/pretrain.py cifar100 vgg13 200 > ./logs/log_cifar100_vgg13_200.log &
CUDA_VISIBLE_DEVICES=7 nohup python scripts/pretrain.py cifar100 vgg16 200 > ./logs/log_cifar100_vgg16_200.log &
CUDA_VISIBLE_DEVICES=0 nohup python scripts/pretrain.py cifar100 vgg19 200 > ./logs/log_cifar100_vgg19_200.log &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/pretrain.py cifar100 resnet20 200 > ./logs/log_cifar100_resnet20_200.log &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/pretrain.py cifar100 resnet56 200 > ./logs/log_cifar100_resnet56_200.log &
CUDA_VISIBLE_DEVICES=6 nohup python scripts/pretrain.py cifar100 resnet164 200 > ./logs/log_cifar100_resnet164_200.log &
