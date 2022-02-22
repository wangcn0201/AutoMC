
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
