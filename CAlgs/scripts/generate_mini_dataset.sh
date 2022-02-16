'''
重构mini_cifar10/mini_cifar100 for resnet20
重构mini_cifar10/mini_cifar100 for vgg13
'''

CUDA_VISIBLE_DEVICES=0 nohup python generate_mini_dataset.py cifar10 resnet56 > ./logs/mini_dataset_cifar10_resnet56.log &
CUDA_VISIBLE_DEVICES=1 nohup python generate_mini_dataset.py cifar100 resnet56 > ./logs/mini_dataset_cifar100_resnet56.log &
CUDA_VISIBLE_DEVICES=2 nohup python generate_mini_dataset.py cifar10 vgg16 > ./logs/mini_dataset_cifar10_vgg16.log &
CUDA_VISIBLE_DEVICES=7 nohup python generate_mini_dataset.py cifar100 vgg16 > ./logs/mini_dataset_cifar100_vgg16.log &