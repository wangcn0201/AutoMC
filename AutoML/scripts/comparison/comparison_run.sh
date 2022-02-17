#!/bin/bash

CUDA_VISIBLE_DEVICES=7 bash scripts/comparison/resnet56+mini_cifar10+0.3/evolution.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/comparison/resnet56+mini_cifar10+0.3/our_exp.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/comparison/resnet56+mini_cifar10+0.3/our_kg.sh
CUDA_VISIBLE_DEVICES=6 bash scripts/comparison/resnet56+mini_cifar10+0.3/our_kmodel.sh
CUDA_VISIBLE_DEVICES=6 bash scripts/comparison/resnet56+mini_cifar10+0.3/our_pmodel.sh
CUDA_VISIBLE_DEVICES=6 bash scripts/comparison/resnet56+mini_cifar10+0.3/our_space.sh
CUDA_VISIBLE_DEVICES=5 bash scripts/comparison/resnet56+mini_cifar10+0.3/our.sh
CUDA_VISIBLE_DEVICES=5 bash scripts/comparison/resnet56+mini_cifar10+0.3/random.sh
CUDA_VISIBLE_DEVICES=5 bash scripts/comparison/resnet56+mini_cifar10+0.3/random2.sh
CUDA_VISIBLE_DEVICES=4 bash scripts/comparison/resnet56+mini_cifar10+0.3/rl.sh

CUDA_VISIBLE_DEVICES=4 bash scripts/comparison/resnet56+mini_cifar100+0.3/evolution.sh
CUDA_VISIBLE_DEVICES=4 bash scripts/comparison/resnet56+mini_cifar100+0.3/our_exp.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/comparison/resnet56+mini_cifar100+0.3/our_kg.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/comparison/resnet56+mini_cifar100+0.3/our_kmodel.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/comparison/resnet56+mini_cifar100+0.3/our_pmodel.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/comparison/resnet56+mini_cifar100+0.3/our_space.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/comparison/resnet56+mini_cifar100+0.3/our.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/comparison/resnet56+mini_cifar100+0.3/random.sh
CUDA_VISIBLE_DEVICES=6 bash scripts/comparison/resnet56+mini_cifar100+0.3/random2.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/comparison/resnet56+mini_cifar100+0.3/rl.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/comparison/resnet56+mini_cifar100+0.3/our.sh

CUDA_VISIBLE_DEVICES=2 bash scripts/comparison/vgg16+mini_cifar10+0.3/evolution.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/comparison/vgg16+mini_cifar10+0.3/our_exp.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/comparison/vgg16+mini_cifar10+0.3/our_kg.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/comparison/vgg16+mini_cifar10+0.3/our_kmodel.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/comparison/vgg16+mini_cifar10+0.3/our_pmodel.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/comparison/vgg16+mini_cifar10+0.3/our_space.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/comparison/vgg16+mini_cifar10+0.3/our.sh
CUDA_VISIBLE_DEVICES=4 bash scripts/comparison/vgg16+mini_cifar10+0.3/random.sh
CUDA_VISIBLE_DEVICES=4 bash scripts/comparison/vgg16+mini_cifar10+0.3/random2.sh
CUDA_VISIBLE_DEVICES=4 bash scripts/comparison/vgg16+mini_cifar10+0.3/rl.sh





CUDA_VISIBLE_DEVICES=7 bash scripts/comparison/vgg16+mini_cifar100+0.3/our_exp.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/comparison/vgg16+mini_cifar100+0.3/our_kg.sh
CUDA_VISIBLE_DEVICES=7 bash scripts/comparison/vgg16+mini_cifar100+0.3/our_kmodel.sh
CUDA_VISIBLE_DEVICES=6 bash scripts/comparison/vgg16+mini_cifar100+0.3/our_pmodel.sh
CUDA_VISIBLE_DEVICES=6 bash scripts/comparison/vgg16+mini_cifar100+0.3/our_space.sh

CUDA_VISIBLE_DEVICES=2 bash scripts/comparison/vgg16+mini_cifar100+0.3/evolution.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/comparison/vgg16+mini_cifar100+0.3/our.sh
CUDA_VISIBLE_DEVICES=4 bash scripts/comparison/vgg16+mini_cifar100+0.3/random2.sh
CUDA_VISIBLE_DEVICES=5 bash scripts/comparison/vgg16+mini_cifar100+0.3/rl.sh
