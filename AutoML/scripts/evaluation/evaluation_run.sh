#!/bin/bash

#
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/resnet56+cifar10+0.4/evolution.sh 300
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/resnet56+cifar10+0.4/our_exp.sh 300
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/resnet56+cifar10+0.4/our_kg.sh 300
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/resnet56+cifar10+0.4/our_kmodel.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/resnet56+cifar10+0.4/our_pmodel.sh 300

#
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/resnet56+cifar10+0.4/our_space.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/resnet56+cifar10+0.4/our.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/resnet56+cifar10+0.4/random.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/resnet56+cifar10+0.4/random2.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/resnet56+cifar10+0.4/rl.sh 300

#
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/resnet56+cifar10+0.7/evolution.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/resnet56+cifar10+0.7/our_exp.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/resnet56+cifar10+0.7/our_kg.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/resnet56+cifar10+0.7/our_kmodel.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/resnet56+cifar10+0.7/our_pmodel.sh 300

#
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/resnet56+cifar10+0.7/our_space.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/resnet56+cifar10+0.7/our.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/resnet56+cifar10+0.7/random.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/resnet56+cifar10+0.7/random2.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/resnet56+cifar10+0.7/rl.sh 300





# 
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/resnet20+cifar10+0.7/our.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/resnet164+cifar10+0.7/our.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/resnet20+cifar10+0.4/our.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/resnet164+cifar10+0.4/our.sh 300











#
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/vgg16+cifar100+0.4/evolution.sh 300
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/vgg16+cifar100+0.4/our.sh 300
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/vgg16+cifar100+0.4/random2.sh 300
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/vgg16+cifar100+0.4/rl.sh 300
#
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/vgg16+cifar100+0.7/evolution.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/vgg16+cifar100+0.7/our.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/vgg16+cifar100+0.7/random2.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/vgg16+cifar100+0.7/rl.sh 300


#
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/vgg13+cifar100+0.7/our.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/vgg19+cifar100+0.7/our.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/vgg13+cifar100+0.4/our.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/vgg19+cifar100+0.4/our.sh 300


#
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/resnet20+cifar10+0.4/random2.sh 300
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/resnet20+cifar10+0.4/rl.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/resnet20+cifar10+0.4/evolution.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/resnet20+cifar10+0.7/random2.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/resnet20+cifar10+0.7/rl.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/resnet20+cifar10+0.7/evolution.sh 300

#
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/resnet164+cifar10+0.4/random2.sh 300
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/resnet164+cifar10+0.4/rl.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/resnet164+cifar10+0.4/evolution.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/resnet164+cifar10+0.7/random2.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/resnet164+cifar10+0.7/rl.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/resnet164+cifar10+0.7/evolution.sh 300

#
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/vgg19+cifar100+0.4/random2.sh 300
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/vgg19+cifar100+0.4/rl.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/vgg19+cifar100+0.4/evolution.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/vgg19+cifar100+0.7/random2.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/vgg19+cifar100+0.7/rl.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/vgg19+cifar100+0.7/evolution.sh 300

#
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/vgg13+cifar100+0.4/random2.sh 300
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/vgg13+cifar100+0.4/rl.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/vgg13+cifar100+0.4/evolution.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/vgg13+cifar100+0.7/random2.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/vgg13+cifar100+0.7/rl.sh 300
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/vgg13+cifar100+0.7/evolution.sh 300


#
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/vgg16+cifar100+0.4/our_exp.sh 300
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/vgg16+cifar100+0.4/our_kg.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/vgg16+cifar100+0.4/our_kmodel.sh 300
CUDA_VISIBLE_DEVICES=4 bash scripts/evaluation/vgg16+cifar100+0.4/our_pmodel.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/vgg16+cifar100+0.4/our_space.sh 300

#
CUDA_VISIBLE_DEVICES=2 bash scripts/evaluation/vgg16+cifar100+0.7/our_exp.sh 300
CUDA_VISIBLE_DEVICES=7 bash scripts/evaluation/vgg16+cifar100+0.7/our_kg.sh 300
CUDA_VISIBLE_DEVICES=6 bash scripts/evaluation/vgg16+cifar100+0.7/our_kmodel.sh 300
CUDA_VISIBLE_DEVICES=5 bash scripts/evaluation/vgg16+cifar100+0.7/our_pmodel.sh 300
CUDA_VISIBLE_DEVICES=3 bash scripts/evaluation/vgg16+cifar100+0.7/our_space.sh 300