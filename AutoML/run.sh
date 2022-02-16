#!/bin/sh

CUDA_VISIBLE_DEVICES=0 nohup sh comparison_0.sh > ./comparison_resnet_cifar10_random_0.log &
CUDA_VISIBLE_DEVICES=1 nohup sh comparison_1.sh > ./logs_run/comparison_resnet_cifar10_rl_1.log &
CUDA_VISIBLE_DEVICES=2 nohup sh comparison_2.sh > ./logs_run/comparison_resnet_cifar10_evolution_2.log &
CUDA_VISIBLE_DEVICES=3 nohup sh comparison_3.sh > ./logs_run/comparison_resnet_cifar10_our_3.log &
CUDA_VISIBLE_DEVICES=4 nohup sh comparison_4.sh > ./logs_run/comparison_resnet_cifar10_our_exp_4.log &
CUDA_VISIBLE_DEVICES=5 nohup sh comparison_5.sh > ./logs_run/comparison_resnet_cifar10_our_kg_5.log &
CUDA_VISIBLE_DEVICES=6 nohup sh comparison_6.sh > ./logs_run/comparison_resnet_cifar10_our_kmodel_6.log &
CUDA_VISIBLE_DEVICES=7 nohup sh comparison_7.sh > ./logs_run/comparison_resnet_cifar10_our_pmodel_7.log &
CUDA_VISIBLE_DEVICES=0 nohup sh comparison.sh > ./logs_run/comparison_resnet_cifar10_our_space_0.log &
