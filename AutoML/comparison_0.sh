#!/bin/sh

python main_automl_comparison.py \
--automl_method "random" \
--config_path "config.json" \
--task_name "resnet56+mini_cifar10+0.3" \
--task_info "{'class_num': 10, 'image_size': 32, 'image_channel': 3, 'avg_data_num_per_class': 600, 'top1_acc': 0.6763, 'top5_acc': 0, 'parameter_amount': 0.284, 'flops_amount': 43.42}" 

:<<!
python main_automl_comparison.py \
--automl_method "random" \
--config_path "config.json" \
--task_name "vgg16+mini_cifar10+0.3" \
--task_info "{'class_num': 10, 'image_size': 32, 'image_channel': 3, 'avg_data_num_per_class': 600, 'top1_acc': 0.7556, 'top5_acc': 0, 'parameter_amount': 9.416, 'flops_amount': 229.39}" 

python main_automl_comparison.py \
--automl_method "random" \
--config_path "config.json" \
--task_name "resnet56+mini_cifar100+0.3" \
--task_info "{'class_num': 100, 'image_size': 32, 'image_channel': 3, 'avg_data_num_per_class': 600, 'top1_acc': 0.2072, 'top5_acc': 0, 'parameter_amount': 0.290, 'flops_amount': 43.42}" 

python main_automl_comparison.py \
--automl_method "random" \
--config_path "config.json" \
--task_name "vgg16+mini_cifar100+0.3" \
--task_info "{'class_num': 100, 'image_size': 32, 'image_channel': 3, 'avg_data_num_per_class': 600, 'top1_acc': 0.3478, 'top5_acc': 0, 'parameter_amount': 9.462, 'flops_amount': 229.44}" 
!