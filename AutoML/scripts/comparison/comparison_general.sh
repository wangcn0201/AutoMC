#!/bin/sh

python main_automl_comparison.py \
--automl_method "$1" \
--config_path "config.json" \
--task_name "$2+mini_cifar$3+$4" \
--task_info "{'class_num': $3, 'image_size': 32, 'image_channel': 3, 'avg_data_num_per_class': 600, 'top1_acc': $5, 'top5_acc': $6, 'parameter_amount': $7, 'flops_amount': $8}" 
