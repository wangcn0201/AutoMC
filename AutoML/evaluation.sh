#!/bin/sh

python main_automl_evaluation_additional_finetune.py \
--config_path "config.json" \
--task_name "resnet56+cifar10+0.3" \
--pretrain_epochs 200 \
--scheme_code "[['prune_C2', {'HP9': 'l2_weight', 'HP8': 0.01, 'HP7': 0.6, 'HP6': 0.9, 'HP2': 0.2, 'HP1': 0.5}], ['prune_C2', {'HP9': 'l2_weight', 'HP8': 0.01, 'HP7': 0.4, 'HP6': 0.7, 'HP2': 0.3, 'HP1': 0.2}], ['prune_C5', {'HP15': 1, 'HP14': 0.5, 'HP13': 'skew_kur', 'HP12': 'P1', 'HP2': 0.5, 'HP1': 0.1}]]" \
--scheme_code_source "automl_random_pareto_1" \
--final_finetune_epoch_total 300 

:<<!
python main_automl_evaluation_additional_finetune.py \
--config_path "config.json" \
--task_name "resnet56+cifar10+0.3" \
--pretrain_epochs 200 \
--scheme_code "[['prune_C2', {'HP9': 'l2_weight', 'HP8': 0.01, 'HP7': 0.6, 'HP6': 0.9, 'HP2': 0.2, 'HP1': 0.5}], ['prune_C2', {'HP9': 'l2_weight', 'HP8': 0.01, 'HP7': 0.4, 'HP6': 0.7, 'HP2': 0.3, 'HP1': 0.2}], ['prune_C5', {'HP15': 1, 'HP14': 0.5, 'HP13': 'skew_kur', 'HP12': 'P1', 'HP2': 0.5, 'HP1': 0.1}]]" \
--scheme_code_source "automl_random_pareto_1" \
--final_finetune_epoch_total -1 \
!