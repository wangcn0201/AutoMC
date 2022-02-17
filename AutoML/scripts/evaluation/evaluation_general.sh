#!/bin/sh

python main_automl_evaluation_additional_finetune.py \
--config_path "config.json" \
--task_name "$6" \
--pretrain_epochs 200 \
--scheme_code "$4" \
--scheme_code_source "$1-$2-$3" \
--final_finetune_epoch_total $5 