import os
import sys
import logging
import shutil
import glob
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
import json
import argparse
from scheme_evaluation_additional_finetune import SchemeEvaluation


# Parser arguments
parser = argparse.ArgumentParser(description='AutoML Method for Model Compression Scheme Design.')
parser.add_argument('--config_path', '--cp', type=str, default="config.json", help='the file path for the config information')
parser.add_argument('--task_name', '--tn', type=str, default="resnet56+cifar10+0.3", help='the compression task name')
parser.add_argument('--pretrain_epochs', '--pe', type=int, default=200, help='the pretrain epochs')
parser.add_argument('--scheme_code', '--sc', type=str, default="", help='the task information')
parser.add_argument('--scheme_code_source', '--scs', type=str, default="", help='the source of the compression scheme code')
parser.add_argument('--final_finetune_epoch_total', '--ffet', type=int, default=300, help='the total fine tune epochs in the last compression method (-1 means on additional fine tuning)')

args = parser.parse_args()

def create_exp_dir(path, scripts_to_save=None):
	if not os.path.exists(path):
		os.mkdir(path)

	if scripts_to_save is not None:
		os.mkdir(os.path.join(path, 'scripts'))
		for script in scripts_to_save:
			dst_file = os.path.join(path, 'scripts', os.path.basename(script))
			shutil.copyfile(script, dst_file)
	return
def environment_setting(config):
	logging_dir = config["logging_dir"]
	if not os.path.exists(logging_dir):
		os.mkdir(logging_dir)
	scs = config["scheme_code_source"].split('-')
	logging_path = logging_dir + 'Evaluation-Run-{}-{}-{}'.format(scs[1], scs[2], time.strftime("%Y%m%d-%H%M%S"))
	create_exp_dir(logging_path, scripts_to_save=glob.glob('*.py'))
	logging_path = logging_path

	log_format = '%(asctime)s %(message)s'
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	fh = logging.FileHandler(os.path.join(logging_path, 'log.txt'))
	fh.setFormatter(logging.Formatter(log_format))
	logger.addHandler(fh)
	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(logging.Formatter(log_format))
	logger.addHandler(stream_handler)

	if not torch.cuda.is_available():
		logger.info('* no gpu device available')
		np.random.seed(config["seed"])
		logger.info("* config = %s", str(config))
		sys.exit(1)

	np.random.seed(config["seed"])
	#torch.cuda.set_device(config["gpu_device"])
	cudnn.benchmark = True
	torch.manual_seed(config["seed"])
	cudnn.enabled = True
	torch.cuda.manual_seed(config["seed"])
	logger.info("* config = %s", str(config))
	return logging_path, logger

def get_real_schemecode(codesring):
	i = 1
	items = []
	while i < len(codesring):
		if codesring[i] == '[':
			if 1 == 1:
				for j in range(i+1, len(codesring)):
					if codesring[j] == ']':
						break
				item = codesring[i:j+1]
				split_index = item.index(', ')+1
				alg, hpo = item[1:split_index-1], item[split_index+2:len(item)-2]
				new_hpo = {}
				hpos = hpo.split(', ')
				for i in range(len(hpos)):
					key, value = hpos[i].split(': ')
					try:
						value = eval(value)
					except:
						value = value
					if "'" in key:
						key = eval(key)
					new_hpo[key] = value

				if "'" in alg:
					alg = eval(alg)
				new_item = [alg, new_hpo]
				items.append(new_item)
				i = j+1
		else:
			i = i+1
	return items


if __name__ == '__main__':
	with open(args.config_path,"r") as f:
		config = json.load(f)
	config['task_name'] = args.task_name
	config['pretrain_epochs'] = args.pretrain_epochs
	config['scheme_code_source'] = args.scheme_code_source

	logging_path, logger = environment_setting(config)
	logger.info('@ args: %s', str(args))
	logger.info('@ config: %s', str(config))
	
	evaluator = SchemeEvaluation(config, logging_path, logger, args.final_finetune_epoch_total)

	logger.info('@ scheme_code: %s', str(args.scheme_code))
	scheme_code = get_real_schemecode(args.scheme_code)
	logger.info('@ real scheme_code: %s', str(scheme_code))

	start_time = time.time()
	scheme_score, table_infos = evaluator.main(scheme_code)
	logger.info('@ scheme_score: %s', str(scheme_score))
	logger.info('@ table_infos: %s', str(table_infos))
	logger.info('@ All running_time %.4f (seconds), (hours): %.4f', time.time() - start_time, (time.time() - start_time)/60.0/60.0)

