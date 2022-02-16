import os
import sys
import logging
import shutil
import glob
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from search_space import SearchSpace
from scheme_evaluation import SchemeEvaluation
import random
import json
import copy


class AutoMLRandom2(object):
	def __init__(self, config_path, task_name, task_info):
		super(AutoMLRandom2, self).__init__()
		self.space = SearchSpace
		self.calgs = self.space.keys()

		with open(config_path,"r") as f:
			config = json.load(f)
		config['task_name'] = task_name
		config['task_info'] = task_info
		self.environment_setting(config)

		self.cstartegies = self.get_spacecomponents(self.space)
		self.end_index = len(self.cstartegies)

		self.evaluator = SchemeEvaluation(config, self.logging_path, self.logger)
		self.max_calg_num = config['max_calg_num']
		self.target_compression_rate = float(task_name.split("+")[2])

		self.automl_search_time_s = float(config['automl_search_time(h)'])*60*60

		self.codes = 0
		self.valid_codes = 0
		self.steps = 0
		self.valid_steps = 0
		return

	def create_exp_dir(self, path, scripts_to_save=None):
		if not os.path.exists(path):
			os.mkdir(path)

		if scripts_to_save is not None:
			os.mkdir(os.path.join(path, 'scripts'))
			for script in scripts_to_save:
				dst_file = os.path.join(path, 'scripts', os.path.basename(script))
				shutil.copyfile(script, dst_file)
		return
	def environment_setting(self, config):
		logging_dir = config["logging_dir"]
		if not os.path.exists(logging_dir):
			os.mkdir(logging_dir)
		logging_path = logging_dir + 'AutoML_Random2-Run-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
		self.create_exp_dir(logging_path, scripts_to_save=glob.glob('*.py'))
		self.logging_path = logging_path
		
		log_format = '%(asctime)s %(message)s'
		self.logger = logging.getLogger()
		self.logger.setLevel(logging.INFO)
		fh = logging.FileHandler(os.path.join(logging_path, 'log.txt'))
		fh.setFormatter(logging.Formatter(log_format))
		self.logger.addHandler(fh)
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter(log_format))
		self.logger.addHandler(stream_handler)

		if not torch.cuda.is_available():
			self.logger.info('* no gpu device available')
			np.random.seed(config["seed"])
			self.logger.info("* config = %s", str(config))
			sys.exit(1)

		np.random.seed(config["seed"])
		#torch.cuda.set_device(config["gpu_device"])
		cudnn.benchmark = True
		torch.manual_seed(config["seed"])
		cudnn.enabled = True
		torch.cuda.manual_seed(config["seed"])
		self.logger.info("* config = %s", str(config))
		return

	def get_sub_spacecomponents(self, calg, hpo_dict, hpo_name):
		sub_cstartegies = []
		if len(hpo_name) == 1:
			for hpo_value in hpo_dict[hpo_name[0]]:
				content = {}
				content[hpo_name[0]] = hpo_value
				sub_cstartegies.append([calg, content])
			return sub_cstartegies
		contents = self.get_sub_spacecomponents(calg, hpo_dict, list(hpo_name[1:]))
		for hpo_value in hpo_dict[hpo_name[0]]:
			for i in range(len(contents)):
				content = copy.deepcopy(contents[i][1])
				content[hpo_name[0]] = hpo_value
				sub_cstartegies.append([calg, content])
		return sub_cstartegies
	def get_spacecomponents(self, space):
		self.logger.info('* get_spacecomponents now')
		if os.path.exists("cstartegies.txt"):
			with open("cstartegies.txt", "r") as f:
				cstartegies = eval(f.readline())
			self.logger.info('* %d cstrategies in all' % len(cstartegies))
			return cstartegies
		cstartegies = []
		for calg in space.keys():
			calg_hpo_dict = space[calg]
			calg_hpo_name = list(calg_hpo_dict.keys())
			sub_cstartegies = self.get_sub_spacecomponents(calg, calg_hpo_dict, calg_hpo_name)
			cstartegies.extend(sub_cstartegies)
			self.logger.info('* %d cstrategies in calg: %s', len(sub_cstartegies), calg)
		self.logger.info('* %d cstrategies in all', len(cstartegies))
		with open("cstartegies.txt", "w") as f:
			f.write(str(cstartegies))
		return cstartegies

	def get_scheme_code(self):
		scheme_code = []
		cstartegy_slections = [i for i in range(self.end_index+1)]
		while len(scheme_code) < self.max_calg_num:
			cstartegy_id = random.sample(cstartegy_slections, 1)[0]
			if cstartegy_id == self.end_index:
				break
			cstartegy = self.cstartegies[cstartegy_id]
			scheme_code.append(list(cstartegy))
		self.logger.info('* generate a random code ......')
		return scheme_code

	def scheme_code_validation_examine(self, scheme_code):
		'''
		Requirements: 
			scheme_code's finished_compression_rate >= 1.0 
			scheme_code's finished_compression_rate * self.target_compression_rate < 0.85
			scheme_code's finished_finetune_rate < 4.0 
		'''
		finished_compression_rate, finished_finetune_rate = 0.0, 0.0
		calg_num = len(scheme_code)
		for i in range(calg_num):
			calg, calg_hpo = scheme_code[i]

			finished_compression_rate += calg_hpo["HP2"]
			if "HP10" in list(calg_hpo.keys()):
				finished_finetune_rate += calg_hpo["HP10"]
			else:
				finished_finetune_rate += calg_hpo["HP1"]

			if calg == "prune_C7":
				if i != 0:
					scheme_code = list(scheme_code[:i])
					finished_compression_rate -= calg_hpo["HP2"]
					finished_finetune_rate -= calg_hpo["HP1"]
				else:
					scheme_code = list([scheme_code[0]])
				break
			elif calg == "prune_C5":
				scheme_code = list(scheme_code[:i+1])
				break

		valid = True
		if finished_compression_rate < 1.0 or self.target_compression_rate*finished_compression_rate >= 0.85 or finished_finetune_rate >= 4.0:
			valid = False
		self.logger.info('* finished_compression_rate: %.4f, target_compression_rate*finished_compression_rate: %.4f, finished_finetune_rate: %.4f', finished_compression_rate, self.target_compression_rate*finished_compression_rate, finished_finetune_rate)
		return scheme_code, valid

	def pareto_opt_tell(self, score1, score2):
		if score1[0] == score2[0] and score1[1] > score2[1]: # 0: acc, 1: compression_rate
			return True # score1 better, delete score2

		if score1[0] > score2[0] and score1[1] > score2[1]:
			return True # score1 better, delete score2

		if score1[0] > score2[0] and score1[1] == score2[1]:
			return True # score1 better, delete score2

		if score1[0] == score2[0] and score1[1] < score2[1]:
			return False # score2 better, score1 not Pareto

		if score1[0] == score2[0] and score1[1] == score2[1]:
			return False # score2 better, score1 not Pareto

		if score1[0] < score2[0] and score1[1] < score2[1]:
			return False # score2 better, score1 not Pareto

		if score1[0] < score2[0] and score1[1] == score2[1]:
			return False # score2 better, score1 not Pareto
		else:
			return None

	def main(self):
		pareto_front_schemes_info = []
		score_opt_scheme = {"codes/valid_codes": [None,None], "steps/valid_steps": [None,None], "scheme_code": None, "scheme_score": None}
		
		start_time = time.time()
		search_time = 0.0
		while search_time < self.automl_search_time_s:
			self.codes += 1
			scheme_code = self.get_scheme_code()
			self.steps += len(scheme_code)
			self.logger.info('@ codes: %s, steps: %s', str(self.codes), str(self.steps))
			self.logger.info('@ original scheme_code: %s', str(scheme_code))

			scheme_code, valid = self.scheme_code_validation_examine(scheme_code)
			if valid:
				self.valid_codes += 1
				self.valid_steps += len(scheme_code)
			self.logger.info('@ valid: %s', str(valid))
			self.logger.info('@ valid_codes: %s, valid_steps: %s', str(self.valid_codes), str(self.valid_steps))
			self.logger.info('@ adjusted scheme_code: %s', str(scheme_code))

			if valid:
				scheme_score, table_infos = self.evaluator.main(scheme_code)
			else:
				scheme_score = [0, 0, 0, '0M', '0G']
				table_infos = {
						"top1/top1_increased": [0, -1], 
						"top5/top5_increased": [0, -1],
						"flops/flops_decreased": ["-", 0], 
						"parameter/parameter_decreased": ["-", 0]
						}
			self.logger.info('@ scheme_score: %s', str(scheme_score))
			self.logger.info('@ table_infos: %s', str(table_infos))

			self.logger.info('@ update optimal result......')
			if score_opt_scheme["scheme_score"] == None or (scheme_score[0] > score_opt_scheme["scheme_score"][0] and scheme_score[1] >= self.target_compression_rate):
				score_opt_scheme = {"codes/valid_codes": [self.codes, self.valid_codes], "steps/valid_steps": [self.steps, self.valid_steps], "scheme_code": scheme_code, "scheme_score": scheme_score, "table_infos": table_infos}
			if scheme_score[1] >= self.target_compression_rate:
				pareto_front, removed = True, []
				for pareto_front_scheme in pareto_front_schemes_info:
					if self.pareto_opt_tell(scheme_score, pareto_front_scheme["scheme_score"]) == True:
						removed.append(pareto_front_scheme)
					elif self.pareto_opt_tell(scheme_score, pareto_front_scheme["scheme_score"]) == False:
						pareto_front = False
				if pareto_front:
					pareto_front_schemes_info.append({"codes/valid_codes": [self.codes, self.valid_codes], "steps/valid_steps": [self.steps, self.valid_steps], "scheme_code": scheme_code, "scheme_score": scheme_score, "table_infos": table_infos})
				for item in removed:
					pareto_front_schemes_info.remove(item)
			self.logger.info('@ score_opt_scheme: %s', str(score_opt_scheme))
			self.logger.info('@ pareto_front_schemes_info: %s', str(pareto_front_schemes_info))
			#for k in range(len(pareto_front_schemes_info)):
			#	self.logger.info('@ pareto_front_schemes_info %d: %s', k, str(pareto_front_schemes_info[k]))
			
			search_time = time.time()-start_time
			self.logger.info('@ automl_search_time (seconds): %.4f, (hours): %.4f', search_time, search_time/60.0/60.0)
			self.logger.info('@ automl_search_time left (seconds): %.4f, (hours): %.4f', self.automl_search_time_s-search_time, (self.automl_search_time_s-search_time)/60.0/60.0)
			self.logger.info('@ average time (seconds)/(minutes) used for per valid_codes: %.4f / %.4f', search_time/max(self.valid_codes,1), search_time/max(self.valid_codes,1)/60.0)

		self.pareto_front_schemes_info = pareto_front_schemes_info
		self.score_opt_scheme = score_opt_scheme
		self.logger.info('@ Final pareto_front_schemes_info: %s', str(pareto_front_schemes_info))
		for k in range(len(pareto_front_schemes_info)):
			self.logger.info('@ Final pareto_front_schemes_info %d: %s', k, str(pareto_front_schemes_info[k]))
		self.logger.info('@ Final score_opt_scheme: %s', str(score_opt_scheme))
		search_time = time.time()-start_time
		self.logger.info('@ Final automl_search_time (seconds): %.4f, (hours): %.4f', search_time, search_time/60.0/60.0)
		return 

'''
if __name__ == '__main__':
	config_path = "config.json"
	task_name = "resnet56+mini_cifar10+0.3"
	task_info = "{'class_num': 10, 'image_size': 32, 'image_channel': 3, 'avg_data_num_per_class': 600, 'top1_acc': 0.6763, 'top5_acc': 0, 'parameter_amount': 0.284, 'flops_amount': 43.42}"
	nas_obj = AutoMLRandom2(config_path, task_name, task_info)
	nas_obj.main()
'''