import os
import sys
import shutil
import logging
import torch
import numpy as np
import torch.backends.cudnn as cudnn
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))),"CAlgs")) 
import prune_C1, prune_C2, prune_C3, prune_C4, prune_C5, prune_C7
import time
import copy
import math


class SchemeEvaluationStep(object):
	def __init__(self, config, logging_path, logger):
		task_name = config["task_name"]
		model_name, self.data_name, self.target_compression_rate = task_name.split("+")
		self.target_compression_rate = float(self.target_compression_rate)

		self.data = {'dir':'../CAlgs/data', 'name':self.data_name}
		if model_name == "resnet":
			model_name = "resnet56"
		elif model_name == "vgg":
			model_name = "vgg16"
		elif model_name == "densenet":
			model_name = "densenet40"
		elif model_name == "wideresnet":
			model_name = "wide_resnet28"
		else: 
			model_name = model_name
		self.model_name = model_name

		self.logging_path = logging_path
		self.pretrain_epochs = config["pretrain_epochs"]
		self.logger = logger
		return

	def main(self, step_code, pre_model_dir=None, pre_parameter_remain_rate=None, pre_flops_remain_rate=None, pre_acc_top1_rate=None, pre_acc_top5_rate=None):
		try:
			if pre_model_dir == None or pre_parameter_remain_rate == None or pre_flops_remain_rate == None:
				self.model = {'dir': '../CAlgs/trained_models/{}/{}.pth.tar'.format(self.data_name, self.model_name), 'name': self.model_name}
				self.pre_init_rate = float(self.target_compression_rate) # delected rate
				pre_parameter_remain_rate = 1.0 
				pre_flops_remain_rate = 1.0
				pre_acc_top1_rate = 1.0
				pre_acc_top5_rate = 1.0
			else:
				self.model = {'dir': pre_model_dir, 'name': self.model_name}
				self.pre_init_rate = self.target_compression_rate/pre_parameter_remain_rate # delected rate

			self.logger.info('\t## evaluation step_code: %s', str(step_code))
			data = self.data
			model = self.model
			save_dir = self.logging_path + "/SchemeEvaluationStep_LOG/"
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			else:
				cmd = "rm -rf "+str(save_dir)+"ceckpoint*.pth.tar"
				os.system(cmd)

			init_rate = self.pre_init_rate
			start_time = time.time()
			calg, calg_hpo = step_code

			self.logger.info('\t## init_rate: %.4f, rate: %.4f', init_rate, max(1-init_rate*calg_hpo["HP2"],0.1))
			self.logger.info('\t## input model: %s', str(model))

			if calg == "prune_C1": # HP1: fine_tune_epochs; HP2: prune_ratio's ratio; HP3: LMA_segment_num; HP4: distillation_temperature_factor; HP5: distillation_alpha_factor 
				result_metrics1, result_param1, result_flops1, model_dir1 = prune_C1.knowledge_distillation(data, save_dir, model, \
																epochs = int(self.pretrain_epochs*calg_hpo["HP1"]), \
																rate_based_on_original = None, \
																rate_based_on_teacher = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																lma_numBins = calg_hpo["HP3"], \
																kd_params = (calg_hpo["HP5"], calg_hpo["HP4"]),
																use_logger = False).main()
				new_pre_model_dir = model_dir1
				parameter_amount, step_parameters_decreased = result_param1
				flops_amount, step_flops_decreased = result_flops1
				parameter_remain_rate = pre_parameter_remain_rate * (1.0-step_parameters_decreased)
				flops_remain_rate = pre_flops_remain_rate * (1.0-step_flops_decreased)
				step_score, step_score_increased_rate = result_metrics1['acc_top1'], result_metrics1['acc_top1_increased']
				acc_top1_rate = pre_acc_top1_rate * (result_metrics1['acc_top1_increased']+1)
				acc_top5_rate = pre_acc_top5_rate * (result_metrics1['acc_top5_increased']+1)
				self.logger.info('\t## acc_top1_increased: %s', str(result_metrics1['acc_top1_increased']))

				top1, top1_increased = step_score, acc_top1_rate-1
				top5, top5_increased = result_metrics1['acc_top5'], acc_top5_rate-1
				flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
				parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
			elif calg == "prune_C2": # HP1: fine_tune_epochs; HP2: prune_ratio's ratio; HP6: max_layer_prune_ratio; HP7: ea_epochs; HP8: ea_fine_tune_epochs; HP9: filter_importance_metric
				result_metrics2, result_param2, result_flops2, model_dir2 = prune_C2.LeGR(data, save_dir, model, \
																additional_fine_tune_epochs = int(self.pretrain_epochs*calg_hpo["HP1"]), \
																rate = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																max_prune_per_layer = calg_hpo["HP6"], \
																generations = int(self.pretrain_epochs*calg_hpo["HP7"]), \
																fine_tune_epochs = int(self.pretrain_epochs*calg_hpo["HP8"]),
																rank_type = calg_hpo["HP9"],
																use_logger = False).main()
				new_pre_model_dir = model_dir2
				parameter_amount, step_parameters_decreased = result_param2
				flops_amount, step_flops_decreased = result_flops2
				parameter_remain_rate = pre_parameter_remain_rate * (1.0-step_parameters_decreased)
				flops_remain_rate = pre_flops_remain_rate * (1.0-step_flops_decreased)
				step_score, step_score_increased_rate = result_metrics2['acc_top1'], result_metrics2['acc_top1_increased']
				acc_top1_rate = pre_acc_top1_rate * (result_metrics2['acc_top1_increased']+1)
				acc_top5_rate = pre_acc_top5_rate * (result_metrics2['acc_top5_increased']+1)
				self.logger.info('\t## acc_top1_increased: %s', str(result_metrics2['acc_top1_increased']))

				top1, top1_increased = step_score, acc_top1_rate-1
				top5, top5_increased = result_metrics2['acc_top5'], acc_top5_rate-1
				flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
				parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
			elif calg == "prune_C3": # HP1: fine_tune_epochs; HP2: prune_ratio's ratio; HP6: max_layer_prune_ratio
				result_metrics3, result_param3, result_flops3, model_dir3 = prune_C3.NetworkSlimming(data, save_dir, model, \
																fine_tune_epochs = int(self.pretrain_epochs*calg_hpo["HP1"]), \
																rate = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																max_prune_per_layer = calg_hpo["HP6"],
																use_logger = False).main()
				new_pre_model_dir = model_dir3
				parameter_amount, step_parameters_decreased = result_param3
				flops_amount, step_flops_decreased = result_flops3
				parameter_remain_rate = pre_parameter_remain_rate * (1.0-step_parameters_decreased)
				flops_remain_rate = pre_flops_remain_rate * (1.0-step_flops_decreased)
				step_score, step_score_increased_rate = result_metrics3['acc_top1'], result_metrics3['acc_top1_increased']
				acc_top1_rate = pre_acc_top1_rate * (result_metrics3['acc_top1_increased']+1)
				acc_top5_rate = pre_acc_top5_rate * (result_metrics3['acc_top5_increased']+1)
				self.logger.info('\t## acc_top1_increased: %s', str(result_metrics3['acc_top1_increased']))

				top1, top1_increased = step_score, acc_top1_rate-1
				top5, top5_increased = result_metrics3['acc_top5'], acc_top5_rate-1
				flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
				parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
			elif calg == "prune_C4": # HP2: prune_ratio's ratio; HP10: back_propagation_epochs; HP11: prune_update_frequency
				result_metrics4, result_param4, result_flops4, model_dir4 = prune_C4.SoftFilterPruning(data, save_dir, model, \
																rate = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																epochs = int(self.pretrain_epochs*calg_hpo["HP10"]), \
																epoch_prune = calg_hpo["HP11"],
																additional_fine_tune_epochs = max(int(self.pretrain_epochs/40.0),3),
																use_logger = False).main()
				new_pre_model_dir = model_dir4
				parameter_amount, step_parameters_decreased = result_param4
				flops_amount, step_flops_decreased = result_flops4
				parameter_remain_rate = pre_parameter_remain_rate * (1.0-step_parameters_decreased)
				flops_remain_rate = pre_flops_remain_rate * (1.0-step_flops_decreased)
				step_score, step_score_increased_rate = result_metrics4['acc_top1'], result_metrics4['acc_top1_increased']
				acc_top1_rate = pre_acc_top1_rate * (result_metrics4['acc_top1_increased']+1)
				acc_top5_rate = pre_acc_top5_rate * (result_metrics4['acc_top5_increased']+1)
				self.logger.info('\t## acc_top1_increased: %s', str(result_metrics4['acc_top1_increased']))

				top1, top1_increased = step_score, acc_top1_rate-1
				top5, top5_increased = result_metrics4['acc_top5'], acc_top5_rate-1
				flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
				parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
			elif calg == "prune_C5": # HP1: fine_tune_epochs; HP2: prune_ratio's ratio; HP12: global filter importance basis; HP13: local filter importance basis; HP14: KD_fine_tune_epochs; HP15: factor of auxiliary MSE losses
				result_metrics5, result_param5, result_flops5, model_dir5 = prune_C5.HOS(data, save_dir, model, \
																epochs_step4 = int(self.pretrain_epochs*calg_hpo["HP1"]), \
																rate = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																strategy = calg_hpo["HP12"], \
																metric = calg_hpo["HP13"], \
																epochs_step3 = int(self.pretrain_epochs*calg_hpo["HP14"]), \
																mse_factor = calg_hpo["HP15"],
																use_logger = False).main()
				new_pre_model_dir = model_dir5
				parameter_amount, step_parameters_decreased = result_param5
				flops_amount, step_flops_decreased = result_flops5
				parameter_remain_rate = pre_parameter_remain_rate * (1.0-step_parameters_decreased)
				flops_remain_rate = pre_flops_remain_rate * (1.0-step_flops_decreased)
				step_score, step_score_increased_rate = result_metrics5['acc_top1'], result_metrics5['acc_top1_increased']
				acc_top1_rate = pre_acc_top1_rate * (result_metrics5['acc_top1_increased']+1)
				acc_top5_rate = pre_acc_top5_rate * (result_metrics5['acc_top5_increased']+1)
				self.logger.info('\t## acc_top1_increased: %s', str(result_metrics5['acc_top1_increased']))

				top1, top1_increased = step_score, acc_top1_rate-1
				top5, top5_increased = result_metrics5['acc_top5'], acc_top5_rate-1
				flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
				parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
			elif calg == "prune_C7": # HP1: fine_tune_epochs; HP2: decomposition_ratio's ratio; HP16: factor of auxiliary losses; HP17: fine tune auxiliary loss
				result_metrics7, result_param7, result_flops7, model_dir7 = prune_C7.LFB(data, save_dir, self.model_name, \
																epochs = int(self.pretrain_epochs*calg_hpo["HP1"]), \
																rate = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																losses = str(calg_hpo["HP16"])+"*"+calg_hpo["HP17"],
																use_logger = False).main()
				new_pre_model_dir = model_dir7
				parameter_amount, step_parameters_decreased = result_param7
				flops_amount, step_flops_decreased = result_flops7
				parameter_remain_rate = pre_parameter_remain_rate * (1.0-step_parameters_decreased)
				flops_remain_rate = pre_flops_remain_rate * (1.0-step_flops_decreased)
				step_score, step_score_increased_rate = result_metrics7['acc_top1'], result_metrics7['acc_top1_increased']
				acc_top1_rate = pre_acc_top1_rate * (result_metrics7['acc_top1_increased']+1)
				acc_top5_rate = pre_acc_top5_rate * (result_metrics7['acc_top5_increased']+1)
				self.logger.info('\t## acc_top1_increased: %s', str(result_metrics7['acc_top1_increased']))

				top1, top1_increased = step_score, acc_top1_rate-1
				top5, top5_increased = result_metrics7['acc_top5'], acc_top5_rate-1
				flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
				parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
		except:
			self.logger.info('\t## time used: %.2f s', time.time()-start_time)
			self.logger.info('\t## evaluation finished (FAILED), step_score: 0, compression_rate: 0, flops_decreased_rate: 0, parameter_amount: -M, flops_amount: -G')
			self.logger.info('\t## evaluation finished (FAILED), top1/top1_increased: 0 / -1, top5/top5_increased: 0 / -1, flops/flops_decreased: - / 0, parameter/parameter_decreased: - / 0')
			step_info = [0,0,0]
			score_info = [0, 0, 0, '0M', '0G']
			new_pre_info = ["FAILED", "FAILED", "FAILED", "FAILED", "FAILED"]
			table_infos = {
					"top1/top1_increased": [0, -1], 
					"top5/top5_increased": [0, -1],
					"flops/flops_decreased": ["-", 0], 
					"parameter/parameter_decreased": ["-", 0]
					}
			self.logger.info('\t## evaluation finished (FAILED), model_dir: FAILED')
			self.logger.info('\t## evaluation finished (FAILED), step_info: %s, score_info: %s, new_pre_info: %s, table_infos: %s', step_info, score_info, new_pre_info, str(table_infos))
			return [step_info, score_info, new_pre_info, table_infos]

		compression_rate = 1.0-parameter_remain_rate
		new_pre_parameter_remain_rate = parameter_remain_rate
		new_pre_flops_remain_rate = flops_remain_rate
		new_pre_acc_top1_rate = acc_top1_rate
		new_pre_acc_top5_rate = acc_top5_rate
		new_pre_info = [new_pre_model_dir, new_pre_parameter_remain_rate, new_pre_flops_remain_rate, new_pre_acc_top1_rate, new_pre_acc_top5_rate]

		step_info = [step_parameters_decreased, step_flops_decreased, step_score_increased_rate]

		if math.isinf(step_score_increased_rate) or math.isinf(step_flops_decreased):
			self.logger.info('\t## time used: %.2f s', time.time()-start_time)
			self.logger.info('\t## evaluation finished (FAILED), step_score: 0, compression_rate: 0, flops_decreased_rate: 0, parameter_amount: -M, flops_amount: -G')
			self.logger.info('\t## evaluation finished (FAILED), top1/top1_increased: 0 / -1, top5/top5_increased: 0 / -1, flops/flops_decreased: - / 0, parameter/parameter_decreased: - / 0')
			step_info = [0,0,0]
			score_info = [0, 0, 0, '0M', '0G']
			new_pre_info = ["FAILED", "FAILED", "FAILED", "FAILED", "FAILED"]
			table_infos = {
					"top1/top1_increased": [0, -1], 
					"top5/top5_increased": [0, -1],
					"flops/flops_decreased": ["-", 0], 
					"parameter/parameter_decreased": ["-", 0]
					}
			self.logger.info('\t## evaluation finished (FAILED), model_dir: FAILED')
			self.logger.info('\t## evaluation finished (FAILED), step_info: %s, score_info: %s, new_pre_info: %s, table_infos: %s', step_info, score_info, new_pre_info, str(table_infos))
			return [step_info, score_info, new_pre_info, table_infos]

		flops_decreased_rate = 1-flops_remain_rate
		score_info = [step_score, compression_rate, flops_decreased_rate, parameter_amount, flops_amount]

		table_infos = {
					"top1/top1_increased": [top1, top1_increased], 
					"top5/top5_increased": [top5, top5_increased],
					"flops/flops_decreased": [flops, flops_decreased], 
					"parameter/parameter_decreased": [parameter, parameter_decreased]
					}

		self.logger.info('\t## time used: %.2f s', time.time()-start_time)
		self.logger.info('\t## evaluation finished, step_score: %.2f, compression_rate: %.2f, flops_decreased_rate:%.2f, flops_amount: %s, parameter_amount: %s', step_score, compression_rate, flops_decreased_rate, flops_amount, parameter_amount)
		self.logger.info('\t## evaluation finished, top1/top1_increased: %.4f / %.4f, top5/top5_increased: %.4f / %.4f, flops/flops_decreased: %s / %.4f, parameter/parameter_decreased: %s / %.4f', top1, top1_increased, top5, top5_increased, flops, flops_decreased, parameter, parameter_decreased)
		self.logger.info('\t## evaluation finished, model_dir: %s', str(new_pre_model_dir))
		self.logger.info('\t## evaluation finished, step_info: %s, score_info: %s, new_pre_info: %s, table_infos: %s', step_info, score_info, new_pre_info, str(table_infos))
		return [step_info, score_info, new_pre_info, table_infos]
