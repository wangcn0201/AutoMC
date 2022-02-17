import os
import sys
import shutil
import logging
import torch
import numpy as np
import torch.backends.cudnn as cudnn
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))),"CAlgs")) 
import prune_C1, prune_C2, prune_C3, prune_C4, prune_C5, prune_C7, train
import time
import copy


class SchemeEvaluation(object):
	def __init__(self, config, logging_path, logger, final_finetune_epoch_total):
		self.epoch_total = final_finetune_epoch_total
		task_name = config["task_name"]
		model_name, data_name, target_compression_rate = task_name.split("+")

		self.data = {'dir':'../CAlgs/data', 'name':data_name}
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
		self.model = {'dir': '../CAlgs/trained_models/{}/{}.pth.tar'.format(data_name, model_name), 'name': model_name}
		self.rate = float(target_compression_rate) # delected rate

		self.logging_path = logging_path
		self.pretrain_epochs = config["pretrain_epochs"]
		self.logger = logger
		return

	def main(self, scheme_code):
		self.logger.info('\t## evaluation scheme_code: %s', str(scheme_code))
		data = self.data
		model = copy.deepcopy(self.model)
		save_dir = self.logging_path + "/SchemeEvaluation_LOG/"
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)
		else:
			cmd = "rm -rf "+str(save_dir)
			os.system(cmd)
			os.mkdir(save_dir)

		init_rate = self.rate
		scheme_score, parameter_remain_rate, flops_remain_rate = None, 1, 1
		acc_top1_rate, acc_top5_rate = 1, 1
		# try:
		for i in range(len(scheme_code)):
				self.logger.info('\t## iter: %d, scheme_code_i: %s', i, str(scheme_code[i]))
				start_time = time.time()
				calg, calg_hpo = scheme_code[i]
				self.logger.info('\t## init_rate: %.4f, rate: %.4f', init_rate, max(1-init_rate*calg_hpo["HP2"],0.1))
				self.logger.info('\t## input model: %s', str(model))
				if calg == "prune_C1": # HP1: fine_tune_epochs; HP2: prune_ratio's ratio; HP3: LMA_segment_num; HP4: distillation_temperature_factor; HP5: distillation_alpha_factor 
					result_metrics1, result_param1, result_flops1, model_dir1 = prune_C1.knowledge_distillation(data, save_dir, model, \
																epochs = int(self.pretrain_epochs*calg_hpo["HP1"]), \
																rate_based_on_original = None, \
																rate_based_on_teacher = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																lma_numBins = calg_hpo["HP3"], \
																kd_params = (calg_hpo["HP5"], calg_hpo["HP4"]),
																use_logger = self.logger).main()
					
					model['dir'] = model_dir1
					parameter_amount = result_param1[0]
					flops_amount = result_flops1[0]
					parameter_remain_rate *= (1.0-result_param1[1])
					flops_remain_rate *= (1.0-result_flops1[1])
					scheme_score = result_metrics1['acc_top1']
					acc_top1_rate *= (result_metrics1['acc_top1_increased']+1)
					acc_top5_rate *= (result_metrics1['acc_top5_increased']+1)

					top1, top1_increased = scheme_score, acc_top1_rate-1
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
																use_logger = self.logger).main()
					model['dir'] = model_dir2
					parameter_amount = result_param2[0]
					flops_amount = result_flops2[0]
					parameter_remain_rate *= (1.0-result_param2[1])
					flops_remain_rate *= (1.0-result_flops2[1])
					scheme_score = result_metrics2['acc_top1']
					acc_top1_rate *= (result_metrics2['acc_top1_increased']+1)
					acc_top5_rate *= (result_metrics2['acc_top5_increased']+1)

					top1, top1_increased = scheme_score, acc_top1_rate-1
					top5, top5_increased = result_metrics2['acc_top5'], acc_top5_rate-1
					flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
					parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
				elif calg == "prune_C3": # HP1: fine_tune_epochs; HP2: prune_ratio's ratio; HP6: max_layer_prune_ratio
					result_metrics3, result_param3, result_flops3, model_dir3 = prune_C3.NetworkSlimming(data, save_dir, model, \
																fine_tune_epochs = int(self.pretrain_epochs*calg_hpo["HP1"]), \
																rate = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																max_prune_per_layer = calg_hpo["HP6"],
																use_logger = self.logger).main()
					model['dir'] = model_dir3
					parameter_amount = result_param3[0]
					flops_amount = result_flops3[0]
					parameter_remain_rate *= (1.0-result_param3[1])
					flops_remain_rate *= (1.0-result_flops3[1])
					scheme_score = result_metrics3['acc_top1']
					acc_top1_rate *= (result_metrics3['acc_top1_increased']+1)
					acc_top5_rate *= (result_metrics3['acc_top5_increased']+1)

					top1, top1_increased = scheme_score, acc_top1_rate-1
					top5, top5_increased = result_metrics3['acc_top5'], acc_top5_rate-1
					flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
					parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
				elif calg == "prune_C4": # HP2: prune_ratio's ratio; HP10: back_propagation_epochs; HP11: prune_update_frequency
					result_metrics4, result_param4, result_flops4, model_dir4 = prune_C4.SoftFilterPruning(data, save_dir, model, \
																rate = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																epochs = int(self.pretrain_epochs*calg_hpo["HP10"]), \
																epoch_prune = calg_hpo["HP11"],
																additional_fine_tune_epochs = max(int(self.pretrain_epochs/40.0),3),
																use_logger = self.logger).main()
					model['dir'] = model_dir4
					parameter_amount = result_param4[0]
					flops_amount = result_flops4[0]
					parameter_remain_rate *= (1.0-result_param4[1])
					flops_remain_rate *= (1.0-result_flops4[1])
					scheme_score = result_metrics4['acc_top1']
					acc_top1_rate *= (result_metrics4['acc_top1_increased']+1)
					acc_top5_rate *= (result_metrics4['acc_top5_increased']+1)

					top1, top1_increased = scheme_score, acc_top1_rate-1
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
																use_logger = self.logger).main()
					model['dir'] = model_dir5
					parameter_amount = result_param5[0]
					flops_amount = result_flops5[0]
					parameter_remain_rate *= (1.0-result_param5[1])
					flops_remain_rate *= (1.0-result_flops5[1])
					scheme_score = result_metrics5['acc_top1']
					acc_top1_rate *= (result_metrics5['acc_top1_increased']+1)
					acc_top5_rate *= (result_metrics5['acc_top5_increased']+1)

					top1, top1_increased = scheme_score, acc_top1_rate-1
					top5, top5_increased = result_metrics5['acc_top5'], acc_top5_rate-1
					flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
					parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
					break
				elif calg == "prune_C7": # HP1: fine_tune_epochs; HP2: decomposition_ratio's ratio; HP16: factor of auxiliary losses; HP17: fine tune auxiliary loss
					result_metrics7, result_param7, result_flops7, model_dir7 = prune_C7.LFB(data, save_dir, self.model_name, \
																epochs = int(self.pretrain_epochs*calg_hpo["HP1"]), \
																rate = max(1-init_rate*calg_hpo["HP2"], 0.1), \
																losses = str(calg_hpo["HP16"])+"*"+calg_hpo["HP17"],
																use_logger = self.logger).main()
					model['dir'] = model_dir7
					parameter_amount = result_param7[0]
					flops_amount = result_flops7[0]
					parameter_remain_rate *= (1.0-result_param7[1])
					flops_remain_rate *= (1.0-result_flops7[1])
					scheme_score = result_metrics7['acc_top1']
					acc_top1_rate *= (result_metrics7['acc_top1_increased']+1)
					acc_top5_rate *= (result_metrics7['acc_top5_increased']+1)

					top1, top1_increased = scheme_score, acc_top1_rate-1
					top5, top5_increased = result_metrics7['acc_top5'], acc_top5_rate-1
					flops, flops_decreased = flops_amount, 1.0-flops_remain_rate
					parameter, parameter_decreased = parameter_amount, 1.0-parameter_remain_rate
					break

				init_rate = self.rate/parameter_remain_rate
				self.logger.info('\t## iter: %d, strategy_code: %s', i, str(scheme_code[i]))
				self.logger.info('\t## iter: %d, current scheme_score: %.2f, compression_rate: %.2f, flops_decreased_rate: %.2f, parameter_amount: %s, flops_amount: %s', i, scheme_score, 1.0-parameter_remain_rate, 1.0-flops_remain_rate, parameter_amount, flops_amount)
				self.logger.info('\t## iter: %d, current top1/top1_increased: %.4f / %.4f, top5/top5_increased: %.4f / %.4f, flops/flops_decreased: %s / %.4f, parameter/parameter_decreased: %s / %.4f', i, top1, top1_increased, top5, top5_increased, flops, flops_decreased, parameter, parameter_decreased)
				self.logger.info('\t## iter: %d, current model_dir: %s', i, str(model['dir']))
				self.logger.info('\t## time used: %.2f s', time.time()-start_time)
		# except:
		# 	self.logger.info('\t## evaluation finished (FAILED), scheme_score: 0, compression_rate: 0, flops_decreased_rate: 0, parameter_amount: -M, flops_amount: -G')
		# 	self.logger.info('\t## evaluation finished (FAILED), top1/top1_increased: 0 / -1, top5/top5_increased: 0 / -1, flops/flops_decreased: - / 0, parameter/parameter_decreased: - / 0')
		# 	table_infos = {
		# 			"top1/top1_increased": [0, -1], 
		# 			"top5/top5_increased": [0, -1],
		# 			"flops/flops_decreased": ["-", 0], 
		# 			"parameter/parameter_decreased": ["-", 0]
		# 			}
		# 	return [0, 0, 0, '0M', '0G'], table_infos

		compression_rate = 1.0-parameter_remain_rate
		flops_decreased_rate = 1.0-flops_remain_rate
		self.logger.info('\t## evaluation finished, scheme_score: %.2f, compression_rate: %.2f, flops_decreased_rate: %.2f, parameter_amount: %s, flops_amount: %s', scheme_score, compression_rate, flops_decreased_rate, parameter_amount, flops_amount)
		self.logger.info('\t## evaluation finished, top1/top1_increased: %.4f / %.4f, top5/top5_increased: %.4f / %.4f, flops/flops_decreased: %s / %.4f, parameter/parameter_decreased: %s / %.4f', top1, top1_increased, top5, top5_increased, flops, flops_decreased, parameter, parameter_decreased)
		table_infos = {
					"top1/top1_increased": [top1, top1_increased], 
					"top5/top5_increased": [top5, top5_increased],
					"flops/flops_decreased": [flops, flops_decreased], 
					"parameter/parameter_decreased": [parameter, parameter_decreased]
					}

		if self.epoch_total != -1:
			self.logger.info('## entering additional finetune...')
			if calg == "prune_C1":
				epochs = self.epoch_total - int(self.pretrain_epochs*calg_hpo["HP1"])
			elif calg == "prune_C2":
				epochs = self.epoch_total - int(self.pretrain_epochs*calg_hpo["HP1"])
			elif calg == "prune_C3":
				epochs = self.epoch_total - int(self.pretrain_epochs*calg_hpo["HP1"])
			elif calg == "prune_C4":
				epochs = self.epoch_total - int(self.pretrain_epochs*calg_hpo["HP10"])
			elif calg == "prune_C5":
				epochs = self.epoch_total - int(self.pretrain_epochs*calg_hpo["HP1"]) - int(self.pretrain_epochs*calg_hpo["HP14"])
			elif calg == "prune_C7":
				epochs = self.epoch_total - int(self.pretrain_epochs*calg_hpo["HP1"])

			self.logger.info('## remaining epochs is {}'.format(epochs))
			if epochs > 0:
				acc_dict, _ = train.Train(data, save_dir, model, logger=self.logger,
					epochs=epochs, lr=1e-3, lr_sche='MultiStepLR', return_file=False, get_relative_acc=True, use_logger=True).main()
				scheme_score = acc_dict['acc_top1']

				acc_top1_rate *= (acc_dict['acc_top1_increased']+1)
				acc_top5_rate *= (acc_dict['acc_top5_increased']+1)
				top1, top1_increased = acc_dict['acc_top1'], acc_top1_rate-1
				top5, top5_increased = acc_dict['acc_top5'], acc_top5_rate-1
				self.logger.info('\t## additional fine tune finished, scheme_score: %.2f, compression_rate: %.2f, flops_decreased_rate: %.2f, parameter_amount: %s, flops_amount: %s', scheme_score, compression_rate, flops_decreased_rate, parameter_amount, flops_amount)
				self.logger.info('\t## additional fine tune finished, top1/top1_increased: %.4f / %.4f, top5/top5_increased: %.4f / %.4f, flops/flops_decreased: %s / %.4f, parameter/parameter_decreased: %s / %.4f', top1, top1_increased, top5, top5_increased, flops, flops_decreased, parameter, parameter_decreased)
				table_infos["top1/top1_increased"] = [top1, top1_increased]
				table_infos["top5/top5_increased"] = [top5, top5_increased]
			else:
				self.logger.info('\t## additional fine tune finished, scheme_score: %.2f, compression_rate: %.2f, flops_decreased_rate: %.2f, parameter_amount: %s, flops_amount: %s', scheme_score, compression_rate, flops_decreased_rate, parameter_amount, flops_amount)
				self.logger.info('\t## additional fine tune finished, top1/top1_increased: %.4f / %.4f, top5/top5_increased: %.4f / %.4f, flops/flops_decreased: %s / %.4f, parameter/parameter_decreased: %s / %.4f', top1, top1_increased, top5, top5_increased, flops, flops_decreased, parameter, parameter_decreased)
		return [scheme_score, compression_rate, flops_decreased_rate, parameter_amount, flops_amount], table_infos
