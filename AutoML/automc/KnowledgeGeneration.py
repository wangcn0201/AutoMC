import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import random
import pickle


class KnowledgeGeneration(object):
	def __init__(self, space, cstartegies, knowledge_path, logger):
		super(KnowledgeGeneration, self).__init__()

		self.space = space
		self.cstartegies = cstartegies
		self.knowledge_path = knowledge_path
		self.logger = logger
		return

	def get_real_strategy(self, codesring):
		left_num = 0
		i = 0
		while i < len(codesring):
			if codesring[i] == '[':
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
				break
			else:
				break
		return new_item
		
	def GenerateKG(self, space, cstartegies, knowledge_path):
		reverse_entity_mapping, entity_mapping, relation_mapping, kg_info = [], {}, {}, []
		complete_hpo_dict = {}
		complete_cstartegies = []

		# add cstrategy nodes 
		for i in range(len(cstartegies)):
			entity_mapping["cstrategy: "+str(i)] = i + 1
			reverse_entity_mapping.append("cstrategy: "+str(i))
			if cstartegies[i] not in complete_cstartegies:
				complete_cstartegies.append(list(cstartegies[i]))

		# add calg and hpo nodes 
		sum_item_id = len(cstartegies)
		for calg in space.keys():
			if "calg: "+str(calg) not in entity_mapping.keys():
				entity_mapping["calg: "+str(calg)] = sum_item_id + 1
				reverse_entity_mapping.append("calg: "+str(calg))
				sum_item_id += 1

			calg_hpo_dict = space[calg]
			for calg_hpo in calg_hpo_dict.keys():
				if "hpo: "+str(calg_hpo) not in entity_mapping.keys():
					entity_mapping["hpo: "+str(calg_hpo)] = sum_item_id + 1
					reverse_entity_mapping.append("hpo: "+str(calg_hpo))
					sum_item_id += 1
				if calg_hpo not in complete_hpo_dict.keys():
					complete_hpo_dict[calg_hpo] = []
				for hpo_value in calg_hpo_dict[calg_hpo]:
					if "hpo:"+str(calg_hpo)+": "+str(hpo_value) not in entity_mapping.keys():
						entity_mapping["hpo:"+str(calg_hpo)+": "+str(hpo_value)] = sum_item_id + 1
						reverse_entity_mapping.append("hpo:"+str(calg_hpo)+": "+str(hpo_value))
						sum_item_id += 1
					if hpo_value not in complete_hpo_dict[calg_hpo]:
						complete_hpo_dict[calg_hpo].append(hpo_value)

		# add calg's technique nodes 
		with open("automc/kb_calg_info.txt", "r") as f:
			kb_calg_info = f.readlines()
		for i in range(1, len(kb_calg_info)):
			technique = kb_calg_info[i].split("\t")[1]
			if "tec: "+str(technique) not in entity_mapping.keys():
				entity_mapping["tec: "+str(technique)] = sum_item_id + 1
				reverse_entity_mapping.append("tec: "+str(technique))
				sum_item_id += 1

		# add nodes from exp info
		with open("automc/kb_exp_info.txt", "r") as f:
			kb_exp_info = f.readlines()
		added_cstartegies = []
		for i in range(1, len(kb_exp_info)):
			cstrategy = self.get_real_strategy(kb_exp_info[i].split("\t")[3])
			if cstrategy not in cstartegies and cstrategy not in added_cstartegies:
				entity_mapping["cstrategy: "+str(len(cstartegies)+len(added_cstartegies))] = sum_item_id + 1
				reverse_entity_mapping.append("cstrategy: "+str(len(cstartegies)+len(added_cstartegies)))
				complete_cstartegies.append(list(cstrategy))
				cstrategy_id = sum_item_id + 1
				added_cstartegies.append(cstrategy)
				sum_item_id += 1

				calg, hpo_dict = cstrategy
				calg_id = entity_mapping["calg: "+str(calg)]
				kg_info.append([cstrategy_id, 0, calg_id])

				for hpo_name in hpo_dict.keys():
					hpo_id = entity_mapping["hpo: "+str(hpo_name)]
					hpo_value = hpo_dict[hpo_name]
					if "hpo:"+str(hpo_name)+": "+str(hpo_value) not in entity_mapping.keys():
						entity_mapping["hpo:"+str(hpo_name)+": "+str(hpo_value)] = sum_item_id + 1
						reverse_entity_mapping.append("hpo:"+str(hpo_name)+": "+str(hpo_value))
						hpo_value_id = sum_item_id + 1
						kg_info.append([hpo_id, 3, hpo_value_id])
						sum_item_id += 1
					else:
						hpo_value_id = entity_mapping["hpo:"+str(hpo_name)+": "+str(hpo_value)]
					kg_info.append([cstrategy_id, 1, hpo_value_id])

					if hpo_value not in complete_hpo_dict[hpo_name]:
						complete_hpo_dict[hpo_name].append(hpo_value)


		# define relation_mapping
		relation_mapping = {
			"cstrategy-calg": 0,
			"cstrategy-hpo_value": 1,
			"calg-hpo": 2,
			"hpo-hpo_value": 3,
			"calg-tec": 4,
			"hpo_value-hpo_value": 5
		}

		# generate kg_info's "calg-hpo" and "hpo-hpo_value" parts
		for calg in space.keys():
			calg_id = entity_mapping["calg: "+str(calg)]
			for calg_hpo in calg_hpo_dict.keys():
				hpo_id = entity_mapping["hpo: "+str(calg_hpo)]
				kg_info.append([calg_id, 2, hpo_id])
				for hpo_value in calg_hpo_dict[calg_hpo]:
					hpo_value_id = entity_mapping["hpo:"+str(calg_hpo)+": "+str(hpo_value)]
					kg_info.append([hpo_id, 3, hpo_value_id])

		# generate kg_info's "calg-tec" part
		for i in range(1, len(kb_calg_info)):
			calg, technique = kb_calg_info[i].split("\t")
			calg_id = entity_mapping["calg: "+str(calg)]
			technique_id = entity_mapping["tec: "+str(technique)]
			kg_info.append([calg_id, 4, technique_id])

		# generate kg_info's "cstrategy-calg" and "cstrategy-hpo_value" parts
		for i in range(len(cstartegies)):
			calg, content = cstartegies[i]
			calg_id = entity_mapping["calg: "+str(calg)]
			cstrategy_id = entity_mapping["cstrategy: "+str(i)]
			kg_info.append([cstrategy_id, 0, calg_id])
			for calg_hpo in content.keys():
				hpo_value = content[calg_hpo]
				hpo_value_id = entity_mapping["hpo:"+str(calg_hpo)+": "+str(hpo_value)]
				kg_info.append([cstrategy_id, 1, hpo_value_id])

		# generate kg_info's "hpo_value-hpo_value" parts
		for hpo_name in complete_hpo_dict.keys():
			complete_hpo_dict[hpo_name].sort(reverse=False)
			for i in range(len(complete_hpo_dict[hpo_name])-1):
				hpo_value1 = complete_hpo_dict[hpo_name][i]
				hpo_value_id1 = entity_mapping["hpo:"+str(hpo_name)+": "+str(hpo_value1)]
				hpo_value2 = complete_hpo_dict[hpo_name][i+1]
				hpo_value_id2 = entity_mapping["hpo:"+str(hpo_name)+": "+str(hpo_value2)]
				kg_info.append([hpo_value_id1, 5, hpo_value_id2])


		# write all kg information to the files
		with open(knowledge_path + '/kg_reverse_entity_mapping.pkl', 'wb') as file:
			pickle.dump(reverse_entity_mapping, file)
		with open(knowledge_path + '/kg_entity_mapping.pkl', 'wb') as file:
			pickle.dump(entity_mapping, file)
		with open(knowledge_path + '/kg_relation_mapping.pkl', 'wb') as file:
			pickle.dump(relation_mapping, file)
		with open(knowledge_path + '/kg_kg_info.pkl', 'wb') as file:
			pickle.dump(kg_info, file)

		#print(reverse_entity_mapping[:3])
		#print(list(entity_mapping.keys())[:3])
		#print(kg_info[:3])
		#return
		return entity_mapping, complete_cstartegies

	def GenerateEXP(self, cstartegies, entity_mapping, complete_cstartegies, knowledge_path):
		# task feature
		# data feature: 类别数、数据量、输入维度、平均每个类别的个数均值、个类别的个数方差
		# model feature: 模型性能、loss值、参数量、flops值

		data_infos, model_infos = {}, {}
		exp_infos_1 = {
			"cstrategy_infos": [], 
			"task_infos": [], 
			"performance_infos": []
		}
		exp_infos_2 = {
			"cstrategy_infos": [], 
			"task_infos": [], 
			"performance_infos": []
		}

		with open("automc/kb_data_info.txt", "r") as f:
			kb_data_info = f.readlines()
		for i in range(1, len(kb_data_info)):
			data_name, class_num, _, image_size, image_channel, \
					avg_data_num_per_class = kb_data_info[i].split("\t")
			data_info = [int(class_num), int(image_size), int(image_channel), int(avg_data_num_per_class)]
			data_infos[data_name] = list(data_info)

		with open("automc/kb_model_info.txt", "r") as f:
			kb_model_info = f.readlines()
		for i in range(1, len(kb_model_info)):
			source_name, model_name, data_name, top1_acc, parameter_amount, \
					flops_amount = kb_model_info[i].split("\t")
			model_info = [float(top1_acc), float(parameter_amount), float(flops_amount)]
			model_infos[source_name+"+"+data_name+"+"+model_name] = list(model_info)


		with open("automc/kb_exp_info.txt", "r") as f:
			kb_exp_info = f.readlines()
		for i in range(1, len(kb_exp_info)):
			source_name, model_name, data_name, cstartegy_name, flops_decrease_ratio, score_increase_ratio = kb_exp_info[i].split("\t")
			cstartegy_name, score_increase_ratio = self.get_real_strategy(cstartegy_name), float(score_increase_ratio)

			cstrategy_id = entity_mapping["cstrategy: "+str(complete_cstartegies.index(cstartegy_name))]

			data_info = data_infos[data_name]
			model_info = model_infos[source_name+"+"+data_name+"+"+model_name]
			task_info = list(data_info+model_info)

			if flops_decrease_ratio != "None":
				performance_info = [float(flops_decrease_ratio), float(score_increase_ratio)]
				flag = 2
			else:
				performance_info = [float(score_increase_ratio)]
				flag = 1
			
			if flag == 1:
				exp_infos_1["cstrategy_infos"].append(cstrategy_id)
				exp_infos_1["task_infos"].append(task_info)
				exp_infos_1["performance_infos"].append(performance_info)
			elif flag == 2:
				exp_infos_2["cstrategy_infos"].append(cstrategy_id)
				exp_infos_2["task_infos"].append(task_info)
				exp_infos_2["performance_infos"].append(performance_info)

		# write all exp information to the files
		with open(knowledge_path + '/exp_infos_1.pkl', 'wb') as file:
			pickle.dump(exp_infos_1, file)
		with open(knowledge_path + '/exp_infos_2.pkl', 'wb') as file:
			pickle.dump(exp_infos_2, file)
		return 

	def main(self):
		entity_mapping, complete_cstartegies = self.GenerateKG(self.space, self.cstartegies, self.knowledge_path)
		self.logger.info("* GenerateKG finished")
		self.GenerateEXP(self.cstartegies, entity_mapping, complete_cstartegies, self.knowledge_path)
		self.logger.info("* GenerateEXP finished")
		return 
