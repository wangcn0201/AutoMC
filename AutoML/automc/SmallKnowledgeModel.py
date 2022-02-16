import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import random
import pickle
import time


class SmallKnowledgeModel(torch.nn.Module):
	def __init__(self, config, cstrategies_path, knowledge_path, logger, KG=True, EXP=True, embedding_dim=32, output_dim=2, hidden_dim=8, dropout=0.5):
		super(SmallKnowledgeModel, self).__init__()

		self.KG = KG
		self.EXP = EXP
		self.logger = logger

		self.cstrategies_num, self.cstrategies = self.GetCstrategyIndex(cstrategies_path)

		self.head_ids, self.tail_ids, self.relation_ids, numTriple, node_num, relation_num = self.GetKGInfo(knowledge_path)
		self.exp_infos_1, self.exp_infos_2, self.task_info_norm_value, task_info_dim = self.GetExpInfo(knowledge_path)

		self.r_embeddings = nn.Embedding(relation_num, embedding_dim)
		self.i_embeddings = nn.Embedding(node_num+1, embedding_dim)

		self.lin = nn.Sequential(
					Linear(embedding_dim+task_info_dim, hidden_dim),
					nn.ReLU(),
					Linear(hidden_dim, output_dim),
					) 
		self.dropout = dropout

		self.epochs = config['our_kmodel_epochs']
		self.batch_size = config['our_kmodel_batch_size']
		self.learning_rate = config['our_kmodel_learning_rate']
		return
		
	def GetCstrategyIndex(self, cstrategies_path):
		with open(cstrategies_path, "r") as f:
			cstrategies = eval(f.readline())
		cstrategies_num = len(cstrategies)
		return cstrategies_num, cstrategies

	def GetKGInfo(self, knowledge_path):
		self.logger.info("* GetKGInfo begin")
		head_ids, tail_ids, relation_ids = [], [], []
		numTriple, node_num, relation_num = 0, 0, 0

		with open(knowledge_path + '/small_kg_reverse_entity_mapping.pkl', 'rb') as file:
			reverse_entity_mapping = pickle.load(file)
		with open(knowledge_path + '/small_kg_entity_mapping.pkl', 'rb') as file:
			entity_mapping = pickle.load(file)
		with open(knowledge_path + '/small_kg_relation_mapping.pkl', 'rb') as file:
			relation_mapping = pickle.load(file)
		with open(knowledge_path + '/small_kg_kg_info.pkl', 'rb') as file:
			kg_info = pickle.load(file)

		numTriple, node_num, relation_num = len(kg_info), len(entity_mapping.keys()), len(relation_mapping.keys())

		for i in range(numTriple):
			head, relation, tail = kg_info[i]
			neg_heads = np.random.randint(1, node_num+1)
			while (neg_heads, relation, tail) in kg_info:
				neg_heads = np.random.randint(1, node_num+1)
			head_id = np.array([head, head, head, neg_heads])
			neg_tails = np.random.randint(1, node_num+1)
			while (head, relation, neg_tails) in kg_info:
				neg_tails = np.random.randint(1, node_num+1)
			tail_id = np.array([tail, tail, neg_tails, tail])
			relation_id = np.array([relation] * 4)
			head_ids.append(head_id)
			tail_ids.append(tail_id)
			relation_ids.append(relation_id)
		self.logger.info("* GetKGInfo finished")
		return head_ids, tail_ids, relation_ids, numTriple, node_num, relation_num

	def GetExpInfo(self, knowledge_path):
		self.logger.info("* GetExpInfo begin")
		exp_infos_1, exp_infos_2, task_info_norm_value, task_info_dim = None, None, None, 0

		with open(knowledge_path + '/small_exp_infos_1.pkl', 'rb') as file:
			exp_infos_1 = pickle.load(file)
		with open(knowledge_path + '/small_exp_infos_2.pkl', 'rb') as file:
			exp_infos_2 = pickle.load(file)

		task_infos_1 = np.array(exp_infos_1["task_infos"])
		task_infos_2 = np.array(exp_infos_2["task_infos"])
		if len(task_infos_2) == 0:
			task_info_norm_value = task_infos_1
		elif len(task_infos_1) == 0:
			task_info_norm_value = task_infos_2
		else:
			task_info_norm_value = np.concatenate((task_infos_1,task_infos_2),axis=0).mean(axis=0)
		task_info_dim = task_info_norm_value.shape[1]
		#print(task_info_norm_value)

		cstrategy_ids = np.array(exp_infos_1["cstrategy_infos"])
		if len(task_infos_1) == 0:
			task_infos_normed = task_infos_1
		else:
			task_infos_normed = task_infos_1 / task_info_norm_value
		performance_infos = np.array(exp_infos_1["performance_infos"])
		exp_infos_1 = {
			"cstrategy_ids": cstrategy_ids,
			"task_infos": task_infos_normed,
			"performance_infos": performance_infos
		}

		cstrategy_ids = np.array(exp_infos_2["cstrategy_infos"])
		if len(task_infos_2) == 0:
			task_infos_normed = task_infos_2
		else:
			task_infos_normed = task_infos_2 / task_info_norm_value
		performance_infos = np.array(exp_infos_2["performance_infos"])
		exp_infos_2 = {
			"cstrategy_ids": cstrategy_ids,
			"task_infos": task_infos_normed,
			"performance_infos": performance_infos
		}
		self.logger.info("* task_info_norm_value: %s, task_info_dim: %s", str(task_info_norm_value), str(task_info_dim))
		self.logger.info("* GetExpInfo finished")
		return exp_infos_1, exp_infos_2, task_info_norm_value, task_info_dim

	def get_parameters(self):
		return list(self.parameters())

	def KG_forward(self, batchId):
		# get batch data
		start_idx = batchId * self.batch_size
		end_idx = start_idx + self.batch_size

		if end_idx > self.kg_trainsize:
			end_idx = self.kg_trainsize
			start_idx = end_idx - self.batch_size

		if end_idx == start_idx:
			start_idx = 0
			end_idx = start_idx + self.batch_size

		head_ids = self.head_ids[start_idx:end_idx]
		tail_ids = self.tail_ids[start_idx:end_idx]
		relation_ids = self.relation_ids[start_idx:end_idx]

		# get prediction
		relation_vectors = self.r_embeddings(torch.Tensor(relation_ids).long().cuda())
		head_vectors = self.i_embeddings(torch.Tensor(head_ids).long().cuda())
		tail_vectors = self.i_embeddings(torch.Tensor(tail_ids).long().cuda())

		prediction = -((head_vectors + relation_vectors - tail_vectors) ** 2).sum(-1)

		# get kg sloss
		pos_pred, neg_pred = prediction[:, :2].flatten(), prediction[:, 2:].flatten()

		target = torch.from_numpy(np.ones(self.batch_size * 2, dtype=np.float32)).cuda()
		loss = self.kg_loss(pos_pred, neg_pred, target)
		return loss

	def EXP_forward(self, batchId, flag="exp_infos_2"):
		# get batch data
		start_idx = batchId * self.batch_size
		end_idx = start_idx + self.batch_size

		if end_idx > self.kg_trainsize:
			end_idx = self.kg_trainsize
			start_idx = end_idx - self.batch_size

		if end_idx == start_idx:
			start_idx = 0
			end_idx = start_idx + self.batch_size

		cstrategy_embedddings = self.i_embeddings(torch.Tensor(self.exp_infos_2["cstrategy_ids"][start_idx:end_idx]).long().cuda())
		task_infos = torch.Tensor(self.exp_infos_2["task_infos"][start_idx:end_idx]).cuda()
		performance_infos = torch.Tensor(self.exp_infos_2["performance_infos"][start_idx:end_idx]).cuda()

		# get prediction
		cstrategy_embedddings = F.dropout(cstrategy_embedddings, p=self.dropout, training=self.training)
		model_inputs = torch.cat((cstrategy_embedddings,task_infos),1) 
		prediction = self.lin(model_inputs)

		# get exp sloss
		if flag=="exp_infos_1":
			loss = self.exp_loss(prediction[:,1].reshape(-1,1), performance_infos[:,1].reshape(-1,1))
		else:
			loss = self.exp_loss(prediction, performance_infos)
		return loss

	def main(self):
		train_batch_size = self.batch_size
		kg_trainsize = len(self.head_ids)
		kg_trainbatchnum = int(kg_trainsize // train_batch_size) + 1
		self.kg_trainsize = kg_trainsize
		self.kg_loss = nn.MarginRankingLoss(margin=1)

		exp1_trainsize = len(self.exp_infos_1["cstrategy_ids"])
		exp1_trainbatchnum = int(exp1_trainsize // train_batch_size) + 1
		self.exp1_trainsize = exp1_trainsize
		exp2_trainsize = len(self.exp_infos_2["cstrategy_ids"])
		exp2_trainbatchnum = int(exp2_trainsize // train_batch_size) + 1
		self.exp2_trainsize = exp2_trainsize
		self.exp_loss = nn.MSELoss()
		
		optimizer = opt.Adam(self.get_parameters(), lr=self.learning_rate, betas=(0.9, 0.999))

		self.logger.info("\n\n* Doing item embedding training.........")
		for epoch in range(self.epochs):
			kg_totaloss, exp1_totaloss, exp2_totaloss = 0, 0, 0
			kg_loss_lst, exp1_loss_lst, exp2_loss_lst = [], [], []

			# train kg
			if self.KG:
				begin = time.time()
				for batch_id in range(kg_trainbatchnum):
					optimizer.zero_grad()
					loss = self.KG_forward(batch_id)
					kg_loss_lst.append(loss.detach().cpu().data.numpy())
					kg_totaloss += loss
					loss.backward()
					optimizer.step()
				avg_kg_loss = np.mean(kg_loss_lst).item()
				end = time.time()
				self.logger.info("* epoch: %d [kg], time: %lf, kg_totaloss: %lf, avg_kg_loss: %lf", epoch, (end - begin), kg_totaloss, avg_kg_loss)

			# train exp1
			if self.EXP:
				begin = time.time()
				for batch_id in range(exp1_trainbatchnum):
					optimizer.zero_grad()
					loss = self.EXP_forward(batch_id, flag="exp_infos_1")
					exp1_loss_lst.append(loss.detach().cpu().data.numpy())
					exp1_totaloss += loss
					loss.backward()
					optimizer.step()
				avg_exp1_loss = np.mean(exp1_loss_lst).item()
				end = time.time()
				self.logger.info("* epoch: %d [exp1], time: %lf, exp1_totaloss: %lf, avg_exp1_loss: %lf", epoch, (end - begin), exp1_totaloss, avg_exp1_loss)

				# train exp2
				begin = time.time()
				for batch_id in range(exp2_trainbatchnum):
					optimizer.zero_grad()
					loss = self.EXP_forward(batch_id, flag="exp_infos_2")
					exp2_loss_lst.append(loss.detach().cpu().data.numpy())
					exp2_totaloss += loss
					loss.backward()
					optimizer.step()
				avg_exp2_loss = np.mean(exp2_loss_lst).item()
				end = time.time()
				self.logger.info("* epoch: %d [exp2], time: %lf, exp2_totaloss: %lf, avg_exp2_loss: %lf", epoch, (end - begin), exp2_totaloss, avg_exp2_loss)
		self.logger.info("* Item embedding training finished\n\n")

		cstrategy_embeddings = self.i_embeddings.weight.data[:self.cstrategies_num+1, :].detach().cpu()
		return cstrategy_embeddings
