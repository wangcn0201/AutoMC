import torch
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable 
import torch.nn.functional as F
import numpy as np
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ParetoModel(torch.nn.Module):
	def __init__(self, embeddings_tensor, task_array, target_compression_rate, controller_dim=32, output_dim=2, hidden_dim=32, num_layers=2, dropout=0.5):
		super(ParetoModel, self).__init__()

		self.target_compression_rate = target_compression_rate

		embeddings_tensor = torch.FloatTensor(embeddings_tensor)
		self.cstartegy_embeddings = Variable(embeddings_tensor.cuda(), requires_grad=False)
		embedding_dim = embeddings_tensor.shape[1]

		task_tensor = torch.FloatTensor(task_array).reshape(1, -1)
		self.task_embeddings = Variable(task_tensor.cuda(), requires_grad=False)
		task_dim = task_tensor.shape[1]

		# the core of controller
		self.lstm = torch.nn.LSTM(embedding_dim, controller_dim, num_layers, batch_first=True)

		self.lin = nn.Sequential(
					Linear(controller_dim*num_layers+embedding_dim+task_dim, hidden_dim),
					nn.ReLU(),
					Linear(hidden_dim, output_dim),
					) # flops_decreased_ratio, acc_increased_ratio
		self.dropout = dropout
		return
		
	def ProcessPreSequences(self, pre_sequences):
		#print(pre_sequences)
		max_length = 0
		seq_lengths = []
		for i in range(len(pre_sequences)):
			if len(pre_sequences[i]) > max_length:
				max_length = len(pre_sequences[i])
			seq_lengths.append([i, len(pre_sequences[i])])

		seq_lengths = sorted(seq_lengths, key=(lambda x: x[1]), reverse=True)

		normed_pre_sequences = np.zeros((len(pre_sequences), max_length)) 
		for i in range(len(seq_lengths)):
			index, length = seq_lengths[i]
			normed_pre_sequences[i][:length] = pre_sequences[index]

		#with open("tett.txt", "a+") as f:
		#	f.write(str(normed_pre_sequences.shape)+"\n")
		#	f.write(str(seq_lengths)+"\n")
		#	f.write(str(pre_sequences)+"\n")
		#	f.write(str(normed_pre_sequences)+"\n")
		#	f.write(str(torch.tensor(normed_pre_sequences).long())+"\n")
		normed_pre_sequences_embedding = self.cstartegy_embeddings[torch.tensor(normed_pre_sequences).long()] # batch, seq, feature
		normed_seq_lengths = [seq_lengths[i][1] for i in range(len(seq_lengths))]

		pre_sequences_embedding_packed = pack_padded_sequence(normed_pre_sequences_embedding, normed_seq_lengths, batch_first=True) 
		#print(pre_sequences_embedding_packed.shape)

		output, (pre_sequence_features, c_last) = self.lstm(pre_sequences_embedding_packed) # batch, feature'
		pre_sequence_features = pre_sequence_features.transpose(0,1)
		pre_sequence_features = pre_sequence_features.reshape(pre_sequence_features.shape[0],-1)
		return pre_sequence_features, seq_lengths

	def GetOptimalCandidate(self, predicted_scheme_scores, history_scheme_scores, optimal_num, seq_lengths):
		batch_size = predicted_scheme_scores.shape[0] # batch, 2 (parameter_decreased_ratio, acc_increased_ratio)

		# get normed_predicted_real_scheme_scores
		normed_history_scheme_scores = np.zeros((batch_size, 2)) 
		normed_history_compression_rate = np.zeros((batch_size, 1)) 
		history_scheme_scores = np.array(history_scheme_scores) # parameter, acc, compression_rate
		for i in range(batch_size):
			normed_history_scheme_scores[i] = history_scheme_scores[seq_lengths[i][0]][:2]
			normed_history_compression_rate[i] = history_scheme_scores[seq_lengths[i][0]][2:]
		normed_history_compression_rate = np.array(normed_history_compression_rate).tolist()

		normed_history_scheme_scores = torch.FloatTensor(normed_history_scheme_scores).cuda()
		flag = torch.FloatTensor(np.array([[-1,1]])).cuda()

		predicted_real_scheme_scores = normed_history_scheme_scores + normed_history_scheme_scores*predicted_scheme_scores*flag
		normed_predicted_real_scheme_scores = predicted_real_scheme_scores*flag

		# get pareto_optimal_results
		pareto_optimal_results = []
		selected = []
		
		if random.random() > 0.5:
			index = random.randint(0,batch_size-1)
			score = normed_predicted_real_scheme_scores[index]
			left_compression_rate = max(self.target_compression_rate-normed_history_compression_rate[index][0], 0)+1.0
			score = score/left_compression_rate
			pareto_optimal_results.append([index, score])
			selected.append(index)

		for i in range(batch_size):
			score1 = normed_predicted_real_scheme_scores[i]
			left_compression_rate1 = max(self.target_compression_rate-normed_history_compression_rate[i][0], 0)+1.0
			score1 = score1/left_compression_rate1
			pareto_optimal = True
			for j in range(batch_size):
				score2 = normed_predicted_real_scheme_scores[j]
				left_compression_rate2 = max(self.target_compression_rate-normed_history_compression_rate[j][0], 0)+1.0
				score2 = score2/left_compression_rate2
				results = torch.gt(score2, score1, out=None)
				if results[0] == True and results[1] == True:
					# score2 > score1 for all elements (-flops, acc)
					pareto_optimal = False
					break
			if pareto_optimal:
				if i not in selected:
					pareto_optimal_results.append([i, score1])
					selected.append(i)

		#print(pareto_optimal_results)
		# get optimal_num pareto_optimal_results
		if len(pareto_optimal_results) <= optimal_num:
			output_pareto_optimal_results = list(pareto_optimal_results)
			while len(output_pareto_optimal_results) < optimal_num:
				index = random.randint(0,batch_size-1)
				score1 = normed_predicted_real_scheme_scores[index]
				if index not in selected:
					output_pareto_optimal_results.append([index, score1])
					selected.append(index)
		else:
			pareto_optimal_results = sorted(pareto_optimal_results, key=lambda info: info[1][1], reverse=True)
			output_pareto_optimal_results = pareto_optimal_results[:optimal_num]

		# get real index
		optimal_scheme_indexs, predicted_optimal_step_scores = [], []
		for i in range(len(output_pareto_optimal_results)):
			index, score = output_pareto_optimal_results[i]
			ori_index = seq_lengths[index][0]
			optimal_scheme_indexs.append(ori_index)
			predicted_optimal_step_scores.append(predicted_scheme_scores[index].unsqueeze(0))
		predicted_optimal_step_scores = torch.cat(predicted_optimal_step_scores, dim=0)

		return optimal_scheme_indexs, predicted_optimal_step_scores

	def main(self, pre_sequences, next_cstrategies, history_scheme_scores, optimal_num):
		pre_sequence_features, seq_lengths = self.ProcessPreSequences(pre_sequences)
		next_cstrategy_embeddings = self.cstartegy_embeddings[torch.tensor(np.array(next_cstrategies).reshape(-1,1)).long()].squeeze(1)
		batch_size = pre_sequence_features.shape[0]

		model_input = torch.cat([next_cstrategy_embeddings, self.task_embeddings.expand(batch_size, -1), pre_sequence_features], 1)
		predicted_scheme_scores = self.lin(model_input)

		if optimal_num == -1:
			return predicted_scheme_scores
		optimal_scheme_indexs, predicted_optimal_step_scores = self.GetOptimalCandidate(predicted_scheme_scores, history_scheme_scores, optimal_num, seq_lengths)
		return optimal_scheme_indexs, predicted_optimal_step_scores
