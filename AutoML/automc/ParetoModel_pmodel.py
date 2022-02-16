import torch
import torch.nn.functional as F
import rl.utils as utils
from torch.autograd import Variable 


# not contains skip-connection
class ParetoModel_pmodel(torch.nn.Module):
	def __init__(self, embeddings_tensor, search_space, cstartegies, action_list, controller_hid=100, cuda=True, mode="train",
				 softmax_temperature=5.0, tanh_c=2.5):
		super(ParetoModel_pmodel, self).__init__()
		self.mode = mode
		# operator categories for each controller RNN output
		self.action_list = action_list
		self.controller_hid = controller_hid
		self.is_cuda = cuda

		# set hyperparameters
		self.softmax_temperature = softmax_temperature
		self.tanh_c = tanh_c

		# build encoder
		self.search_space = search_space
		embeddings_tensor = torch.FloatTensor(embeddings_tensor)
		self.encoder = Variable(embeddings_tensor.cuda(), requires_grad=False)
		self.embedding_dim = embeddings_tensor.shape[1]

		num_total_tokens = len(cstartegies) + 1
		self.num_total_tokens = num_total_tokens
		self.cstartegies = cstartegies
		self.end_index = len(self.cstartegies)

		# the core of controller
		self.lstm = torch.nn.LSTMCell(self.embedding_dim, controller_hid)

		# build decoder
		self.decoder = torch.nn.Linear(controller_hid, self.num_total_tokens)

		self.reset_parameters()

	def reset_parameters(self):
		init_range = 0.1
		for param in self.parameters():
			param.data.uniform_(-init_range, init_range)
		self.decoder.bias.data.fill_(0)

	def forward(self, inputs, hidden, action_name, is_embed):
		embed = inputs

		hx, cx = self.lstm(embed, hidden)
		logits = self.decoder(hx)

		logits /= self.softmax_temperature

		# exploration
		if self.mode == 'train':
			logits = (self.tanh_c * torch.tanh(logits))

		return logits, (hx, cx)

	def _construct_action(self, actions):
		scheme_code_list = []
		for single_action in actions:
			scheme_code = []
			for action, action_name in zip(single_action, self.action_list):
				if action != self.num_total_tokens:
					action_content = self.search_space[action_name][action]
					if action_content == self.end_index:
						break
					action_content = self.cstartegies[action_content]
					scheme_code.append(action_content)
				else:
					break
			scheme_code_list.append(scheme_code)
		return scheme_code_list

	def action_index(self, action_name):
		key_names = self.search_space.keys()
		for i, key in enumerate(key_names):
			if action_name == key:
				return i

	def sample(self, batch_size=1, with_details=False):

		if batch_size < 1:
			raise Exception(f'Wrong batch_size: {batch_size} < 1')

		inputs = torch.zeros([batch_size, self.embedding_dim])
		hidden = (torch.zeros([batch_size, self.controller_hid]), torch.zeros([batch_size, self.controller_hid]))
		if self.is_cuda:
			inputs = inputs.cuda()
			hidden = (hidden[0].cuda(), hidden[1].cuda())
		entropies = []
		log_probs = []
		actions = []
		for block_idx, action_name in enumerate(self.action_list):
			decoder_index = self.action_index(action_name)

			logits, hidden = self.forward(inputs, hidden, action_name, is_embed=(block_idx == 0))

			if block_idx == 0:
				logits = logits[:,:-1]
			probs = F.softmax(logits, dim=-1)
			log_prob = F.log_softmax(logits, dim=-1)

			entropy = -(log_prob * probs).sum(1, keepdim=False)
			action = probs.multinomial(num_samples=1).data
			selected_log_prob = log_prob.gather(
				1, utils.get_variable(action, requires_grad=False))

			entropies.append(entropy)
			log_probs.append(selected_log_prob[:, 0])

			inputs = utils.get_variable(
				action[:, 0],
				self.is_cuda,
				requires_grad=False)

			inputs = self.encoder[inputs]

			actions.append(action[:, 0])

		actions = torch.stack(actions).transpose(0, 1)
		scheme_code_list = self._construct_action(actions)

		if with_details:
			return scheme_code_list, torch.cat(log_probs), torch.cat(entropies)
		return scheme_code_list

	def init_hidden(self, batch_size):
		zeros = torch.zeros(batch_size, self.controller_hid)
		return (utils.get_variable(zeros, self.is_cuda, requires_grad=False),
				utils.get_variable(zeros.clone(), self.is_cuda, requires_grad=False))
