import argparse
import sys
import os
import torch, random
import numpy as np
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))),"CAlgs")) 
from utils import *

def seed_torch(seed=19260817):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()
# Parser arguments
parser = argparse.ArgumentParser(description='Test metrics.')
parser.add_argument('--model_path', '--mp', type=str, default=None, help='model path')
parser.add_argument('--data_name', '--dn', type=str, default=None, help='data name')
parser.add_argument('--arch_name', '--an', type=str, default=None, help='arch name')
args = parser.parse_args()

data_dir = '../CAlgs/data'
data_name = args.data_name
model = torch.load(args.model_path).cuda()
logger = None
arch_name = args.arch_name


test_at_beginning_original(model, data_name, data_dir, logger, arch_name)
print(get_model_num_param_flops(model))