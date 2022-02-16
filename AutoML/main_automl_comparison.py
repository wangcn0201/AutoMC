import argparse
import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))),"CAlgs")) 
from automl_random import AutoMLRandom
from automl_random2 import AutoMLRandom2
from automl_evolution import AutoMLEvolution
from automl_rl import AutoMLRl
from automl_our import AutoMLOur
from automl_our_ablation_exp import AutoMLOur_AblationEXP
from automl_our_ablation_kg import AutoMLOur_AblationKG
from automl_our_ablation_kmodel import AutoMLOur_AblationKModel
from automl_our_ablation_pmodel import AutoMLOur_AblationPModel
from automl_our_ablation_space import AutoMLOur_AblationSpace


# Parser arguments
parser = argparse.ArgumentParser(description='AutoML Method for Model Compression Scheme Design.')
parser.add_argument('--automl_method', '--am', type=str, default="random", help='the AutoML algorithm name')
parser.add_argument('--config_path', '--cp', type=str, default="config.json", help='the file path for the config information')
parser.add_argument('--task_name', '--tn', type=str, default="resnet56+mini_cifar10+0.3", help='the compression task name')
parser.add_argument('--task_info', '--ti', type=str, default="", help='the task information')
args = parser.parse_args()

if __name__ == '__main__':
	if args.automl_method == "random":
		AutoMLRandom(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "random2":
		AutoMLRandom2(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "evolution":
		AutoMLEvolution(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "rl":
		AutoMLRl(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "our":
		AutoMLOur(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "our_exp":
		AutoMLOur_AblationEXP(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "our_kg":
		AutoMLOur_AblationKG(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "our_kmodel":
		AutoMLOur_AblationKModel(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "our_pmodel":
		AutoMLOur_AblationPModel(args.config_path, args.task_name, args.task_info).main()
	elif args.automl_method == "our_space":
		AutoMLOur_AblationSpace(args.config_path, args.task_name, args.task_info).main()
