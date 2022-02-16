'''
prune_C1: LMA (knowledge_distillation), AAAI 2020
	TE1: knowledge distillation based on LMA function
	@@@@@
	HP1: fine_tune_epochs * {0.2, 0.4, 0.6, 0.8, 1.0, 1.5} (TE1)
	HP2: prune_ratio's ratio # {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} (TE1) --> prune_ratio
	HP3: LMA_segment_num {6, 8, 10} (TE1)
	HP4: distillation_temperature_factor {1, 3, 6, 10} (TE1)
	HP5: distillation_alpha_factor {0.05, 0.3, 0.5, 0.99} (TE1)
prune_C2: LeGR, CVPR 2020
	TE2: filter pruning based on evolutionary algorithm
	TE3: common fine tuning 
	@@@@@
	HP1: fine_tune_epochs * {0.2, 0.4, 0.6, 0.8, 1.0, 1.5} (TE3)
	HP2: prune_ratio's ratio # {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} (TE2) --> prune_ratio
	HP6: max_layer_prune_ratio {0.7, 0.9} (TE2)
	HP7: ea_epochs * {0.4, 0.5, 0.6, 0.7} (TE2)
	HP8: ea_fine_tune_epochs * {0.01} (TE2)
	HP9: filter_importance_metric {l1_weight, l2_weight, l2_bn, l2_bn_param} (TE2)
prune_C3: NetworkSlimming, ICCV 2017
	TE4: channel pruning based on Scaling Factors in BN Layers
	TE3: common fine tuning 
	@@@@@
	HP1: fine_tune_epochs * {0.2, 0.4, 0.6, 0.8, 1.0, 1.5} (TE3)
	HP2: prune_ratio's ratio # {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} (TE4) --> prune_ratio
	HP6: max_layer_prune_ratio {0.7, 0.9} (TE4)
prune_C4: SFP (SoftFilterPruning), IJCAI 2018
	TE5: filter pruning based on back-propagation
	@@@@@
	HP2: prune_ratio's ratio # {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} (TE5) --> prune_ratio
	HP10: back_propagation_epochs * {0.2, 0.4, 0.6, 0.8, 1.0, 1.5} (TE5)
	HP11: prune_update_frequency {1, 3, 5} (TE5)
prune_C5: HOS, CVPR 2020
	TE6: filter pruning based on HOS
	TE7: low-rank kernel approximation based on HOOI 
	TE3: common fine tuning 
	@@@@@
	HP1: fine_tune_epochs * {0.2, 0.4, 0.6, 0.8, 1.0, 1.5} (TE3)
	HP2: prune_ratio's ratio # {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} (TE6) --> prune_ratio
	HP12: global filter importance basis {P1, P2, P3} (TE6)
	HP13: local filter importance basis {l1norm, k34, skew_kur} (TE6)
	HP14: KD_fine_tune_epochs * {0.2, 0.4, 0.6, 0.8, 1.0, 1.5} (TE7)
	HP15: factor of auxiliary MSE losses {1, 3, 5} (TE7)
prune_C7: LFB, ICCV 2019
	TE8: low-rank filter approximation based on filter basis
	@@@@@
	HP1: fine_tune_epochs * {0.2, 0.4, 0.6, 0.8, 1.0, 1.5} (TE8)
	HP2: decomposition_ratio's ratio # {1.0} (TE8) --> decomposition ratio
	HP16: factor of auxiliary losses {0.5, 1, 1.5, 3, 5} (TE8)
	HP17: fine tune auxiliary loss {NLL, CE} (TE8)
'''

SearchSpace  = {
	"prune_C1": { 
		"HP1": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
		"HP2": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
		"HP3": [6, 8, 10],
		"HP4": [1, 3, 6, 10],
		"HP5": [0.05, 0.3, 0.5, 0.99]
	},
	"prune_C2": {
		"HP1": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
		"HP2": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
		"HP6": [0.7, 0.9],
		"HP7": [0.4, 0.5, 0.6, 0.7],
		"HP8": [0.01],
		"HP9": ["l1_weight", "l2_weight", "l2_bn", "l2_bn_param"],
	},
	"prune_C3": {
		"HP1": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
		"HP2": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
		"HP6": [0.7, 0.9]
	},
	"prune_C4": {
		"HP2": [0.1, 0.3, 0.5, 0.9, 1.0],
		"HP10": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
		"HP11": [1, 3, 5]
	},
	"prune_C5": {
		"HP1": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
		"HP2": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
		"HP12": ["P1", "P2", "P3"],
		"HP13": ["l1norm", "k34", "skew_kur"],
		"HP14": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
		"HP15": [1, 3, 5]
	},
	"prune_C7": {
		"HP1": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
		"HP2": [1.0],
		"HP16": [0.5, 1, 1.5, 3, 5],
		"HP17": ["NLL", "CE"]
	}
}