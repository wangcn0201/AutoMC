'''
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
'''

SubSearchSpace  = {
	"prune_C2": {
		"HP1": [0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
		"HP2": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
		"HP6": [0.7, 0.9],
		"HP7": [0.4, 0.5, 0.6, 0.7],
		"HP8": [0.01],
		"HP9": ["l1_weight", "l2_weight", "l2_bn", "l2_bn_param"],
	}
}