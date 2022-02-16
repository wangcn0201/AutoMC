import copy
from ptflops import get_model_complexity_info


def get_model_num_param(model):
    model = copy.deepcopy(model)
    model = model.cuda()

    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    return '{:.3f}M'.format(params / 1e6)

def get_model_flops(model):
    model = copy.deepcopy(model)
    model = model.cuda()

    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    return '{:.5f}G'.format(macs * 2 / 1e9)

def get_model_num_param_flops(model):
    model = copy.deepcopy(model)
    model = model.cuda()

    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    return '{:.3f}M'.format(params / 1e6), '{:.5f}G'.format(macs * 2 / 1e9)

def get_compression_rate_for_result(model, small_model):
    model = copy.deepcopy(model)
    small_model = copy.deepcopy(small_model)
    model = model.cuda()
    small_model = small_model.cuda()

    flops_std_small, num_param_small = get_model_complexity_info(small_model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    flops_std_small *= 2
    flops_std, num_param = get_model_complexity_info(model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    flops_std *= 2

    rate_decreased_param = 1 - num_param_small / num_param
    rate_decreased_flops = 1 - flops_std_small / flops_std
    
    return ('{:.3f}M'.format(num_param_small / 1e6), rate_decreased_param), ('{:.5f}G'.format(flops_std_small / 1e9), rate_decreased_flops)

def get_compression_rate(model, small_model):
    model = copy.deepcopy(model)
    small_model = copy.deepcopy(small_model)
    model = model.cuda()
    small_model = small_model.cuda()

    flops_std_small, num_param_small = get_model_complexity_info(small_model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    flops_std, num_param = get_model_complexity_info(model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)

    compression_rate = num_param_small / num_param
    
    return compression_rate