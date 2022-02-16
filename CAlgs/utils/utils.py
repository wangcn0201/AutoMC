import os, time, shutil
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import matplotlib.pyplot as plt
import json
import logging
import models
from .compute_model_complexity import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs

def time_file_str():
    return time.strftime("%Y%m%d-%H%M%S")

def layer_parameter_num(module):
    return isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Conv2d)

def is_layer_with_parameters(module):
    return isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.MaxPool2d)
    # return isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.ReLU) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.MaxPool2d)

def save_checkpoint(model, is_best, filename, bestname):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    dirname = os.path.dirname(bestname)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model, filename)
    if is_best:
        shutil.copyfile(filename, bestname)

def test_at_beginning(model, val_loader, logger, message='accu before is'):
    if torch.cuda.is_available():
        model = model.cuda()
    metrics = validate(model, val_loader, logger)
    val_acc_top1_1, val_acc_top5_1 = metrics['acc_top1'], metrics['acc_top5']
    if logger:
        logger.info("{}: {} {}".format(message, val_acc_top1_1, val_acc_top5_1))
    return metrics

def test_at_beginning_original(model, data_name, data_dir, logger, arch_name, message='accu before is'):
    train_loader, val_loader = models.load_data(data_name, data_dir, arch_name=arch_name)
    if logger:
        logger.info("Loaded dateset '{}' from '{}'".format(data_name, data_dir))
    return test_at_beginning(model, val_loader, logger, message=message)

def calc_result(model_original, metrics_original, model, metrics, model_dir, logger):
    '''
    1. valmetric【字典】：acc_top1, acc_top1提升率；acc_top5, acc_top5提升率（相对输入原模型的）
    2. 参数量，实际参数量下降率（相对输入原模型的）
    3. flops值，flops下降率（相对输入原模型的）
    4. compressed_model
    '''
    result_metrics = {}
    keys = ['acc_top1', 'acc_top5']
    for key in keys:
        new_key = key + '_increased'
        result_metrics[key] = metrics[key]
        result_metrics[new_key] = (metrics[key] - metrics_original[key]) / metrics_original[key]

    result_param, result_flops = get_compression_rate_for_result(model_original, model)
    
    if logger:
        logger.info('$ acc_top1 is {:.5f}, increased {:.5f}%'.format(result_metrics['acc_top1'], result_metrics['acc_top1_increased'] * 100))
        logger.info('$ acc_top5 is {:.5f}, increased {:.5f}%'.format(result_metrics['acc_top5'], result_metrics['acc_top5_increased'] * 100))
        logger.info('$ Compression rate(remaining rate) is {:.5f}%'.format(100 - result_param[1] * 100))
        logger.info('$ Num of parameters is {}, decreased {:.5f}%'.format(result_param[0], result_param[1] * 100))
        logger.info('$ FLOPs is {}, decreased {:.5f}%'.format(result_flops[0], result_flops[1] * 100))
        logger.info('$ Model dir is {}'.format(model_dir))
        logger.info('{}, {}, {}, {}'.format(result_metrics, result_param, result_flops, model_dir))

    return result_metrics, result_param, result_flops, model_dir

def save_result_to_json(save_dir, result):
    json_path = os.path.join(save_dir, "metrics_val_best_weights.json")
    add_to_json('\n', json_path)
    add_to_json(result, json_path)



def validate(model, val_loader, logger=None, criterion=torch.nn.CrossEntropyLoss()):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        val_loader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    cuda = torch.cuda.is_available()

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    if cuda:
        criterion = criterion.cuda()

    # compute metrics over the dataset
    with torch.no_grad():
        for input, label in val_loader:

            # move to GPU if available
            if cuda:
                input, label = input.cuda(), label.cuda()
            # fetch the next evaluation batch
            input, label = Variable(input), Variable(label)
            
            # compute model output
            output = model(input)

            prec1, prec5 = accuracy(output, label, topk=(1, 5))

            loss = criterion(output, label)

            # compute all metrics on this batch
            summary_batch = {}
            summary_batch['acc_top1'] = prec1[0].cpu()
            summary_batch['acc_top5'] = prec5[0].cpu()
            summary_batch['loss'] = loss.data.cpu()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    if logger == None:
        print("- Eval metrics : " + metrics_string)
    else:
        if logger:
            logger.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
   
def add_to_json(content, json_path):
    with open(json_path, 'a') as f:
        f.write(str(content))

def set_logger(logger_name, log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    log_path = os.path.join(log_path, '{}.log'.format(logger_name))

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    return logger

def close_logger():
    logging.shutdown()

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def get_lr_scheduler(lr_sche, optimizer, epochs):
    if lr_sche == 'StepLR':
        lr_scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    elif lr_sche == 'MultiStepLR':
        lr_scheduler = MultiStepLR(optimizer, [epochs * 0.25, epochs * 0.5, epochs * 0.75], gamma=0.1)
    elif lr_sche == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=100)
    return lr_scheduler

class train_timer:
    def __init__(self):
        self.epoch_time = AverageMeter()
        self.start_time = time.time()
    
    def get_need_time(self, epochs, epoch):
        need_hour, need_mins, need_secs = convert_secs2time(self.epoch_time.val * (epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        return need_time
    
    def update(self):
        self.epoch_time.update(time.time() - self.start_time)
        self.start_time = time.time()

def get_filename_training(save_dir, working_type):
    filename = os.path.join(save_dir, 'checkpoint.{:}.{:}.pth.tar'.format(working_type, time_file_str()))
    bestname = os.path.join(save_dir, 'best.{:}.{:}.pth.tar'.format(working_type, time_file_str()))
    return filename, bestname

def seed_torch(seed=19260817):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def matrics_to_string(summ):
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    return metrics_string

def rv_duplicate_ele(l, sort=False):
    if sort: l.sort()
    tmp = [l[0]]
    for i in range(1, len(l)):
        if abs(l[i] - l[i - 1]) > 1e-6:
            tmp.append(l[i])
    return tmp