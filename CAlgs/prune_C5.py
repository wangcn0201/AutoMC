import numpy as np
import math
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import models
import torch.nn.functional as F
from utils import *
from train import *
import copy


class HOS:
    def __init__(self, data, save_dir, arch, epochs_step3=250, epochs_step4=50, rate=0.7, metric='k34', strategy='P2', mse_factor=1, kd_params=(None, None), fixed_seed=False, use_logger=True):
        self.cuda = torch.cuda.is_available()
        self.data_dir = data['dir']
        self.data_name = data['name']
        self.save_dir = save_dir
        self.arch = arch['dir']
        self.arch_name = arch['name']
        self.rate = rate
        self.rate1 = math.sqrt(math.sqrt(rate))
        self.epochs_step3 = epochs_step3
        self.epochs_step4 = epochs_step4
        self.lr_step3 = 0.001
        self.lr_step4 = 0.0001
        self.lr_sche = 'MultiStepLR'
        self.metric = metric
        self.strategy = strategy
        self.mse_factor = mse_factor
        self.kd_params = kd_params
        if fixed_seed:
            seed_torch()
        self.use_logger = use_logger
        if self.use_logger == True:
            self.logger = set_logger('{}_{}_C5'.format(self.data_name, self.arch_name), self.save_dir)
        elif self.use_logger == False:
            self.logger = None
        else:
            self.logger = use_logger

    def step1(self, model):
        if self.logger:
            self.logger.info(">>> Step1")
        if self.cuda:
            model = model.cuda()

        cfg, cfg_mask = [], []
        modules = get_modules(model, self.arch_name, 'conv')
        T = get_T(modules, self.strategy)
        coef, rate = get_coef(T, modules, self.rate1, self.strategy, self.logger)

        if self.logger:
            self.logger.info("The calculated coef is {}".format(coef))
            self.logger.info("The actual pruning rate is {}".format(rate))

        # Decide how many and which filters to prune at each layer
        compressed_channels = 0
        total_channels = 0
        for index in range(len(modules)):
            if 'criterion' in modules[index][1]:
                m = modules[index][0]
                out_channels = m.weight.data.shape[0]
                a = m.weight.data.clone().cpu().numpy().flatten()

                prune_prob_stage = get_prune_prob_stage(T, a, self.strategy, coef)
                arg_max = get_arg_max(m, self.metric)

                num_keep = max(int(out_channels * (1 - prune_prob_stage)), 2)
                arg_max_rev = arg_max[::-1][:num_keep]
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                for i in range(len(mask)):
                    m.weight.data[i].mul_(mask[i])
                cfg_mask.append(mask.clone())
                cfg.append(num_keep)
                # self.logger.info('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(index, mask.shape[0], num_keep))
                compressed_channels += num_keep
                total_channels += mask.shape[0]
        if self.logger:
            self.logger.info('compressed channel num is {}'.format(1.0 * compressed_channels / total_channels))
        small_model = get_small_model(model, self.arch_name, 'conv')
        model_dir = os.path.join(self.save_dir, self.arch_name + '_step1_{}.pth.tar'.format(time_file_str()))
        torch.save(small_model, model_dir)
        return small_model
    
    def get_rate(self, rate1):
        self.rate2 = self.rate / rate1
        if self.logger:
            self.logger.info('the decomposition ratio is {}'.format(self.rate2))
        self.X_FACTOR = 1.0 / self.rate2

    def step2(self, model):
        if self.logger:
            self.logger.info(">>> Step2")
        if self.cuda:
            model = model.cuda()

        layers = get_modules(model, self.arch_name, 'conv')

        for layer in layers:
            if 'prune' in layer[1]:
                if layer[0].weight.shape[0] == 1 or layer[0].weight.shape[1] == 1: continue
                try:
                    layer_now = layer[2]
                    if self.logger:
                        self.logger.info('Decomposing layer: ' + str(layer_now))
                    subm_names = layer_now.strip().split('.')

                    layer_now = model.__getattr__(subm_names[0])

                    for s in subm_names[1:]:
                        layer_now = layer_now.__getattr__(s)

                    decomposed_layer = Tucker2DecomposedLayer(layer_now, subm_names[-1], self.X_FACTOR)

                    if len(subm_names) > 1:
                        m = model.__getattr__(subm_names[0])
                        for s in subm_names[1:-1]:
                            m = m.__getattr__(s)
                        m.__setattr__(subm_names[-1], decomposed_layer.new_layers)
                    else:
                        model.__setattr__(
                            subm_names[-1], decomposed_layer.new_layers)
                except:
                    if self.logger:
                        self.logger.info("Error while decomposing layer, shape is %s".format(str(layer[0].weight.shape)))
                    continue

        model_dir = os.path.join(
            self.save_dir, self.arch_name + '_step2_{}.pth.tar'.format(time_file_str()))
        torch.save(model, model_dir)
        return model

    def step3(self, model, model_init, train_loader, val_loader):
        if self.logger:
            self.logger.info(">>> Step3")
        if self.cuda:
            model = model.cuda()
            model_init = model_init.cuda()

        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=self.lr_step3,  weight_decay=1e-4)
        lr_scheduler = get_lr_scheduler(self.lr_sche, optimizer, self.epochs_step3)

        # Extract the softmax layer values from the baseline network
        output = test_orig_for_step3(model_init, train_loader, self.data_name, self.logger)

        # Test before training
        if self.logger:
            self.logger.info("Model:")
        test_at_beginning(model, val_loader, self.logger)
        if self.logger:
            self.logger.info("Model_init:")
        test_at_beginning(model_init, val_loader, self.logger)

        best_acc_top1 = 0.0
        timer = train_timer()

        # Get filename
        filename, bestname = get_filename_training(self.save_dir, 'step3')

        for epoch in range(self.epochs_step3):
            # Run one epoch
            need_time = timer.get_need_time(self.epochs_step3, epoch)
            if self.logger:
                self.logger.info("Epoch {}/{}  {:s}  lr={}".format(epoch + 1,
                                 self.epochs_step3, need_time, optimizer.param_groups[0]['lr']))

            # Train for one epoch
            train_for_step3(model, optimizer, train_loader, output, self.mse_factor, self.logger)

            lr_scheduler.step()

            # Evaluate on validation set
            val_metrics = validate(model, val_loader, self.logger)
            val_acc_top1 = val_metrics['acc_top1']
            is_best = val_acc_top1 >= best_acc_top1

            # Save weights
            save_checkpoint(model, is_best, filename, bestname)
            if is_best:
                if self.logger:
                    self.logger.info("- Found new best accuracy on validation set")
                best_acc_top1 = val_acc_top1
                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(self.save_dir, "metrics_val_best_weights.json")
                save_dict_to_json(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(self.save_dir, "metrics_val_last_weights.json")
            save_dict_to_json(val_metrics, last_json_path)

            timer.update()
        return torch.load(bestname)

    def main(self):
        if self.logger:
            self.logger.info(">>>>>> Starting C5")
            self.logger.info("Target rate is: {}".format(self.rate))

        # Load original model
        model = torch.load(self.arch)
        if self.logger:
            self.logger.info("loaded model '{}' from {}".format(self.arch_name, self.arch))
            self.logger.info("The original model's cfg={}".format(model.cfg))

        if self.cuda:
            model = model.cuda()
        
        # Load data
        train_loader, val_loader = models.load_data(self.data_name, self.data_dir, arch_name=self.arch_name)
        if self.logger:
            self.logger.info("Loaded dateset '{}' from '{}'".format(self.data_name, self.data_dir))

        # Test before compressing
        metrics_original = test_at_beginning(model, val_loader, self.logger)

        # Step1
        model_step1 = self.step1(copy.deepcopy(model))
        compression_rate = get_compression_rate(model, model_step1)
        if self.logger:
            self.logger.info("After step1, the model's cfg={}".format(model_step1.cfg))
            self.logger.info('After step1, The compression rate is {}'.format(compression_rate))
        test_at_beginning(model_step1, val_loader, self.logger, message='After step1, accu is')

        # Step2
        self.get_rate(compression_rate)
        model_step2 = self.step2(copy.deepcopy(model_step1))
        compression_rate = get_compression_rate(model, model_step2)
        if self.logger:
            self.logger.info("After step2, the model's arch is {}".format(model_step2))
            self.logger.info('After step2, The compression rate is {}'.format(compression_rate))

        # Step3
        model_step3 = self.step3(copy.deepcopy(model_step2), model, train_loader, val_loader)

        # Step4
        if self.logger:
            self.logger.info(">>> Step4")
        model_step4, val_metrics = fine_tune(self.save_dir, copy.deepcopy(model_step3), train_loader, val_loader,
                                             epochs=self.epochs_step4, lr=self.lr_step4, lr_sche=self.lr_sche, logger=self.logger, kd_params=self.kd_params, return_file=True, use_logger=self.use_logger).main()

        # Calculate metrics
        result = calc_result(model, metrics_original, torch.load(model_step4), val_metrics, model_step4, self.logger)
        save_result_to_json(self.save_dir, result)

        if self.use_logger == True:
            close_logger()
        return result

'''
if __name__ == '__main__':
    arch_name = 'vgg13'
    data = {'dir': './data', 'name': 'mini_cifar10'}
    save_dir = './snapshots/{}/C5/'.format(arch_name)
    arch = {'dir': './trained_models/{}/{}.pth.tar'.format(data['name'], arch_name), 'name': arch_name}
    h = HOS(data, save_dir, arch, metric='k34', strategy='P2', fixed_seed=True, epochs_step3=1, epochs_step4=1)
    print(h.main())
'''