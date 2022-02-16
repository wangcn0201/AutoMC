import os
import torch
import torch.nn as nn
from utils import *
from train import *


class NetworkSlimming:
    def __init__(self, data, save_dir, arch, fine_tune_epochs=300, rate=0.5, max_prune_per_layer=0.9, kd_params=(None, None), fixed_seed=False, use_logger=True):
        self.cuda = torch.cuda.is_available()
        self.data_dir = data['dir']
        self.data_name = data['name']
        self.save_dir = save_dir
        self.arch = arch['dir']
        self.arch_name = arch['name']
        self.rate = rate
        self.kd_params = kd_params
        self.lr = 1e-3 # default learning rate is 0.1 but decay
        self.lr_sche = 'MultiStepLR'
        self.max_prune_per_layer = max_prune_per_layer
        self.fine_tune_epochs = fine_tune_epochs
        if fixed_seed:
            seed_torch()
        self.use_logger = use_logger
        if self.use_logger == True:
            self.logger = set_logger('{}_{}_C3'.format(self.data_name, self.arch_name), self.save_dir)
        elif self.use_logger == False:
            self.logger = None
        else:
            self.logger = use_logger

    def get_mask(self, weight_copy, thre):
        mask = weight_copy.gt(thre).float().cuda()
        current_rate = mask.sum(dim=0) / len(mask)
        if 1 - current_rate > self.max_prune_per_layer:
            _, sort_index = torch.sort(weight_copy)
            mask = torch.ones_like(mask).float().cuda()
            for i in range(min(int(torch.floor(torch.tensor(self.max_prune_per_layer * len(mask)))), len(mask) - 1)):
                mask[sort_index[i]] = 0
        return mask

    def get_thre(self, model):
        if self.logger:
            self.logger.info("Getting filter's original ranking...")
        filter_ranks = {}
        total = 0  # the total number of kernals in BN layer
        modules = get_modules(model, self.arch_name, 'bn')
        
        for index in range(len(modules)):
            layer = modules[index][0]
            if 'criterion' in modules[index][1]:
                total += layer.weight.data.shape[0]
                filter_ranks[index] = layer.weight.data.abs().clone()
        
        def get_model(thre):
            model_copy = copy.deepcopy(model)
            return self.get_compressed_model(thre, total, model_copy, small_model_with_param=False)[1]
        
        thre_candidates = []
        for k in filter_ranks:
            for i in filter_ranks[k]:
                thre_candidates.append(i.float())
        
        thre_candidates = rv_duplicate_ele(thre_candidates, sort=True)

        if self.logger:
            self.logger.info('length of thre candidates:' + str(len(thre_candidates)))
        l, r = 0, len(thre_candidates)
        while l < r - 1:
            mid = (l + r) // 2
            now_rate = get_model(thre_candidates[mid])
            if now_rate > self.rate: l = mid
            else: r = mid

        now_rate = get_model(thre_candidates[l])
        rate1 = abs(now_rate - self.rate)
        now_rate = get_model(thre_candidates[r])
        rate2 = abs(now_rate - self.rate)
        return (thre_candidates[l] if rate1 < rate2 else thre_candidates[r]), total

    def get_compressed_model(self, thre, total, model, small_model_with_param=True):
        modules = get_modules(model, self.arch_name, 'bn')
        pruned = 0
        for index in range(len(modules)):
            m = modules[index][0]
            if 'criterion' in modules[index][1]:
                weight_copy = m.weight.data.abs().clone()
                mask = self.get_mask(weight_copy, thre)
                pruned += mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                # self.logger.info('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(index, mask.shape[0], int(torch.sum(mask))))

        compressed_model = get_small_model(model, self.arch_name, 'bn', small_model_with_param=small_model_with_param)
        compression_rate = 1 - pruned / total
        if small_model_with_param:
            if self.logger:
                self.logger.info('bias of pruning rate(num of channel):' + str(abs(compression_rate - self.rate)))
        compression_rate = get_compression_rate(model, compressed_model)
        if small_model_with_param:
            if self.logger:
                self.logger.info('bias of pruning rate(num of parameters):' + str(abs(compression_rate - self.rate)))
        return compressed_model, compression_rate

    def main(self):
        if self.logger:
            self.logger.info(">>>>>> Starting C3")

        # Load original model
        model = torch.load(self.arch)
        if self.logger:
            self.logger.info("Loaded model '{}' from {}".format(self.arch_name, self.arch))
            self.logger.info("The original model's cfg={}".format(model.cfg))

        if self.cuda:
            model = model.cuda()

        # Test before training
        metrics_original = test_at_beginning_original(model, self.data_name, self.data_dir, self.logger, self.arch_name)

        # Calculate thre
        if self.logger:
            self.logger.info("Calculating thre...")
        thre, total = self.get_thre(model)

        # Get compressed model
        if self.logger:
            self.logger.info("Getting compressed model...")
        small_model, _ = self.get_compressed_model(thre, total, model)
        if self.logger:
            self.logger.info("Got new model '{}'".format(self.arch_name))
            self.logger.info("The new model's cfg={}".format(small_model.cfg))

        # Save new but untrained model
        model_dir = os.path.join(self.save_dir, '{}_small_unfinetuned_{}.pth.tar'.format(self.arch_name, time_file_str()))
        torch.save(small_model, model_dir)

        # Finetune
        if self.logger:
            self.logger.info("Entering finetune...")
        if self.kd_params == (None, None):
            original_model = None
        else:
            original_model = model
        t = Train({'dir': self.data_dir, 'name': self.data_name}, self.save_dir,
                  {'dir': model_dir, 'name': self.arch_name}, logger=self.logger,
                  epochs=self.fine_tune_epochs, lr=self.lr, lr_sche=self.lr_sche, return_file=False, original_model=original_model,
                  kd_params=self.kd_params, use_logger=self.use_logger)
        acc_dict, small_model = t.main()

        # Save final model
        model_dir = os.path.join(self.save_dir, '{}_small_{}.pth.tar'.format(self.arch_name, time_file_str()))
        torch.save(small_model, model_dir)

        # Calculate metrics
        result = calc_result(torch.load(self.arch), metrics_original, small_model, acc_dict, model_dir, self.logger)
        save_result_to_json(self.save_dir, result)

        if self.use_logger == True:
            close_logger()
        return result

'''
if __name__ == '__main__':
    arch_name = 'vgg13'
    data = {'dir': './data', 'name': 'mini_cifar10'}
    save_dir = './snapshots/{}/C3/'.format(arch_name)
    arch = {'dir': './trained_models/{}/{}.pth.tar'.format(data['name'], arch_name), 'name': arch_name}
    ns = NetworkSlimming(data, save_dir, arch, fine_tune_epochs=1, rate=0.7, fixed_seed=True)
    print(ns.main())
'''