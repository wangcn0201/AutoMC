'''
Our implement of LMA
'''
import torch
import math

from utils import *
import models
from train import *


class knowledge_distillation:
    def __init__(self, data, save_dir, arch, epochs=300, rate_based_on_teacher=None, rate_based_on_original=0.7, studentmodel_act_func='lma', lma_numBins=8, fixed_seed=False, kd_params=(0.7, 2), use_logger=True):
        self.cuda = torch.cuda.is_available()
        self.save_dir = save_dir
        self.data_dir = data['dir']
        self.data_name = data['name']
        self.arch = arch['dir']
        self.arch_name = arch['name']
        self.use_logger = use_logger
        if self.use_logger == True:
            self.logger = set_logger('{}_{}_C1'.format(self.data_name, self.arch_name), save_dir)
        elif self.use_logger == False:
            self.logger = None
        else:
            self.logger = use_logger
        if rate_based_on_teacher != None and rate_based_on_original != None:
            if self.logger:
                self.logger.error('There can not be two rates at the same time.')
            raise ValueError('There can not be two rates at the same time.')
        if rate_based_on_teacher == None and rate_based_on_original == None:
            if self.logger:
                self.logger.error('There must be one rate.')
            raise ValueError('There must be one rate.')
        if rate_based_on_teacher == None:
            self.rate = rate_based_on_original
            self.cfg = None
        else:
            self.rate = rate_based_on_teacher
            self.cfg = torch.load(self.arch).cfg
        self.rate = math.sqrt(self.rate)
        self.epochs = epochs
        self.lr = 1e-3
        self.lr_sche = 'StepLR'
        self.kd_params = kd_params # alpha, temperature
        self.studentmodel_act_func = studentmodel_act_func
        self.lma_numBins = lma_numBins
        if fixed_seed:
            seed_torch()

    def main(self):
        if self.logger:
            self.logger.info(">>>>>> Starting C1(knowledge distillation)")

        # Create data loader
        train_loader, val_loader = models.load_data(self.data_name, self.data_dir, arch_name=self.arch_name)
        if self.logger:
            self.logger.info("Loaded dateset '{}' from '{}'".format(self.data_name, self.data_dir))

        # Create student model
        model = models.__dict__[self.arch_name](num_classes=models.get_num_classes(self.data_name), 
            rate=self.rate, activation=self.studentmodel_act_func, numBins=self.lma_numBins, cfg=self.cfg)
        if self.logger:
            self.logger.info("Created student model '{}'".format(self.arch_name))
            self.logger.info("The parameters used when creating the model are rate={}, activation={}, numBins={}, cfg={}".format(self.rate, self.studentmodel_act_func, self.lma_numBins, self.cfg))
            self.logger.info("The student model's cfg={}".format(model.cfg))

        # Load teacher model
        teacher_model = torch.load(self.arch)
        if self.logger:
            self.logger.info("Loaded teacher model '{}' from {}".format(self.arch_name, self.arch))
            self.logger.info("The teacher model's cfg={}".format(teacher_model.cfg))

        # Test before training
        metrics_original = test_at_beginning_original(teacher_model, self.data_name, self.data_dir, self.logger, self.arch_name)

        model_dir = os.path.join(self.save_dir, '{}_small_unfinetuned_{}.pth.tar'.format(self.arch_name, time_file_str()))
        torch.save(model, model_dir)

        # Knowledge distillation
        if self.logger:
            self.logger.info("Entering knowledge distillation...")
        ft = fine_tune(self.save_dir, model, train_loader, val_loader, original_model=teacher_model, epochs=self.epochs, lr=self.lr, lr_sche=self.lr_sche, logger=self.logger, kd_params=self.kd_params, return_file=True, use_logger=self.use_logger)
        bestname, val_metrics = ft.main()

        model = torch.load(bestname)
        model_dir = os.path.join(self.save_dir, '{}_small_{}.pth.tar'.format(self.arch_name, time_file_str()))
        torch.save(model, model_dir)

        # Calculate metrics
        result = calc_result(teacher_model, metrics_original, model, val_metrics, model_dir, self.logger)
        save_result_to_json(self.save_dir, result)

        if self.use_logger == True:
            close_logger()
        return result

'''
if __name__ == '__main__':
    arch_name = 'vgg13'
    data = {'dir': './data', 'name': 'mini_cifar10'}
    save_dir = './snapshots/{}/C1/'.format(arch_name)
    arch = {'dir': './trained_models/{}/{}.pth.tar'.format(data['name'], arch_name), 'name': arch_name}
    kd = knowledge_distillation(data, save_dir, arch, studentmodel_act_func='lma', epochs=1,
                                fixed_seed=True, rate_based_on_teacher=0.7, rate_based_on_original=None)
    print(kd.main())
'''