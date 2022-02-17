import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from utils.utils import *
import models


class fine_tune:
    def __init__(self, save_dir, compressed_model, train_loader, val_loader, original_model=None, epochs=200, lr=1e-3, lr_sche='MultiStepLR', logger=None, kd_params=(0.0, 1), return_file=False, use_logger=True):
        self.cuda = torch.cuda.is_available()
        self.save_dir = save_dir
        self.original_model = original_model
        self.compressed_model = compressed_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.lr = lr
        self.lr_sche = lr_sche
        self.kd_params = kd_params # alpha, temperature
        self.return_file = return_file
        self.kd = False if original_model == None else True
        self.use_logger = use_logger
        if self.use_logger == False:
            self.logger = None
        else:
            if logger == None:
                self.logger = set_logger('fine_tune', self.save_dir)
                self.mine_logger = True
            else:
                self.logger = logger
                self.mine_logger = False

    def main(self):
        # Print settings
        if self.logger:
            self.logger.info("Training Settings:")
            self.logger.info("save_dir : " + str(self.save_dir))
            self.logger.info("epochs : " + str(self.epochs))
            self.logger.info("lr : " + str(self.lr))
            self.logger.info("kd_params : " + str(self.kd_params))
            self.logger.info("return_file : " + str(self.return_file))
            self.logger.info("kd : " + str(self.kd))

        if self.cuda:
            self.compressed_model = self.compressed_model.cuda()
            if self.kd:
                self.original_model = self.original_model.cuda()

        # Define loss function (criterion), optimizer and lr scheduler
        optimizer = optim.SGD(self.compressed_model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        if not self.kd:
            criterion = nn.CrossEntropyLoss().cuda()
        lr_scheduler = get_lr_scheduler(self.lr_sche, optimizer, self.epochs)
        '''
        if self.kd:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)
        else:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        '''

        # Get filename
        filename, bestname = get_filename_training(self.save_dir, 'finetune')

        # Train model
        if self.logger:
            self.logger.info("Starting training for {} epoch(s)".format(self.epochs))
        best_acc_top1 = 0.0
        best_val_metrics = {}
        timer = train_timer()

        for epoch in range(self.epochs):
            # Run one epoch
            need_time = timer.get_need_time(self.epochs, epoch)
            if self.logger:
                self.logger.info("Epoch {}/{}  {:s}  lr={}".format(epoch + 1,
                                 self.epochs, need_time, optimizer.param_groups[0]['lr']))

            # Train for one epoch
            if self.kd:
                train(self.compressed_model, self.original_model, None, optimizer,
                      self.train_loader, self.logger, self.kd_params)
            else:
                train(self.compressed_model, None, criterion,
                      optimizer, self.train_loader, self.logger)

            lr_scheduler.step()

            # Evaluate on validation set
            val_metrics = validate(self.compressed_model,
                                   self.val_loader, self.logger)
            val_acc_top1 = val_metrics['acc_top1']
            is_best = val_acc_top1 >= best_acc_top1

            # Save weights
            save_checkpoint(self.compressed_model, is_best, filename, bestname)
            if is_best:
                if self.logger:
                    self.logger.info("- Found new best accuracy on validation set")
                best_acc_top1 = val_acc_top1
                best_val_metrics = val_metrics
                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(
                    self.save_dir, "metrics_val_best_weights.json")
                save_dict_to_json(val_metrics, best_json_path)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(
                self.save_dir, "metrics_val_last_weights.json")
            save_dict_to_json(val_metrics, last_json_path)

            timer.update()
        if self.use_logger == True and self.mine_logger:
            close_logger()
        if self.return_file:
            return bestname, best_val_metrics
        else:
            return torch.load(bestname), best_val_metrics


def train(model, teacher_model, criterion, optimizer, train_loader, logger, kd_params=(None, None), cuda=True):
    # Set model to training mode
    model.train()
    if teacher_model != None:
        teacher_model.eval()

    # Summary for current training loop and a running average object for loss
    summ = []
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    alpha, temperature = kd_params

    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:
        for i, (input, target) in enumerate(train_loader):
            # Move to GPU if available
            if cuda:
                input, target = input.cuda(), target.cuda()
            # Monvert to torch Variables
            input, target = Variable(input), Variable(target)

            # Mompute model output, fetch teacher output, and compute KD loss
            output = model(input)

            # Get one batch output from teacher_outputs list
            if teacher_model != None:
                with torch.no_grad():
                    output_teacher_batch = teacher_model(input)
                if cuda:
                    output_teacher_batch = output_teacher_batch.cuda()

                loss = loss_fn_kd(output, target, output_teacher_batch, alpha, temperature)
            else:
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % 100 == 0:
                # Compute all metrics on this batch
                summary_batch = {}
                summary_batch['acc_top1'] = top1.avg.cpu()
                summary_batch['acc_top5'] = top5.avg.cpu()
                summary_batch['loss'] = losses.avg
                summ.append(summary_batch)

            # Update the average loss
            t.set_postfix(loss='{:05.3f}'.format(losses.avg))
            t.update()

    if logger:
        logger.info("- Train metrics (average): " + matrics_to_string(summ))


class Train:
    def __init__(self, data, save_dir, arch, logger=None, epochs=100, lr=1e-2, lr_sche='MultiStepLR', return_file=True, original_model=None, rate=1, activation='relu', numBins=8, kd_params=(0.0, 1), fixed_seed=False, get_relative_acc=False, use_logger=True):
        self.data_dir = data['dir']
        self.data_name = data['name']
        self.save_dir = save_dir
        if isinstance(arch, str):
            self.arch = None
            self.arch_name = arch
        else:
            self.arch = arch['dir']
            self.arch_name = arch['name']
        self.epochs = epochs
        self.lr = lr
        self.return_file = return_file
        self.original_model = original_model
        self.rate = rate
        self.activation = activation
        self.numBins = numBins
        self.kd_params = kd_params
        self.lr_sche = lr_sche
        self.get_relative_acc = get_relative_acc
        if fixed_seed:
            seed_torch()
        self.use_logger = use_logger
        if self.use_logger == False:
            self.logger = None
        else:
            if logger == None:
                self.logger = set_logger('{}_{}_train'.format(self.data_name, self.arch_name), self.save_dir)
            else:
                self.logger = logger

    def main(self):
        if self.logger:
            self.logger.info(">>>>>> Starting training model")

        if self.arch == None:
            # Create model
            model = models.__dict__[self.arch_name](num_classes=models.get_num_classes(self.data_name), activation=self.activation, numBins=self.numBins, rate=self.rate)
            if self.logger:
                self.logger.info("Created model '{}'".format(self.arch_name))
        else:
            # Load model
            model = torch.load(self.arch)
            if self.logger:
                self.logger.info("Loaded model '{}' from {}".format(
                    self.arch_name, self.arch))
            if self.rate != 1:
                if self.logger:
                    self.logger.warning("Your rate is not working")

        if self.get_relative_acc:
            model_original = model
            metrics_original = test_at_beginning_original(model_original, self.data_name, self.data_dir, self.logger, self.arch_name)

        # Load data
        train_loader, val_loader = models.load_data(self.data_name, self.data_dir, arch_name=self.arch_name)
        if self.logger:
            self.logger.info("Loaded dateset '{}' from '{}'".format(self.data_name, self.data_dir))

        # Finetune
        if self.logger:
            self.logger.info("Entering finetune...")
        trained_model, acc_dict = fine_tune(self.save_dir, model, train_loader, val_loader, 
                                            original_model=self.original_model,
                                            epochs=self.epochs, lr=self.lr, lr_sche=self.lr_sche, logger=self.logger, kd_params=self.kd_params, return_file=True, use_logger=self.use_logger).main()
        model = torch.load(trained_model)
        metric = {}
        metric['Params'], metric['FLOPs'] = get_model_num_param_flops(model)
        if self.logger:
            self.logger.info('$ Num of parameters: {}'.format(metric['Params']))
            self.logger.info('$ FLOPs: {}'.format(metric['FLOPs']))
        save_result_to_json(self.save_dir, metric)

        if self.get_relative_acc:
            acc_dict, _, _, _ = calc_result(model_original, metrics_original, model, acc_dict, None, None)

        if self.use_logger == True:
            close_logger()
        if self.return_file:
            return acc_dict, trained_model
        else:
            cmd = "rm -rf " + trained_model
            os.system(cmd)
            return acc_dict, model

'''
if __name__ == '__main__':
    arch_name = 'resnet20'
    data = {'dir': './data', 'name': 'cifar10'}
    save_dir = './snapshots/{}/train/'.format(arch_name)
    arch = arch_name
    t = Train(data, save_dir, arch, epochs=2)
    print(t.main())
'''