# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import argparse
import torch
import logging
import random
import time
import numpy as np
import torch.backends.cudnn as cudnn

from torch import nn as nn
from time import strftime, localtime
from importlib import import_module
from flyai.dataset import Dataset

from ResNet import args
from ResNet.util.util import Util, AverageMeter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Instructor(object):
    """
    特点： 使用flyai字典的get all data | flyai提供的next batch
    """

    def __init__(self, args):
        self.args = args

        self.dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH, val_batch=args.BATCH)

    @staticmethod
    def getModel(model_name, **kwargs):
        m = import_module('models.' + model_name)
        model = m.createModel(**kwargs)
        if model_name.startswith('alexnet') or model_name.startswith('vgg'):
            model.featrue = nn.DataParallel(module=model.feature)
            model = model.to(DEVICE)
        else:
            model = nn.DataParallel(module=model).to(DEVICE)

        return model

    @staticmethod
    def getOptimizer(model, args):
        if args.optimizer == 'sgd':
            return torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum,
                                   nesterov=args.nesterov, weight_decay=args.weight_decay)
        elif args.optimizer == 'rmsprop':
            return torch.optim.RMSprop(params=model.parameters(), lr=args.lr, alpha=args.alpha,
                                       weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            return torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                                    weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

    def run(self):
        best_err1 = 100.
        best_epoch = 0

        self.args.tensorboard = False
        logger.info('==> creating model "{}"'.format(args.model_name))
        model = Instructor.getModel(**vars(args))

        # 大部分情况下，设置这个flag可以让内置的cuDNN的auto - tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        cudnn.benchmark = True
        # define loss function (criterion) and pptimizer
        criterion = nn.CrossEntropyLoss().to(DEVICE)

        # define optimizer
        optimizer = Instructor.getOptimizer(model=model, args=self.args)

        for epoch in range(0, self.args.EPOCHS):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            # switch to train mode
            model.train()

            lr = Util.adjust_learning_rate(optimizer=optimizer, lr_init=self.args.lr,
                                           decay_rate=self.args.decay_rate, epoch=epoch, num_epochs=self.args.EPOCHS)

            logger.info('Epoch: {0}  lr: {1}'.format(epoch, lr))

            end = time.time()
            for step in range(0, self.dataset.get_step() // self.args.EPOCHS):
                x_train, y_train = self.dataset.next_train_batch()
                x_val, y_val = self.dataset.next_validation_batch()

                train_loader = torch.utils.data.DataLoader(dataset=Util.draw_image(list_dirs=x_train, targets=y_train),
                                                           batch_size=self.args.BATCH,
                                                           shuffle=True,
                                                           num_workers=self.args.num_workers,
                                                           pin_memory=True)
                val_loader = torch.utils.data.DataLoader(dataset=Util.draw_image(list_dirs=x_val, targets=y_val),
                                                         batch_size=self.args.BATCH,
                                                         shuffle=True,
                                                         num_workers=self.args.num_workers,
                                                         pin_memory=True)
                # inputs_shape: (4, 480, 640, 3)
                inputs, targets = train_loader.dataset
                # inputs_shape: (4, 3, 480, 640)
                inputs = inputs.transpose((0, 3, 1, 2))
                inputs = torch.from_numpy(inputs)
                targets = torch.from_numpy(targets)

                # measure data loading time
                data_time.update(time.time() - end)

                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                # compute outputs
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # measure error and record loss
                err1, err5 = Util.error(outputs.data, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(err1.item(), inputs.size(0))
                top5.update(err5.item(), inputs.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if self.args.print_freq > 0 and \
                        (i + 1) % self.args.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.avg:.3f}\t'
                          'Data {data_time.avg:.3f}\t'
                          'Loss {loss.val:.4f}\t'
                          'Err@1 {top1.val:.4f}\t'
                          'Err@5 {top5.val:.4f}'.format(
                        epoch, i + 1, len(train_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1, top5=top5))

            print('Epoch: {:3d} Train loss {loss.avg:.4f} '
                  'Err@1 {top1.avg:.4f}'
                  ' Err@5 {top5.avg:.4f}'
                  .format(epoch, loss=losses, top1=top1, top5=top5))

            return losses.avg, top1.avg, top5.avg, lr

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CV')
    parser.add_argument('-e', '--EPOCHS', default=50, type=int, help='train epochs')
    parser.add_argument('-b', '--BATCH', default=4, type=int, help='batch size')
    config = parser.parse_args()

    args.EPOCHS = config.EPOCHS
    args.BATCH = config.BATCH

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.log_dir) is False:
        os.makedirs(args.log_dir)

    log_file = '{}-{}.log'.format(args.model_name, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, log_file)))

    instructor = Instructor(args=args)
    instructor.run()

# best_score = 0
# for step in range(dataset.get_step()):
#     x_train, y_train = dataset.next_train_batch()
#     x_val, y_val = dataset.next_validation_batch()
