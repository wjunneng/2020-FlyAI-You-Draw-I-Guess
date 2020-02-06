# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import argparse
import torch
import logging
import random
import numpy as np
import torch.backends.cudnn as cudnn

from torch import nn
from time import strftime, localtime
from flyai.dataset import Dataset

import args
from util import Util, Trainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(DEVICE)


class Instructor(object):
    """
    特点： 使用flyai字典的get all data | flyai提供的next batch
    """

    def __init__(self, args):
        self.args = args

        self.dataset = Dataset(epochs=self.args.EPOCHS, batch=self.args.BATCH, val_batch=self.args.BATCH)

    def run(self):
        best_err1 = 100.
        best_epoch = 0

        self.args.tensorboard = False
        logger.info('==> creating model "{}"'.format(args.model_name))
        model = Util.getModel(**vars(args))

        # 大部分情况下，设置这个flag可以让内置的cuDNN的auto - tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        cudnn.benchmark = True
        # define loss function (criterion) and pptimizer
        criterion = nn.CrossEntropyLoss().to(DEVICE)

        # define optimizer
        optimizer = Util.getOptimizer(model=model, args=self.args)

        trainer = Trainer(dataset=self.dataset, model=model, criterion=criterion, optimizer=optimizer, args=self.args,
                          logger=logger)
        for epoch in range(0, self.args.EPOCHS):
            # train for one epoch
            trainer.train(epoch=epoch)

            # evaluate on validation set
            val_loss, val_err1, val_err5 = trainer.test(epoch=epoch)

            # remember best err@1 and save checkpoint
            is_best = val_err1 < best_err1
            if is_best:
                best_err1 = val_err1
                best_epoch = epoch
                logger.info('Best var_err1 {}'.format(best_err1))
            Util.save_checkpoint(model.state_dict(), is_best, args.output_models_dir)
            if not is_best and epoch - best_epoch >= args.patience > 0:
                break

        logger.info('Best val_err1: {:.4f} at epoch {}'.format(best_err1, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CV')
    parser.add_argument('-e', '--EPOCHS', default=2, type=int, help='train epochs')
    parser.add_argument('-b', '--BATCH', default=2, type=int, help='batch size')
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
