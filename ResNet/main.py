# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

import argparse
import torch
import logging
from torch import nn as nn
import random
import numpy as np
from time import strftime, localtime
from importlib import import_module
from flyai.dataset import Dataset

from ResNet import args

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

    @staticmethod
    def getModel(arch, **kwargs):
        m = import_module('models.' + arch)
        model = m.createModel(**kwargs)
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.featrue = nn.DataParallel(module=model.feature)
            model = model.to(DEVICE)
        else:
            model = nn.DataParallel(module=model).to(DEVICE)

        return model

    def run(self):
        pass


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
