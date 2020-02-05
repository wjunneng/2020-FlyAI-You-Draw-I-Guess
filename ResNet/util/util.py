# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import json
import copy
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mimage
from torch.utils.data import Dataset

from ResNet import args


class Util(object):

    @staticmethod
    def draw_image(list_dirs=None, targets=None, image_dirs=args.input_dir, output_draws_dir=args.output_draws_dir):
        """
        绘制图片
        :param image_dir:
        :return:
        """
        if os.path.exists(output_draws_dir) is False:
            os.makedirs(output_draws_dir)

        data = []
        if list_dirs is None:
            list_dirs = os.listdir(image_dirs)
            image_dirs = args.input_draws_dir

        for image_dir in list_dirs:
            with open(file=os.path.join(image_dirs, image_dir), mode='r', encoding='utf-8') as file:
                json_data = json.load(file)

                drawing_data = json_data['drawing']
                for item in drawing_data:
                    plt.plot(item[0], [0 - i + 256 for i in item[1]])

                plt.axis('off')
                save_path = os.path.join(output_draws_dir, str(str(image_dir.split('/')[-1]).split('.')[0]) + '.jpg')
                plt.savefig(save_path)
                data.append(mimage.imread(save_path))

        data = np.asarray(data)

        return data, targets

    @staticmethod
    def adjust_learning_rate(optimizer, lr_init, decay_rate, epoch, num_epochs):
        """
        Decay Learning rate at 1/2 and 3/4 of the num_epochs
        :param optimizer:
        :param lr_init:
        :param decay_rate:
        :param epoch:
        :param num_epoch:
        :return:
        """
        lr = copy.deepcopy(lr_init)
        if epoch >= num_epochs * 0.75:
            lr *= decay_rate ** 2
        elif epoch >= num_epochs * 0.5:
            lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    @staticmethod
    def error(output, target, topk=(1,)):
        """
        Computes the error@k for the specified values of k
        :param output:
        :param target:
        :param topk:
        :return:
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0)
                res.append(100.0 - correct_k.mul_(100.0 / batch_size))

        return res


class AverageMeter(object):
    """
    Computes and stores the averagte and current value
    """

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

# if __name__ == '__main__':
#     from ResNet import args
#
#     Util.draw_image(image_dirs=args.input_draws_dir, output_draws_dir=args.output_draws_dir)
