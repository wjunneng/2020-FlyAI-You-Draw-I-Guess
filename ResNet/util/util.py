# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import json
import copy
import pandas as pd
from matplotlib import pyplot as plt


class Util(object):

    @staticmethod
    def draw_image(image_dirs, output_draws_dir, input_labels_dir):
        """
        绘制图片
        :param image_dir:
        :return:
        """
        # json_path, label
        path_label = pd.read_csv(filepath_or_buffer=input_labels_dir, encoding='utf-8')

        path_label = dict(zip([os.path.basename(i) for i in path_label['json_path']], path_label['label']))

        if os.path.exists(output_draws_dir) is False:
            os.makedirs(output_draws_dir)

        for index, image_dir in enumerate(os.listdir(image_dirs)):
            with open(file=os.path.join(image_dirs, image_dir), mode='r', encoding='utf-8') as file:
                json_data = json.load(file)

                drawing_data = json_data['drawing']
                for index, item in enumerate(drawing_data):
                    plt.plot(item[0], [0 - i + 256 for i in item[1]])

                plt.axis('off')
                # plt.title(image_dir)
                plt.savefig(os.path.join(output_draws_dir, str(image_dir.split('.')[0]) + '.jpg'))
                plt.show()

        return True

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


if __name__ == '__main__':
    from ResNet import args

    Util.draw_image(image_dirs=args.input_draws_dir, output_draws_dir=args.output_draws_dir,
                    input_labels_dir=args.input_labels_dir)
