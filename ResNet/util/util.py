# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import json
import copy
import time
import torch
import shutil
import numpy as np
import torch.utils.data

from torch import nn
from PIL import Image
from importlib import import_module
from matplotlib import pyplot as plt

from ResNet import args

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
            plt.figure()
            with open(file=os.path.join(image_dirs, image_dir), mode='r', encoding='utf-8') as file:
                json_data = json.load(file)

                drawing_data = json_data['drawing']
                for item in drawing_data:
                    plt.plot(item[0], [0 - i + 256 for i in item[1]], marker='o', linestyle='solid')

                plt.axis('off')
                save_path = os.path.join(output_draws_dir, str(str(image_dir.split('/')[-1]).split('.')[0]) + '.jpg')
                plt.savefig(save_path, bbox_inches='tight')
                image = Image.open(save_path).resize((args.dpi, args.dpi), Image.ANTIALIAS)
                data.append(np.asarray(image))
            plt.show()

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

    @staticmethod
    def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pkl'):
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

        filename = os.path.join(save_dir, filename)
        torch.save(obj=state, f=filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(save_dir, 'model_best.pkl'))

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


class Trainer(object):
    def __init__(self, dataset, model, criterion=None, optimizer=None, args=None, logger=None):
        self.dataset = dataset
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.logger = logger

    def train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        lr = Util.adjust_learning_rate(optimizer=self.optimizer, lr_init=self.args.lr,
                                       decay_rate=self.args.decay_rate, epoch=epoch, num_epochs=self.args.EPOCHS)

        self.logger.info('Epoch: {0}  lr: {1}'.format(epoch, lr))

        end = time.time()
        for step in range(0, self.dataset.get_step() // self.args.EPOCHS):
            x_train, y_train = self.dataset.next_train_batch()

            train_loader = torch.utils.data.DataLoader(dataset=Util.draw_image(list_dirs=x_train, targets=y_train),
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

            inputs = inputs.to(DEVICE).float()
            targets = targets.to(DEVICE).long()

            # compute outputs
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # measure error and record loss
            err1, err5 = Util.error(outputs.data, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(err1.item(), inputs.size(0))
            top5.update(err5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)

            end = time.time()

            if self.args.print_freq > 0 and (step + 1) % self.args.print_freq == 0:
                self.logger.info('Epoch: [{0}][{1}/{2}]\t'
                                 'Time {batch_time.avg:.3f}\t'
                                 'Data {data_time.avg:.3f}\t'
                                 'Loss {loss.val:.4f}\t'
                                 'Err@1 {top1.val:.4f}\t'
                                 'Err@5 {top5.val:.4f}'.format(epoch, step + 1, len(train_loader),
                                                               batch_time=batch_time,
                                                               data_time=data_time, loss=losses, top1=top1, top5=top5))

        self.logger.info(
            'Epoch: {:3d} Train loss {loss.avg:.4f} Err@1 {top1.avg:.4f} Err@5 {top5.avg:.4f}'.format(epoch,
                                                                                                      loss=losses,
                                                                                                      top1=top1,
                                                                                                      top5=top5))
        return losses.avg, top1.avg, top5.avg, lr

    def test(self, epoch, silence=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            for step in range(0, self.dataset.get_step() // self.args.EPOCHS):
                x_val, y_val = self.dataset.next_validation_batch()
                val_loader = torch.utils.data.DataLoader(dataset=Util.draw_image(list_dirs=x_val, targets=y_val),
                                                         batch_size=self.args.BATCH,
                                                         shuffle=True,
                                                         num_workers=self.args.num_workers,
                                                         pin_memory=True)

                # inputs_shape: (4, 480, 640, 3)
                inputs, targets = val_loader.dataset
                # inputs_shape: (4, 3, 480, 640)
                inputs = inputs.transpose((0, 3, 1, 2))
                inputs = torch.from_numpy(inputs).to(DEVICE).float()
                targets = torch.from_numpy(targets).to(DEVICE).long()

                # compute outputs
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # measure error and record loss
                err1, err5 = Util.error(outputs.data, targets, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(err1.item(), inputs.size(0))
                top5.update(err5.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        if not silence:
            print('Epoch: {:3d} val   loss {loss.avg:.4f} Err@1 {top1.avg:.4f}'
                  ' Err@5 {top5.avg:.4f}'.format(epoch, loss=losses,
                                                 top1=top1, top5=top5))

        return losses.avg, top1.avg, top5.avg
