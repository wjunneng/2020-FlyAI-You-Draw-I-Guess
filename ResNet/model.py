# -*- coding: utf-8 -*
import os
import math
import torch
import pickle
import torch.utils.data
from flyai.model.base import Base

import args
from util import Util, ImageDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.args = args
        self.model = Util.getModel(**vars(self.args)).to(DEVICE)
        self.model.load_state_dict(torch.load(os.path.join(self.args.output_models_dir, 'checkpoint.pkl')))
        with open(self.args.min_threshold_dir, 'rb') as file:
            self.min_threshold = pickle.load(file=file)


    @staticmethod
    def limit_output(min_threshold, item, threshold):
        item_dict = dict(zip(range(40), item))
        # 按照预测值从大到小排序
        item_dict_sorted = sorted(item_dict.items(), key=lambda item_dict: item_dict[1], reverse=True)

        result = item_dict_sorted[0][0]
        index = 0
        print(threshold, item_dict_sorted)
        for (key, value) in item_dict_sorted:
            if index >= 2:
                index += 1
                break

            if min_threshold[key] <= (threshold+1):
                result = key
                break

        return result


    def do_predict(self, x_data):
        data, _, stroke = Util.draw_image(list_dirs=x_data, targets=None)

        predict_loader = torch.utils.data.DataLoader(dataset=ImageDataset(data=data, targets=None),
                                                     batch_size=len(x_data),
                                                     num_workers=self.args.num_workers,
                                                     pin_memory=True)

        for step, data in enumerate(predict_loader):
            # inputs_shape: (4, 480, 640, 3) -> inputs_shape: (4, 3, 480, 640)
            inputs = data['inputs'].permute((0, 3, 1, 2)).to(DEVICE).float()

            # compute outputs
            outputs = self.model(inputs)

            if self.args.predict_batch:
                # result = []
                # for index in range(len(stroke)):
                #     result.append(Model.limit_output(min_threshold=self.min_threshold, item=outputs.tolist()[index], threshold=stroke[index]))
                #
                # return result
                return torch.argmax(outputs, dim=-1).tolist()
            else:
                return torch.argmax(outputs).tolist()


    def predict(self, **data):
        x_data = self.data.predict_data(**data)

        return self.do_predict(x_data)

    def predict_all(self, datas):
        labels = []

        if self.args.predict_batch:
            for i in range(math.ceil(len(datas) / self.args.BATCH)):
                batch_x_data = [datas[i]['json_path'] for i in range(i*self.args.BATCH, min((i+1)*self.args.BATCH, len(datas)))]
                labels.extend(self.do_predict(batch_x_data))
        else:
            for data in datas:
                predicts = self.predict(json_path=data['json_path'])
                labels.append(predicts)

        return labels
