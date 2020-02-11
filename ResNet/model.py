# -*- coding: utf-8 -*
import os
import math
import torch
import torch.utils.data
from flyai.model.base import Base

import args
from util import Util

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(Base):
    def __init__(self, data):
        self.data = data
        self.args = args
        self.model = Util.getModel(**vars(self.args)).to(DEVICE)
        self.model.load_state_dict(torch.load(os.path.join(self.args.output_models_dir, 'checkpoint.pkl')))

    def do_predict(self, x_data):
        predict_loader = torch.utils.data.DataLoader(dataset=Util.draw_image(list_dirs=x_data, targets=None),
                                                     batch_size=1,
                                                     num_workers=self.args.num_workers,
                                                     pin_memory=True)

        # inputs_shape: (4, 480, 640, 3)
        inputs, _ = predict_loader.dataset
        # inputs_shape: (4, 3, 480, 640)
        inputs = inputs.transpose((0, 3, 1, 2))
        inputs = torch.from_numpy(inputs)

        inputs = inputs.to(DEVICE).float()

        # compute outputs
        outputs = self.model(inputs)

        if self.args.predict_batch:
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
