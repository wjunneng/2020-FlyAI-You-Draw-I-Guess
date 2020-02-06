# -*- coding: utf-8 -*
import os
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
        self.model = Util.getModel(**vars(self.args))
        self.model.load_state_dict(torch.load(os.path.join(self.args.output_models_dir, 'checkpoint.pkl')))

    def predict(self, **data):
        self.model.eval()
        x_data = self.data.predict_data(**data)

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

        return torch.argmax(outputs).tolist()

    def predict_all(self, datas):
        labels = []
        for data in datas:
            predicts = self.predict(json_path=data['json_path'])

            labels.append(predicts)

        return labels
