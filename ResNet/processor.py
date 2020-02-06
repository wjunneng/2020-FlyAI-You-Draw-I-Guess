# -*- coding: utf-8 -*

import numpy
from flyai.processor.base import Base


class Processor(Base):
    def input_x(self, json_path):
        """
        参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。可在该方法中做数据增强
        该方法字段与app.yaml中的input:->columns:对应
        """
        return json_path

    def input_y(self, label):
        """
        参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。
        该方法字段与app.yaml中的output:->columns:对应
        """
        return label

    def output_x(self, json_path):
        """
        参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
        和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
        该方法字段与app.yaml中的input:->columns:对应
        """
        return json_path

    def output_y(self, data):
        """
        输出的结果，会被dataset.to_categorys(data)调用
        """
        return numpy.argmax(data)
