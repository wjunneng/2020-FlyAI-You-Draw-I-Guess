# -*- coding:utf-8 -*-

from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
x_test = [
          # 29
          {'json_path': 'draws/draw_239207.json'},
          # 39
          {'json_path': 'draws/draw_429191.json'},
          # 38
          {'json_path': 'draws/draw_189496.json'},
          # 23
          {'json_path': 'draws/draw_23410.json'},
          # 7
          {'json_path': 'draws/draw_146414.json'},
          # 1
          {'json_path': 'draws/draw_122639.json'},
          # 16
          {'json_path': 'draws/draw_561918.json'},
          # 17
          {'json_path': 'draws/draw_485396.json'},
          # 35
          {'json_path': 'draws/draw_295131.json'},
          # 32
          {'json_path': 'draws/draw_52884.json'}]
p = model.predict_all(x_test)
print(p)
