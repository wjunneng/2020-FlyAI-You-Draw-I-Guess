# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# ############################################# paths
# current dir
current_dir = os.path.abspath('..')
# data dir
data_dir = os.path.join(current_dir, 'data')
# input dir
input_dir = os.path.join(data_dir, 'input')
# draws dir
draws_dir = os.path.join(input_dir, 'draws')
# labels dir
labels_dir = os.path.join(input_dir, 'dev.csv')

# output dir
output_dir = os.path.join(data_dir, 'output')

# ############################################# params
# 类别dict
class_mapping = {0: "airplane",         # 飞机
                 1: "apple",            # 苹果
                 2: "basketball",       # 篮球
                 3: "bear",             # 熊
                 4: "bed",              # 床
                 5: "bicycle",          # 自行车
                 6: "bridge",           # 桥
                 7: "camera",           # 相机
                 8: "car",              # 汽车
                 9: "cat",              # 猫
                 10: "computer",        # 计算机
                 11: "cow",             # 牛
                 12: "cup",             # 杯子
                 13: "dog",             # 狗
                 14: "door",            # 门
                 15: "eye",             # 眼睛
                 16: "fish",            # 鱼
                 17: "flower",          # 花
                 18: "frog",            # 青蛙
                 19: "guitar",          # 吉他
                 20: "hat",             # 帽子
                 21: "horse",           # 马
                 22: "hospital",        # 医院
                 23: "hourglass",       # 沙漏
                 24: "ice cream",       # 冰激凌
                 25: "key",             # 钥匙
                 26: "knife",           # 刀
                 27: "lantern",         # 灯笼
                 28: "lion",            # 狮子
                 29: "mailbox",         # 邮箱
                 30: "matches",         # 火柴
                 31: "microphone",      # 麦克风
                 32: "monkey",          # 猴子
                 33: "mosquito",        # 蚊子
                 34: "mountain",        # 山
                 35: "mushroom",        # 蘑菇
                 36: "ocean",           # 海洋
                 37: "panda",           # 熊猫
                 38: "piano",           # 钢琴
                 39: "pig"}             # 猪

print(class_mapping.values())