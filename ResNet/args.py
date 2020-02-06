# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])

# ############################################# paths
# current dir
# current_dir = os.path.abspath('..')
current_dir = os.getcwd()
# data dir
data_dir = os.path.join(current_dir, 'data')
# input dir
input_dir = os.path.join(data_dir, 'input')
# input draws dir
input_draws_dir = os.path.join(input_dir, 'draws')
# input labels dir
input_labels_dir = os.path.join(input_dir, 'dev.csv')
# log dir
log_dir = os.path.join(input_dir, 'log')

# output dir
output_dir = os.path.join(data_dir, 'output')
# output draws dir
output_draws_dir = os.path.join(output_dir, 'draws')

# ############################################# params
# 模型名称
model_name = 'resnet'
# 类别dict
class_mapping = {0: "airplane",  # 飞机
                 1: "apple",  # 苹果
                 2: "basketball",  # 篮球
                 3: "bear",  # 熊
                 4: "bed",  # 床
                 5: "bicycle",  # 自行车
                 6: "bridge",  # 桥
                 7: "camera",  # 相机
                 8: "car",  # 汽车
                 9: "cat",  # 猫
                 10: "computer",  # 计算机
                 11: "cow",  # 牛
                 12: "cup",  # 杯子
                 13: "dog",  # 狗
                 14: "door",  # 门
                 15: "eye",  # 眼睛
                 16: "fish",  # 鱼
                 17: "flower",  # 花
                 18: "frog",  # 青蛙
                 19: "guitar",  # 吉他
                 20: "hat",  # 帽子
                 21: "horse",  # 马
                 22: "hospital",  # 医院
                 23: "hourglass",  # 沙漏
                 24: "ice cream",  # 冰激凌
                 25: "key",  # 钥匙
                 26: "knife",  # 刀
                 27: "lantern",  # 灯笼
                 28: "lion",  # 狮子
                 29: "mailbox",  # 邮箱
                 30: "matches",  # 火柴
                 31: "microphone",  # 麦克风
                 32: "monkey",  # 猴子
                 33: "mosquito",  # 蚊子
                 34: "mountain",  # 山
                 35: "mushroom",  # 蘑菇
                 36: "ocean",  # 海洋
                 37: "panda",  # 熊猫
                 38: "piano",  # 钢琴
                 39: "pig"}  # 猪
# 类别数目
num_classes = len(class_mapping)
# 随机种子
seed = 42
# 工作节点
num_workers = 0
# 频率
print_freq = 1
# 分辨率
dpi = 64

# ############################################# model params
# depth (default=56)
depth = 56
# dropout rate (default: 0.2)
drop_rate = 0.0
# ['none', 'linear', 'uniform'] death mode for stochastic depth (default: none)
death_mode = 'none'
# death rate rate (default: 0.5)
death_rate = 0.5
# Growth rate of DenseNet (default: 12)
growth_rate = 12
# bottle neck ratio of DenseNet (0 means dot\'t use bottle necks) (default: 4)
bn_size = 4
# compression ratio of DenseNet (1 means dot\'t use compression) (default: 0.5)
compression = 0.5

# ############################################# training related
# trainer file name without ".py" (default: train)
trainer = 'train'
# 'number of total epochs to run (default: 164)'
epochs = 164
# 'manual epoch number (useful on restarts)'
start_epoch = 1
# patience for early stopping (0 means no early stopping)
patience = 0
# mini-batch size (default: 64)
batch_size = 64
# choices=['sgd', 'rmsprop', 'adam'] optimizer (default=sgd)
optimizer = 'sgd'
# initial learning rate (default: 0.1)
lr = 0.1
# decay rate of learning rate (default: 0.1)
decay_rate = 0.1
# momentum (default=0.9)
momentum = 0.9
# action='store_false' do not use Nesterov momentum
nesterov = False
# alpha for
alpha = 0.99
# beta1 for Adam (default: 0.9)
beta1 = 0.9
# beta2 for Adam (default: 0.999)
beta2 = 0.999
# weight decay (default: 1e-4)
weight_decay = 1e-4
