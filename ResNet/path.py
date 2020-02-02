# # -*- coding: utf-8 -*
# import sys
#
# import os
#
# # 训练数据的路径
# DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# # 模型保存的路径
# MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# # 训练log的输出路径
# LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')




# import matplotlib.pyplot as plt
# import numpy as np
#
# p1 = [0, 14, 18, 25, 28, 49, 60, 78, 87, 103, 106, 116, 139, 161, 174, 174, 196, 201, 214, 224, 238, 253, 255]  # 数据点
# p2 = [0, 30, 32, 28, 30, 7, 27, 40, 39, 25, 17, 25, 37, 41, 35, 29, 14, 33, 47, 42, 29, 9, 17]
#
#
# # p1 = [6, 0, 7, 17, 81, 87, 93, 110, 137, 157, 162, 161]
# # p2 = [244, 253, 231, 160, 0, 85, 117, 157, 200, 248, 255, 241]
# # p3 = [37, 56, 57, 67]
# # p4 = [102, 101, 115, 119]
#
#
# # 创建绘图图表对象，可以不显式创建，跟cv2中的cv2.namedWindow()用法差不多
# plt.figure('Draw')
#
# plt.plot(p2, p1)  # plot绘制折线图
# # plt.plot(p3, p4)
#
# plt.draw()  # 显示绘图
#
# plt.show()