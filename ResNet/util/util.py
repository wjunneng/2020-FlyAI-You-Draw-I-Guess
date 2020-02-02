# -*- coding:utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import json
from matplotlib import pyplot as plt


class Util(object):
    @staticmethod
    def draw_image(image_dirs):
        """
        绘制图片
        :param image_dir:
        :return:
        """
        for index, image_dir in enumerate(os.listdir(image_dirs)):
            with open(file=os.path.join(image_dirs, image_dir), mode='r', encoding='utf-8') as file:
                json_data = json.load(file)

                drawing_data = json_data['drawing']
                for index, item in enumerate(drawing_data):
                    plt.plot(item[0], [0 - i + 256 for i in item[1]])

                plt.title(os.path.basename(image_dir))
                plt.show()

        return True

    @staticmethod
    def generate_binary_image(image_dirs):
        for index, image_dir in enumerate(os.listdir(image_dirs)):
            with open(file=os.path.join(image_dirs, image_dir), mode='r', encoding='utf-8') as file:
                json_data = json.load(file)

                drawing_data = json_data['drawing']




if __name__ == '__main__':
    from CNN import args

    # Util.draw_image(image_dirs=args.draws_dir)
