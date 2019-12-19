"""
Author: Zhou Chen
Date: 2019/12/4
Desc: 数据生成模块
"""

import numpy as np
np.random.seed(2019)  # 固定随机数种子


class DataLoader(object):

    def __init__(self, batch_size=1000, data_type='train', scale=False):
        data = np.load('../data/data.npz')
        self.data_size = data['x'].shape[0]
        index = np.random.permutation(np.arange(self.data_size))  # 打乱后的下标
        data_x, data_y = data['x'], data['y']
        data_X = data_x[index, ...]
        data_Y = data_y[index, ...]

        if data_type == 'train':
            self.data_x, self.data_y = data_X[:int(0.8*self.data_size)], data_Y[:int(0.8*self.data_size)]
        else:
            self.data_x, self.data_y = data_X[int(0.8*self.data_size):], data_Y[int(0.8*self.data_size):]
        self.data_type = data_type
        self.data_num = len(self.data_y)
        self.index = 0
        self.batch_size = batch_size
        self.end = False
        self.scale = scale

    def __iter__(self):
        return self

    def __next__(self):
        if self.end:
            raise StopIteration
        else:

            start = self.index
            end = self.index + self.batch_size
            if end <= self.data_num:
                # 若后面还有batch_size个数据则继续取，否则结束迭代
                self.index += self.batch_size
                x, y = self.data_x[start:end, ...], self.data_y[start:end, ...]
                if self.scale:
                    x = x / 255.
                return x, y
            else:
                # 从头开始取，保证整个迭代器不会结束
                if self.data_type == 'test':
                    # 训练可以丢弃，测试必须取到结束
                    x, y = self.data_x[start:self.data_num, ...], self.data_y[start:self.data_num, ...]
                    self.end = True
                    if self.scale:
                        x = x / 255.
                    return x, y
                else:
                    raise StopIteration

    def restart(self):
        """
        将迭代下标置零，从头开始取新的数据并打乱数据，保证迭代不会停止
        """
        self.index = 0


if __name__ == '__main__':
    gen = DataLoader()
    for epoch in range(10):
        print("epoch")
        for step, (x, y) in enumerate(gen):
            print(step, x.shape, y.shape)
        gen.restart()