"""
Author: Zhou Chen
Date: 2019/12/4
Desc: 激活函数及其梯度
"""
import numpy as np


def sigmoid(x):
    """
    Sigmoid激活函数
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x):
    """
    Sigmoid梯度
    :param x:
    :return:
    """
    return sigmoid(x) * (1-sigmoid(x))


def tanh(x):
    """
    Tanh激活函数
    :param x:
    :return:
    """
    # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return np.tanh(x)


def tanh_gradient(x):
    """
    Tanh梯度
    :param x:
    :return:
    """
    return 1 - tanh(x) ** 2


def softmax(x):
    """
    Softmax激活函数
    :param x:
    :return:
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    f1 = lambda x: np.exp(x - np.max(x))  # 对每个样本减去10个输出最大值再softmax以防止溢出
    f2 = lambda x: x / np.sum(x)  # 将每个值概率化
    # 沿着行的维度应用上述函数变化
    x = np.apply_along_axis(f1, axis=1, arr=x)
    x = np.apply_along_axis(f2, axis=1, arr=x)
    return x


def softmax_gradient(x, label):
    """
    Softmax梯度
    :param x:
    :param label
    :return:
    """
    return softmax(x) - label


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4], [0, 2, 3, 4]])
    print(softmax(a))
    np.sum(softmax(a))
