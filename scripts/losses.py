"""
Author: Zhou Chen
Date: 2019/12/4
Desc: 损失函数
"""
import numpy as np


def cross_entropy(p, q):
    """
    交叉熵损失函数，p表示真实分布，q表示预测分布（必须onehot编码）
    :param p:
    :param q:
    :return:
    """
    if p.ndim == 1 or q.ndim == 1:
        p = p.reshape(1, -1)
        q = q.reshape(1, -1)

    m = p.shape[0]
    # p: [64, 10], q: [64, 10]
    loss = p * np.log(q+1e-5)
    return -np.sum(loss) / m
