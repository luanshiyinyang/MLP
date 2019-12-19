"""
Author: Zhou Chen
Date: 2019/12/4
Desc: 优化器模块
"""
import numpy as np


class SGD(object):
    def __init__(self, lr=0.001):
        self.lr = lr

    def optimize(self, weight_num, params, grads, batch_size, bn=False):
        for i in range(1, weight_num + 1):
            params['w' + str(i)] -= self.lr * grads['dw' + str(i)] / batch_size
            if bn:
                params['gamma'+str(i)] -= grads['dgamma'+str(i)] / batch_size
                params['beta'+str(i)] -= grads['dbeta'+str(i)] / batch_size
        return params


class Adam:
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8, weights=None, weight_num=None):
        self.lr = lr
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = dict()
        self.v = dict()
        for i in range(1, weight_num+1):
            self.m['m'+str(i)] = np.zeros(shape=weights['w'+str(i)].shape)
            self.v['v'+str(i)] = np.zeros(shape=weights['w'+str(i)].shape)

    def optimize(self, weight_num, params, grads, batch_size=64):
        self.t += 1
        # 逐层优化
        for i in range(1, weight_num + 1):
            w = params['w' + str(i)]
            g = grads['dw'+str(i)] / batch_size  # theta t -1
            self.m['m'+str(i)] = self.beta1 * self.m['m'+str(i)] + (1 - self.beta1) * g
            self.v['v'+str(i)] = self.beta2 * self.v['v'+str(i)] + (1 - self.beta2) * (g**2)
            m_hat = self.m['m'+str(i)] / (1 - self.beta1 ** self.t)  # 由于mt初始化为0，需要对训练初期的梯度均值mt进行纠正
            v_hat = self.v['v'+str(i)] / (1 - self.beta2 ** self.t)  # 同上
            w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            params['w'+str(i)] = w
        return params
