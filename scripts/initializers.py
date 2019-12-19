"""
Author: Zhou Chen
Date: 2019/12/4
Desc: 参数初始化器
"""
import numpy as np


def xavier(num_layers, units_list):
    """
    保证参数是服从[-a, a]的均匀分布，其中a为sqrt(6)/sqrt(n_in+n_out)
    :param num_layers:
    :param units_list:
    :return:
    """
    params = {}
    np.random.seed(2019)
    for layer in range(1, num_layers):
        # 第一层输入层无参数，之后每层都有参数
        a = np.sqrt(6) / np.sqrt(units_list[layer-1] + units_list[layer])
        params['w'+str(layer)] = np.random.uniform(-a, a, size=(units_list[layer], units_list[layer-1]))
        params['gamma' + str(layer)] = np.ones(shape=(1, units_list[layer]))
        params['beta' + str(layer)] = np.zeros(shape=(1, units_list[layer]))
    return params


def zero(num_layers, units_list):
    """
    全0初始化
    :param num_layers:
    :param units_list:
    :return:
    """
    params = {}
    for layer in range(1, num_layers):
        # 第一层输入层无参数，之后每层都有参数
        params['w' + str(layer)] = np.zeros(shape=(units_list[layer], units_list[layer - 1]))
        params['gamma'+str(layer)] = np.ones(shape=(1, units_list[layer]))
        params['beta' + str(layer)] = np.zeros(shape=(1, units_list[layer]))
    return params
