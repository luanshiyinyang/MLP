"""
Author: Zhou Chen
Date: 2019/12/4
Desc: 构建模型
"""
import numpy as np
from initializers import xavier, zero
from utils import onehot
from activations import tanh, softmax, softmax_gradient, tanh_gradient
from losses import cross_entropy
from optimizers import SGD, Adam


def dropout(x, p):
    """
    以概率p丢弃神经元连接，为了处理方便采用反向Dropout思路，该方法无需修改测试网络
    """
    keep_prob = 1 - p
    # z这里写的时候犯了一个错误，就是不应该批量生成概率矩阵，而是生成的概率矩阵批量重复
    d_temp = np.random.binomial(1, keep_prob, size=x.shape[1:]) / keep_prob
    d_temp = d_temp.reshape(-1)
    x_dropout = x * d_temp
    return x_dropout, d_temp


class Model(object):

    def __init__(self, num_layers, units_list=None, initializer=None, optimizer='adam'):
        self.weight_num = num_layers - 1
        # 根据传入的初始化方法初始化参数，本次实验只实现xavier和全0初始化
        self.params = xavier(num_layers, units_list) if initializer == 'xavier' else zero(num_layers, units_list)
        self.optimizer = Adam(weights=self.params, weight_num=self.weight_num) if optimizer == 'adam' else SGD()
        self.bn_param = {}

    def forward(self, x, dropout_prob=None):
        """
        前向传播，针对一个mini-batch处理
        """
        net_inputs = []  # 各层的输入
        net_outputs = []  # 各层激活后的输出
        net_d = []
        # 为了层号对应，将输入层直接添加
        net_inputs.append(x)
        net_outputs.append(x)
        net_d.append(np.ones(x.shape[1:]))  # 输入层无丢弃概率
        for i in range(1, self.weight_num):  # 参数数量比层数少1
            x = x @ self.params['w'+str(i)].T
            net_inputs.append(x)
            x = tanh(x)
            if dropout_prob:
                # 训练阶段丢弃
                x, d_temp = dropout(x, dropout_prob)
                net_d.append(d_temp)
            net_outputs.append(x)
        out = x @ self.params['w'+str(self.weight_num)].T
        net_inputs.append(out)
        out = softmax(out)
        net_outputs.append(out)
        return {'net_inputs': net_inputs, 'net_outputs': net_outputs, 'd': net_d}, out

    def backward(self, nets, y, pred, dropout_prob=None):
        """
        dz[out] = out - y
        dw[out] = dz[out] @ outputs[out-1].T
        db[out] = dz[out]
        dz[i] = W[i+1]dz[i+1] * grad(z[i])
        dw[i] = dz[i] @ outputs[i-1]
        db[i] = dz[i]sa

        """
        grads = dict()
        grads['dz'+str(self.weight_num)] = (pred - y)  # [b, 10]
        grads['dw'+str(self.weight_num)] = grads['dz'+str(self.weight_num)].T @ nets['net_outputs'][self.weight_num-1]  #[10, 512]

        for i in reversed(range(1, self.weight_num)):
            temp = grads['dz' + str(i + 1)] @ self.params['w' + str(i + 1)] * tanh_gradient(nets['net_inputs'][i])
            if dropout_prob:
                temp = temp * nets['d'][i] / (1-dropout_prob)
            grads['dz'+str(i)] = temp   # [b, 128]
            grads['dw'+str(i)] = grads['dz'+str(i)].T @ nets['net_outputs'][i-1]
        return grads

    def train(self, data_loader, valid_loader, epochs, learning_rate, dropout_prob=None):
        losses_train = []
        losses_valid = []
        for epoch in range(epochs):
            print("epoch", epoch)
            # 训练部分
            epoch_loss_train = 0
            for step, (x, y) in enumerate(data_loader):
                # x:[b, 28, 28] -> [b, 784] , y:[b, 1] -> [b, 10]
                x = x.reshape(-1, 28 * 28)
                y = onehot(y, 10)
                nets, pred = self.forward(x, dropout_prob)
                loss = cross_entropy(y, pred)
                epoch_loss_train += loss
                grads = self.backward(nets, y, pred, dropout_prob)
                # SGD更新参数
                # self.params = optimizer.optimize(self.weight_num, self.params, grads, y.shape[0])
                self.params = self.optimizer.optimize(self.weight_num, self.params, grads, y.shape[0])

                if step % 100 == 0:
                    print("epoch {} training step {} loss {:.4f}".format(epoch, step, loss))
            losses_train.append(epoch_loss_train)
            print(epoch_loss_train)
            data_loader.restart()
            # 验证部分，只进行前向传播
            epoch_loss_valid = 0
            for step, (x, y) in enumerate(valid_loader):
                x = x.reshape(-1, 28 * 28)
                y = onehot(y, 10)
                nets, pred = self.forward(x, dropout_prob)
                loss = cross_entropy(y, pred)
                epoch_loss_valid += loss

                if step % 100 == 0:
                    print("epoch {} validation step {} loss {:.4f}".format(epoch, step, loss))
            losses_valid.append(epoch_loss_valid)
            valid_loader.restart()
        his = {'train_loss': losses_train, 'valid_loss': losses_valid}
        return his

    def batch_norm(self, x, layer_index, mode):
        epsilon = 1e-6
        momentum = 0.9
        N, D = x.shape
        global_mean = self.bn_param.get('global_mean' + str(layer_index), np.zeros(D, dtype=x.dtype))
        global_var = self.bn_param.get('global_var' + str(layer_index), np.zeros(D, dtype=x.dtype))
        cache = None
        if mode == 'train':
            # 计算当前batch的均值和方差
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)
            x_hat = (x - sample_mean) / np.sqrt(sample_var + epsilon)
            out = self.params['gamma' + str(layer_index)] * x_hat + self.params['beta' + str(layer_index)]  # bn结束
            global_mean = momentum * global_mean + (1 - momentum) * sample_mean
            global_var = momentum * global_var + (1 - momentum) * sample_var
            cache = {'x': x, 'x_hat': x_hat, 'sample_mean': sample_mean, 'sample_var': sample_var}
        else:
            # 测试模式，使用全局均值和方差标准化
            x_hat = (x - global_mean) / np.sqrt(global_var + epsilon)
            out = self.params['gamma' + str(layer_index)] * x_hat + self.params['beta' + str(layer_index)]

        self.bn_param['global_mean' + str(layer_index)] = global_mean
        self.bn_param['global_var' + str(layer_index)] = global_var
        return out, cache

    def forward_bn(self, x, bn_mode='train'):
        """
        带BN层的前向传播
        """

        net_inputs = []
        net_outputs = []
        caches = []
        net_inputs.append(x)
        net_outputs.append(x)
        caches.append(x)

        for i in range(1, self.weight_num):
            # 所有隐层的输入都进行BN，输入层和输出层不进行BN
            x = x = x @ self.params['w'+str(i)].T
            net_inputs.append(x)
            x, cache = self.batch_norm(x, i, bn_mode)  # 可以将BN理解为加在隐藏层神经元输入和输出间可训练的一层
            caches.append(cache)
            x = tanh(x)
            net_outputs.append(x)
        out = x @ self.params['w' + str(self.weight_num)].T
        net_inputs.append(out)
        out = softmax(out)
        net_outputs.append(out)

        return {'net_inputs': net_inputs, 'net_outputs': net_outputs, 'cache': caches}, out

    def backward_bn(self, nets, y, pred):
        """
        加入BN层的反向传播
        """
        epsilon = 1e-6
        momentum = 0.9
        grads = dict()
        # 求解输出层梯度，依据链式法则，无BN
        grads['dz' + str(self.weight_num)] = (pred - y)
        grads['dw' + str(self.weight_num)] = grads['dz' + str(self.weight_num)].T @ nets['net_outputs'][self.weight_num - 1]
        for i in reversed(range(1, self.weight_num)):
            N = nets['cache'][i]['x'].shape[0]
            grads['dz'+str(i)] = grads['dz' + str(i + 1)] @ self.params['w' + str(i + 1)]
            grads['dgamma'+str(i)] = np.sum(grads['dz'+str(i)] * nets['cache'][i]['x_hat'])
            grads['dbeta'+str(i)] = np.sum(grads['dz'+str(i)], axis=0)

            dx_hat = grads['dz'+str(i)] * self.params['gamma'+str(i)]
            dsigma = -0.5 * np.sum(dx_hat * (nets['cache'][i]['x'] - nets['cache'][i]['sample_mean']), axis=0) * np.power(nets['cache'][i]['sample_var'][i] + epsilon, -1.5)
            dmu = -np.sum(dx_hat / np.sqrt(nets['cache'][i]['sample_var'] + epsilon), axis=0) - 2 * dsigma * np.sum(nets['cache'][i]['x'] - nets['cache'][i]['sample_mean'], axis=0) / N
            dx = dx_hat / np.sqrt(nets['cache'][i]['sample_var'] + epsilon) + 2.0 * dsigma * (nets['cache'][i]['x'] - nets['cache'][i]['sample_mean']) / N + dmu / N
            temp = dx * tanh_gradient(nets['net_inputs'][i])
            grads['dw'+str(i)] = temp.T @ nets['net_outputs'][i-1]
        return grads

    def train_bn(self, data_loader, valid_loader, epochs, learning_rate):
        losses_train = []
        losses_valid = []
        for epoch in range(epochs):
            print("epoch", epoch)
            epoch_loss_train = 0
            # 重置全局均值和方差
            # 批量训练
            for step, (x, y) in enumerate(data_loader):
                # x:[b, 28, 28] -> [b, 784] , y:[b, 1] -> [b, 10]
                x = x.reshape(-1, 28 * 28)
                y = onehot(y, 10)
                nets, pred = self.forward_bn(x, bn_mode='train')
                grads = self.backward_bn(nets, y, pred)
                self.optimizer.optimize(self.weight_num, self.params, grads, y.shape[0])
                loss = cross_entropy(y, pred)
                epoch_loss_train += loss
                if step % 100 == 0:
                    print("epoch {} step {} loss {:.4f}".format(epoch, step, loss))
            losses_train.append(epoch_loss_train)
            data_loader.restart()
            print(epoch_loss_train)
            # 验证集测试
            epoch_loss_valid = 0
            for step, (x, y) in enumerate(valid_loader):
                x = x.reshape(-1, 28 * 28)
                y = onehot(y, 10)
                nets, pred = self.forward_bn(x, bn_mode='test')
                loss = cross_entropy(y, pred)
                epoch_loss_valid += loss
                if step % 100 == 0:
                    print("epoch {} step {} loss {:.4f}".format(epoch, step, loss))
            losses_valid.append(epoch_loss_valid)
            valid_loader.restart()
        his = {'train_loss': losses_train, 'valid_loss': losses_valid}

        return his

    def predict(self, data_loader, bn=False):
        labels = []
        pred = []
        losses = 0
        for (x, y) in data_loader:
            x = x.reshape(-1, 28 * 28)
            y = onehot(y, 10)
            if bn:
                _, out = self.forward_bn(x, 'test')
            else:
                _, out = self.forward(x)
            loss = cross_entropy(y, out)
            losses += loss
            out = list(np.argmax(out, axis=-1).flatten())
            y = list(np.argmax(y, axis=1).flatten())
            labels += y
            pred += out

        return np.array(pred).astype('int'), np.array(labels).astype('int')

