"""
Author: Zhou Chen
Date: 2019/12/4
Desc: 主模块，程序入口
"""
from model import Model
from data import DataLoader
from utils import confusion_matrix, cm_plot, plot_history
import numpy as np


def contrast_scale(epochs, lr):
    train_loader_no_scale = DataLoader(64, data_type='train', scale=False)
    valid_loader_no_scale = DataLoader(64, data_type='valid', scale=False)
    test_loader_no_scale = DataLoader(64, data_type='test', scale=False)
    model = Model(4, [28*28, 512, 512, 10], initializer='xavier')
    his_no_scale = model.train(train_loader_no_scale, valid_loader_no_scale, epochs, learning_rate=lr)
    pred, label = model.predict(test_loader_no_scale)
    acc = np.sum(pred == label) / len(pred)
    print('acc', acc)
    cm = confusion_matrix(label.reshape(-1), pred.reshape(-1), 10)
    cm_plot(cm, 'no_scale_cm, acc {:.3f}'.format(acc))

    train_loader_scale = DataLoader(64, data_type='train', scale=True)
    valid_loader_scale = DataLoader(64, data_type='valid', scale=True)
    test_loader_scale = DataLoader(64, data_type='test', scale=True)
    model2 = Model(4, [28 * 28, 512, 512, 10], initializer='xavier')
    his_scale = model2.train(train_loader_scale, valid_loader_scale, epochs, learning_rate=lr)
    pred, label = model2.predict(test_loader_scale)
    acc = np.sum(pred == label) / len(pred)
    print('acc', acc)
    cm = confusion_matrix(label.reshape(-1), pred.reshape(-1), 10)
    cm_plot(cm, 'scale_cm, acc {:.3f}'.format(acc))

    plot_history(his_no_scale, his_scale, 'no scale', 'scale', 'sacle and no scale training loss', 'scale an no scale validation loss')


def contrast_dropout(epochs, lr):
    train_loader_no_dropout = DataLoader(64, data_type='train', scale=True)
    valid_loader_no_dropout = DataLoader(64, data_type='valid', scale=True)
    test_loader_no_dropout = DataLoader(64, data_type='test', scale=True)
    model = Model(4, [28 * 28, 512, 512, 10], initializer='xavier')
    his_no_dropout = model.train(train_loader_no_dropout, valid_loader_no_dropout, epochs, learning_rate=lr)
    pred, label = model.predict(test_loader_no_dropout)
    acc = np.sum(pred == label) / len(pred)
    print('acc', acc)
    cm = confusion_matrix(label.reshape(-1), pred.reshape(-1), 10)
    cm_plot(cm, 'no_dropout_cm, acc {:.3f}'.format(acc))

    train_loader_dropout = DataLoader(64, data_type='train', scale=True)
    valid_loader_dropout = DataLoader(64, data_type='valid', scale=True)
    test_loader_dropout = DataLoader(64, data_type='test', scale=True)
    model2 = Model(4, [28 * 28, 512, 512, 10], initializer='xavier')
    his_dropout = model2.train(train_loader_dropout, valid_loader_dropout, epochs, learning_rate=lr, dropout_prob=0.3)
    pred, label = model2.predict(test_loader_dropout)
    acc = np.sum(pred == label) / len(pred)
    print('acc', acc)
    cm = confusion_matrix(label.reshape(-1), pred.reshape(-1), 10)
    cm_plot(cm, 'dropout_cm, acc {:.3f}'.format(acc))

    plot_history(his_no_dropout, his_dropout, 'no dropout', 'dropout', 'dropout and no dropout training loss', 'dropout and no dropout validation loss')


def contrast_bn(epochs, lr):
    train_loader_no_bn = DataLoader(64, data_type='train', scale=True)
    valid_loader_no_bn = DataLoader(64, data_type='valid', scale=True)
    test_loader_no_bn = DataLoader(64, data_type='test', scale=True)
    model = Model(4, [28 * 28, 512, 512, 10], initializer='xavier', optimizer='sgd')
    his_no_bn = model.train(train_loader_no_bn, valid_loader_no_bn, epochs, learning_rate=lr)
    pred, label = model.predict(test_loader_no_bn)
    acc = np.sum(pred == label) / len(pred)
    print('acc', acc)
    cm = confusion_matrix(label.reshape(-1), pred.reshape(-1), 10)
    cm_plot(cm, 'no_bn_cm, acc {:.3f}'.format(acc))

    train_loader_bn = DataLoader(64, data_type='train', scale=True)
    valid_loader_bn = DataLoader(64, data_type='valid', scale=True)
    test_loader_bn = DataLoader(64, data_type='test', scale=True)
    model2 = Model(4, [28 * 28, 512, 512, 10], initializer='xavier', optimizer='sgd')
    his_bn = model2.train_bn(train_loader_bn, valid_loader_bn, epochs, learning_rate=lr)
    pred, label = model2.predict(test_loader_bn, bn=True)
    acc = np.sum(pred == label) / len(pred)
    print('acc', acc)
    cm = confusion_matrix(label.reshape(-1), pred.reshape(-1), 10)
    cm_plot(cm, 'bn_cm, acc {:.3f}'.format(acc))

    plot_history(his_no_bn, his_bn, 'no bn', 'bn', 'bn and no bn training loss', 'bn and no bn validation loss')


def main():
    # contrast_scale(50, 0.001)
    # contrast_dropout(50, 0.001)
    contrast_bn(5, 0.001)


if __name__ == '__main__':
    main()
