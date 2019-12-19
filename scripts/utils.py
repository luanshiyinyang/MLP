"""
Author: Zhou Chen
Date: 2019/12/4
Desc: 工具模块
"""
import numpy as np
import matplotlib.pyplot as plt


def onehot(labels, classes=None):
    """
    onehot编码实现
    """
    if labels.ndim == 1:
        # 保证都是[batch, 1]维度数目
        labels = labels.reshape(-1, 1)

    num_data = labels.shape[0]
    index_offset = np.arange(num_data) * classes
    labels_onehot = np.zeros(shape=(num_data, classes))
    labels_onehot.flat[index_offset + labels.ravel()] = 1
    return labels_onehot


def confusion_matrix(labels, pred, classes=10):
    """
    从真实标签和预测标签生成混淆矩阵
    """
    conf_mat = np.zeros([classes, classes])
    for i in range(len(labels)):
        true_i = np.array(labels[i])
        pre_i = np.array(pred[i])
        conf_mat[true_i, pre_i] += 1.0
    return conf_mat


def cm_plot(cm, title):
    """
    绘制混淆矩阵图像
    """
    cm = cm.astype('int')
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel("True label")
    plt.xlabel("Pred label")
    plt.title(title, y=1.16)
    plt.tight_layout()
    plt.savefig("results/" + title+".png")
    plt.show()


def plot_history(his_a, his_b, legend_a, legend_b, title1, title2):
    plt.plot(np.arange(len(his_a['train_loss'])), his_a['train_loss'], label=legend_a, c='b')
    plt.plot(np.arange(len(his_b['train_loss'])), his_b['train_loss'], label=legend_b, c='y')
    plt.title(title1)
    plt.legend(loc=0)
    plt.savefig("results/" + title1+".png")
    plt.show()

    plt.plot(np.arange(len(his_a['valid_loss'])), his_a['valid_loss'], label=legend_a, c='b')
    plt.plot(np.arange(len(his_b['valid_loss'])), his_b['valid_loss'], label=legend_b, c='y')
    plt.title(title2)
    plt.legend(loc=0)
    plt.savefig("results/" + title2 + ".png")
    plt.show()


if __name__ == '__main__':
    label = np.array([1, 3, 4, 5, 6])
    print(onehot(label, 10))