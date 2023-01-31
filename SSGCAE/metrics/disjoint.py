# -*- coding: utf-8 -*-
# @Author        : 郑裕龙
# @Time          : 2021/10/26 15:58

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score as F1
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import networkx as nx
import community
import warnings


def nmi(label: np.array, pred: np.array):
    """
    计算非重叠互信息
    :param label: 真实社区标签
    :param pred: 预测社区标签
    :return:
    """
    return normalized_mutual_info_score(label, pred)


def ac(label: np.array, pred: np.array):
    """
    非重叠聚类准确率AC(社区发现)
    :param label: 真实社区标签
    :param pred: 预测社区标签
    :return:
    """
    labels = label.astype(np.int64)
    assert pred.size == labels.size
    D = max(pred.max(), labels.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(pred.size):
        w[pred[i], labels[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / pred.size


def ari(label: np.array, pred: np.array):
    """
    非重叠兰德指数
    :param label: 真实社区标签
    :param pred: 预测社区标签
    :return:
    """
    return adjusted_rand_score(label, pred)


def modularity(adj: np.array, pred: np.array):
    """
    非重叠模块度
    :param adj: 邻接矩阵
    :param pred: 预测社区标签
    :return:
    """
    graph = nx.from_numpy_matrix(adj.numpy())
    part = pred.tolist()
    index = range(0, len(part))
    dic = zip(index, part)
    part = dict(dic)
    modur = community.modularity(part, graph)
    return modur


def f1_score():
    return NotImplementedError
