# -*- coding: utf-8 -*-
# @Author        : 郑裕龙
# @Time          : 2021/10/26 15:59

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import f1_score as F1
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import networkx as nx
import community
import warnings


def relu(x: np.array):
    x = (np.abs(x) + x) / 2.0
    return x


def ONMI(X, Y):
    """
        Compute NMI between two overlapping community covers.

        Parameters
        ----------
        X : array-like, shape [N, m]
            Matrix with samples stored as columns.
        Y : array-like, shape [N, n]
            Matrix with samples stored as columns.

        Returns
        -------
        nmi : float
            Float in [0, 1] quantifying the agreement between the two partitions.
            Higher is better.

        References
        ----------
        McDaid, Aaron F., Derek Greene, and Neil Hurley.
        "Normalized mutual information to evaluate overlapping
        community finding algorithms."
        arXiv preprint arXiv:1110.2515 (2011).

    """
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("X should be a binary matrix")
    if not ((Y == 0) | (Y == 1)).all():
        raise ValueError("Y should be a binary matrix")

    if X.shape[1] > X.shape[0] or Y.shape[1] > Y.shape[0]:
        warnings.warn("It seems that you forgot to transpose the F matrix")
    X = X.T
    Y = Y.T

    def cmp(x, y):
        """Compare two binary vectors."""
        a = (1 - x).dot(1 - y)
        d = x.dot(y)
        c = (1 - y).dot(x)
        b = (1 - x).dot(y)
        return a, b, c, d

    def h(w, n):
        """Compute contribution of a single term to the entropy."""
        if w == 0:
            return 0
        else:
            return -w * np.log2(w / n)

    def H(x, y):
        """Compute conditional entropy between two vectors."""
        a, b, c, d = cmp(x, y)
        n = len(x)
        if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
            return h(a, n) + h(b, n) + h(c, n) + h(d, n) - h(b + d, n) - h(a + c, n)
        else:
            return h(c + d, n) + h(a + b, n)

    def H_uncond(X):
        """Compute unconditional entropy of a single binary matrix."""
        return sum(h(x.sum(), len(x)) + h(len(x) - x.sum(), len(x)) for x in X)

    def H_cond(X, Y):
        """Compute conditional entropy between two binary matrices."""
        m, n = X.shape[0], Y.shape[0]
        scores = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                scores[i, j] = H(X[i], Y[j])
        return scores.min(axis=1).sum()

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimensions of X and Y don't match. (Samples must be stored as COLUMNS)")
    H_X = H_uncond(X)
    H_Y = H_uncond(Y)
    I_XY = 0.5 * (H_X + H_Y - H_cond(X, Y) - H_cond(Y, X))
    return I_XY / max(H_X, H_Y)


def find_best_thresh(label: np.array, z: np.array):
    """
    寻找 overlapping_nmi最大时的阈值
    :param label: 重叠社区标签， np.array  [N * K]
    :param z: 社区表示矩阵， np.array  [N * K]
    :return:

    """
    thresh = 0.
    best_thresh = 0.
    max_onmi = 0.
    while thresh <= np.max(z):
        onmi_val = overlapping_nmi(label, z, thresh)
        if onmi_val >= max_onmi:
            best_thresh = thresh
            max_onmi = onmi_val
        thresh += 0.01
    return best_thresh


def overlapping_nmi(label: np.array, z: np.array, thresh=0.5):
    """
    :param label: 重叠社区标签， np.array  [N * K]
    :param z: 社区表示矩阵， np.array  [N * K]
    :param thresh: 阈值, 默认值为0.5 ，可以调用find_best_thresh() 获取最好的阈值
    :return:

    """
    z = relu(z)
    pred = z > thresh
    return ONMI(label, pred)


def overlapping_f1_score(label: np.array, z: np.array, thresh=0.5):
    """
    计算 F1-score
    :param label: 重叠社区标签， np.array  [N * K]
    :param z: 社区表示矩阵， np.array  [N * K]
    :param thresh 阈值, 默认值为0.5 ，可以调用find_best_thresh() 获取ONMI最好的阈值
    :return:
    """
    pred = z > thresh
    return F1(label, pred, average="micro")
