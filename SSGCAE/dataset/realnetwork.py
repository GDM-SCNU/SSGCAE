# -*- coding: utf-8 -*-
# @Author        : 郑裕龙
# @Time          : 2021/10/28 15:00

import dgl
import numpy as np
from dgl.data import DGLDataset
import pandas as pd
import random
import torch


class OverlappingRealNetwork(DGLDataset):
    def __init__(self, path="", name="", mask_shuffle=True, self_loop=True, mask_rate=[0.2, 0.6, 0.2], p_mis=0.):
        if not path.endswith("/"):
            path += "/"
        self.path = path
        self.mask_shuffle = mask_shuffle
        self.self_loop = self_loop
        self.mask_rate = mask_rate
        self.p_mis = p_mis
        # self.num_community = self.get_num_community(self.path + 'community.dat')
        assert sum(mask_rate) == 1
        super(OverlappingRealNetwork, self).__init__(name=name)

    def process(self):
        path = self.path
        features = np.loadtxt(fname=path + "features.dat", dtype=np.float32, delimiter='\t')
        labels = np.loadtxt(path + 'community.dat')
        edges = np.loadtxt(fname=path + "edges.dat", dtype=np.int32)
        self.num_node = labels.shape[0]
        num_nodes = self.num_node
        self.num_community = labels.shape[1]

        mask_index = np.arange(start=0, stop=num_nodes)
        if self.mask_shuffle:
            np.random.shuffle(mask_index)

        train_mask = torch.BoolTensor(np.zeros(num_nodes))
        val_mask = torch.BoolTensor(np.zeros(num_nodes))
        test_mask = torch.BoolTensor(np.zeros(num_nodes))

        for i, v in enumerate(mask_index):
            if i < num_nodes * self.mask_rate[0]:
                train_mask[v] = True
            elif i < num_nodes * self.mask_rate[0] + self.mask_rate[1]:
                val_mask[v] = True
            else:
                test_mask[v] = True

        # 处理边文件
        src = edges[:, 0]
        dest = edges[:, 1]

        #  交换节点属性
        swap_count = int(num_nodes * self.p_mis / 2)
        for i in range(swap_count):
            a = random.randint(0, num_nodes - 1)
            b = random.randint(0, num_nodes - 1)
            # print(a, ' ', b)
            if a == b:
                continue
            features[[a, b], :] = features[[b, a], :]

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        self.graph = dgl.graph((src, dest), num_nodes=num_nodes)
        if self.self_loop:
            self.graph = dgl.add_self_loop(self.graph)

        self.graph.ndata['feat'] = features
        self.graph.ndata['olp_label'] = labels
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        # self.graph.ndata['num_classes'] = self.num_community

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.graph

    @property
    def num_classes(self):
        return self.num_community


def example():
    fb107 = OverlappingRealNetwork(path="./real_network/overlapping/fb_348", name="fb348")
    graph = fb107[0]
    print(graph.ndata['olp_label'])


if __name__ == '__main__':
    example()
