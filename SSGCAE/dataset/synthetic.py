from dgl.data import DGLDataset
import os
import pandas as pd
import numpy as np
import warnings
import torch
import dgl
import random


warnings.filterwarnings('ignore')


class SyntheticNetwork(DGLDataset):
    def __init__(self, dataset_name='', mask_shuffle=False, self_loop=True, mask_rate=None):
        self.dataset_name = dataset_name
        self.path = os.path.abspath(os.pardir) + "/dataset/synthetic_networks/" + self.dataset_name + "/"
        self.mask_shuffle = mask_shuffle
        self.self_loop = self_loop
        self.mask_rate = mask_rate
        assert sum(mask_rate) == 1
        super(SyntheticNetwork, self).__init__(name=dataset_name)

    def process(self):
        edges = pd.read_csv(self.path + 'network.dat', sep='\t', header=None).to_numpy()
        features = np.loadtxt(self.path + "features.txt", dtype=np.float, delimiter='\t')
        labels = np.loadtxt(self.path + 'community.dat', dtype=np.long)[:, 1] - 1

        num_nodes = len(labels)
        self.num_class = np.max(labels)
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

        src = edges[:, 0] - 1
        dist = edges[:, 1] - 1
        esrc = np.hstack((src, dist))
        edest = np.hstack((dist, src))

        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        self.graph = dgl.graph((esrc, edest), num_nodes=num_nodes)
        if self.self_loop:
            self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['feat'] = features
        self.graph.ndata['label'] = labels
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    @property
    def num_classes(self):
        return self.num_class + 1

    def __len__(self):
        return 1


class OverlappingSyntheticNetwork(DGLDataset):
    def __init__(self, path="", n=1000, mu=0.1, on=0, om=0, mask_shuffle=False, self_loop=True, mask_rate=[1.0, 0, 0],
                 p_mis=0.):
        """
        :param path 文件路径  例如: './dataset/synthetic/'
        :param n: 节点数
        :param mu: 混淆系数
        :param on: 重叠节点数
        :param om: 重叠节点所属社区数
        :param mask_shuffle: 打乱标签
        :param self_loop: 是否自环边
        :param mask_rate: 标签率[训练，验证，测试]
        :param p_mis: 属性交换比例， 默认不交换

        文件目录下游三个子文件
        |--文件名
            |--network.dat  边
            |--features.dat 特征
            |--community.dat 社区标签
        """
        if not path.endswith("/"):
            path += "/"
        filename = "n" + str(n) + "-"
        filename += "mu" + str(mu) + "-"
        filename += "on" + str(on) + "-"
        filename += "om" + str(om) + "/"
        path += filename
        self.path = path
        self.mask_shuffle = mask_shuffle
        self.self_loop = self_loop
        self.mask_rate = mask_rate
        self.num_nodes = n
        self.num_community = self.get_num_community(self.path + 'community.dat')
        self.p_mis = p_mis
        assert sum(mask_rate) == 1

        super(OverlappingSyntheticNetwork, self).__init__(name="")
        pass

    def get_num_community(self, path=""):
        cmt = list(pd.read_csv(path, delimiter='\t', header=None).iloc[:, 1])
        max_val = -1
        for row in cmt:
            row = [int(x) for x in row.strip().split()]
            for x in row:
                max_val = max(x, max_val)
        return max_val

    def process(self):
        path = self.path
        num_community = self.num_community
        num_nodes = self.num_nodes
        Z = np.zeros((num_nodes, num_community))
        cmt = list(pd.read_csv(path + 'community.dat', delimiter='\t', header=None).iloc[:, 1])
        for idx, row in enumerate(cmt):
            row = [int(x) for x in row.strip().split()]
            for c in row:
                Z[idx][c - 1] = 1

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
        edges = pd.read_csv(path + 'network.dat', sep='\t', header=None).to_numpy()
        src = edges[:, 0] - 1
        dest = edges[:, 1] - 1

        # 处理特征文件
        features = np.loadtxt(self.path + "olp_features.dat", dtype=np.float32, delimiter='\t')
        n = num_nodes
        swap_count = int(n * self.p_mis / 2)
        for i in range(swap_count):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            # print(a, ' ', b)
            if a == b:
                continue
            features[[a, b], :] = features[[b, a], :]
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(Z)

        self.graph = dgl.graph((src, dest), num_nodes=num_nodes)
        if self.self_loop:
            self.graph = dgl.add_self_loop(self.graph)
        self.graph.ndata['feat'] = features
        self.graph.ndata['olp_label'] = labels
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        pass

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.graph

    @property
    def num_classes(self):
        return self.num_community


def example():
    dataset = OverlappingSyntheticNetwork(path="./synthetic/overlapping/SG1", n=1000, mu=0.1, on=10, om=2)
    graph = dataset[0]
    print(graph)
    print(graph.ndata['feat'].shape)
    print(graph.ndata['olp_label'].shape)
    print(graph.ndata['train_mask'].sum())
    print(graph.ndata['val_mask'].sum())
    print(graph.ndata['test_mask'].sum())

    print(dataset.num_classes)

    print(graph.ndata['olp_label'])
    k = dataset.num_classes
    print('社区数={}'.format(k))

