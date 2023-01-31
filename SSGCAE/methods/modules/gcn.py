# -*- coding: utf-8 -*-
# @Author        : 郑裕龙
# @Time          : 2021/11/7 18:48

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, batch_norm=True):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.BN = nn.BatchNorm1d(h_feats,  eps=1e-03, momentum=0.05, affine=False, track_running_stats=True)
        self.batch_norm = batch_norm

        # nn.init.xavier_uniform_(self.conv1.weight, gain=2.0)
        # nn.init.xavier_uniform_(self.conv2.weight, gain=2.0)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        if self.batch_norm:
            h = self.BN(h)
        h = F.relu(h)

        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(g, h)
        return F.log_softmax(h, dim=1)

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]
