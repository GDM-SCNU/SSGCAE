import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import torch.nn as nn
import torch


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, batch_norm=True):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        # self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        nn.init.xavier_uniform_(self.conv1.weight, gain=1.)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1.)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        # h = F.normalize(h)
        h = self.conv2(g, h)
        return F.log_softmax(h, dim=1)

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]


class CRD(torch.nn.Module):
    def __init__(self, in_feats, out_feats, p):
        super(CRD, self).__init__()
        self.conv = GraphConv(in_feats, out_feats)
        self.p = p

    def reset_parameters(self):
        # self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv.weight, gain=1.0)

    def forward(self, g, x):
        x = F.relu(self.conv(g, x))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]


class CLS(torch.nn.Module):
    def __init__(self, in_feat, out_feat):
        super(CLS, self).__init__()
        self.conv = GraphConv(in_feat, out_feat)

    def reset_parameters(self):
        # self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.conv.weight, gain=1.0)

    def forward(self, g, x):
        x = self.conv(g, x)
        # x = F.log_softmax(x, dim=1)
        x = F.softmax(x, dim=1)
        return x

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]


class ImprovGCN(torch.nn.Module):
    def __init__(self, in_feat, n_hid, n_out, drop_out):
        super(ImprovGCN, self).__init__()
        self.crd = CRD(in_feat, n_hid, drop_out)
        self.cls = CLS(n_hid, n_out)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, g, x):
        x = self.crd(g, x)
        x = self.cls(g, x)
        return x

    def get_weights(self):
        return [w for n, w in self.named_parameters() if 'bias' not in n]

# import dgl
#
# dataset = dgl.data.CoraGraphDataset()
# n_class = dataset.num_classes
# graph = dataset[0]
# features = graph.ndata['feat']
# labels = graph.ndata['label']
# in_feat = features.shape[1]
# n_hid = 128
# n_out = n_class
# drop_out = 0.5
# gcn = ImprovGCN(in_feat, n_hid, n_out, drop_out)
# logist = gcn(graph, features)
# print(logist)
