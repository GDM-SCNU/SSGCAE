import torch
import torch.nn.functional as F
from scipy import sparse
from modules.improvgcn import ImprovGCN
import dgl
import networkx as nx
from dataset.realnetwork import OverlappingRealNetwork
from dataset.synthetic import OverlappingSyntheticNetwork

from metrics.overlapping import overlapping_nmi, overlapping_f1_score, find_best_thresh
import matplotlib.pyplot as plt
import numpy as np


class SMODGCN:
    def __init__(self, graph, num_class, lr=5e-4, normalize_feature=False, lamda=0.5, alpha=1e-8, beta=1e-6):
        """

        :param graph: dgl.DGLGraph
        :param num_class: 社区数
        :param lr: 学习率
        :param normalize_feature: 是否正则化
        :param lamda: 损失函数权衡系数 λ
        """

        self.graph = graph

        self.adj = graph.adjacency_matrix().to_dense()  # 稠密矩阵表示 FloatTensor
        self.sp_adj = sparse.csr_matrix(self.adj.numpy())  # 稀疏矩阵
        self.num_class = num_class  # 社区数
        self.lr = lr  # 学习率
        self.normalize_feature = normalize_feature  # 正则化
        self.lamda = lamda  # 损失函数权衡系数 λ
        self.alpha = alpha
        self.beta = beta
        self.B = self.compute_B_matrix()
        # self.M = self.compute_M_matrix()
        # self.L = self.compute_L_matrix()
        # self.D = self.compute_D_matrix()

        # self.add_semi_supervised_constraint()  # 添加半监督约束信息

    def compute_D_matrix(self):
        n = self.graph.num_nodes()
        D = torch.zeros((n, n))
        for i, d in enumerate(self.graph.in_degrees()):
            D[i][i] = d
        return D

    def compute_M_matrix(self):
        """

        :param H: 重叠社区标签
        :return:
        """
        train_mask = self.graph.ndata['train_mask']
        n = self.graph.num_nodes()
        H = self.graph.ndata['olp_label']
        M = np.zeros((n, n))

        C = []
        for j in range(H.shape[1]):
            col = H[:, j]
            tmp = []
            for i, x in enumerate(col):
                if x == 1 and train_mask[i]:
                    tmp.append(i)
            C.append(tmp)

        for arr in C:
            c_len = len(arr)
            if c_len > 0:
                for i in range(0, c_len):
                    for j in range(i, c_len):
                        M[arr[i]][arr[j]] = 1
                        M[arr[j]][arr[i]] = 1

        return torch.tensor(M, dtype=torch.float32)

    def compute_L_matrix(self):
        n = self.graph.num_nodes()
        D = torch.zeros((n, n))
        for i, d in enumerate(self.graph.in_degrees()):
            D[i][i] = d
        L = D - self.M
        return L

    def add_semi_supervised_constraint(self):
        """
        给图添加半监督约束信息，即在同一个社区的两个节点，若没有边则为这两个节点添加一条无向边
        :return:
        """
        Z = self.graph.ndata['olp_label']
        train_mask = self.graph.ndata['train_mask']
        cmt_list = [[] for i in range(self.num_class)]
        for idx, row in enumerate(Z):

            if not train_mask[idx]:
                continue
            for j, cmt in enumerate(row):
                if cmt == 1:
                    cmt_list[j].append(idx)

        for ids in cmt_list:
            for i in range(0, len(ids) - 1):
                for j in range(i + 1, len(ids)):
                    self.graph.add_edges(i, j)
                    self.graph.add_edges(j, i)

    def compute_B_matrix(self):
        nx_graph = dgl.to_networkx(self.graph)
        degree = nx.degree(nx_graph)
        deg = torch.FloatTensor([d for id, d in degree]).reshape(-1, 1)
        sum_deg = deg.sum()
        B = self.adj - (deg.matmul(deg.t()) / sum_deg)
        return B

    def l2_reg_loss(self, model, scale=1e-4):
        """Get L2 loss for model weights."""
        loss = 0.0
        for w in model.get_weights():
            loss += w.pow(2.).sum()
        return loss * scale

    def l1_reg_loss(self, model, scale=1e-4):
        loss = 0.
        for w in model.get_weights():
            loss += w.abs().sum()
        return loss * scale

    def update_laplacion_matrixToU(self, U, M, N, K, xi=1.5):
        """

        :param U: 社区表示矩阵
        :param M:
        :param N: 节点数
        :param K: 社区数
        :param xi: 超参数
        :return:
        """
        # 转化为概率形式

        pred = torch.from_numpy(U).softmax(dim=0).max(dim=0)
        max_values = pred.values.numpy()
        max_indices = pred.indices.numpy()
        # 拓展维度
        max_values = np.expand_dims(max_values, 0)
        # 通过最大概率，计算每个节点之间的概率值（相乘），得到概率矩阵P
        P = np.dot(max_values.T, max_values)
        # 初始化隶属矩阵,为全0,大小为N * K
        Affiliation = np.zeros(shape=(N, K))
        x_indices = np.arange(N)
        # N * k - k * N
        Affiliation[x_indices, max_indices] = 1
        Affiliation = np.dot(Affiliation, Affiliation.T)
        cal_M = Affiliation * P
        M = M + xi * cal_M
        row, col = np.diag_indices_from(M)
        M[row, col] = 1
        D = np.diag(M.sum(axis=1))
        return M, D

    def lap_loss(self, U, M, D):
        """

        :param U:
        :param M:
        :param D:
        :return:
        """
        # U = U.t()
        M = torch.FloatTensor(M)
        D = torch.FloatTensor(D)

        return 2 * torch.trace(U.t().matmul(D).matmul(U)) - 2 * torch.trace(U.t().matmul(M).matmul(U))

    def train(self, max_epoch=300, unsupervised=False, loss_fun='CE', modul=True):
        """

        :param max_epoch: 最大迭代次数
        :param unsupervised: 默认为False，表示使用半监督的方式。若置位True，表示使用无监督方式训练
        :param loss_fun: GR 或者 CE, 分别表示用图正则、交叉熵作为半监督损失
        :param show_process 显示训练进度
        :return:
        """
        # torch.manual_seed(12)
        features = self.graph.ndata['feat']

        if self.normalize_feature:
            features = F.normalize(features)

        # self.adj = F.normalize(self.adj)

        in_feat = features.shape[1]
        n_hidden = 64
        num_class = self.num_class

        # gcn = GCN(in_feat, n_hidden, num_class, batch_norm=False)

        gcn = ImprovGCN(in_feat, n_hidden, num_class, 0.)
        optimizer = torch.optim.AdamW(gcn.parameters(), lr=self.lr)
        labels = self.graph.ndata['olp_label']
        train_mask = self.graph.ndata['train_mask']
        test_mask = self.graph.ndata['test_mask']

        nmi_results = []
        f1_score_results = []
        # L = self.L
        num_node = self.graph.num_nodes()
        # M = np.eye(num_node)
        # D = self.D.numpy()

        for e in range(1, max_epoch + 1):
            gcn.train()
            u = gcn(self.graph, features).relu()
            # numz = self.non_zero(u)
            # print("非零元素个数=", numz)

            u[u < 0.02] = 0
            # print(u)

            if unsupervised:
                loss = 0.
            else:
                loss = self.lamda * F.binary_cross_entropy(u[train_mask], labels[train_mask].float())

            A_1 = torch.sigmoid(u.matmul(u.t()))
            A_0 = 1 - A_1
            A = self.adj * torch.log(A_1) + (1 - self.adj) * torch.log(A_0)
            loss += self.alpha * (-A.sum().sum())

            if modul == True:
                loss -= self.beta * torch.trace(u.t().matmul(self.B).matmul(u))
            loss += self.l2_reg_loss(gcn, 1e-5)
            loss += self.l1_reg_loss(gcn, 1e-6)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % 1 == 0:
                gcn.eval()
                # thresh = find_best_thresh(labels[test_mask].numpy(), u[test_mask].detach().numpy())
                thresh = 3.0 / num_class
                nmi = overlapping_nmi(labels[test_mask].numpy(), u[test_mask].detach().numpy(), thresh)

                f1_score = overlapping_f1_score(labels[test_mask].numpy(), u[test_mask].detach().numpy(), thresh)

                print('epoch={}, loss={:.3f}, overlapping nmi: {:.3f}, f1_score={:.3f}'.format(e, loss, nmi,
                                                                                               f1_score))
                nmi_results.append(nmi)
                f1_score_results.append(f1_score)

            if e == max_epoch:
                print('Max Overlapping NMI:{:.3f}'.format(np.max(nmi_results)))
                print('Max Overlapping F1_Score:{:.3f}'.format(np.max(f1_score_results)))


def example():
    ds = OverlappingRealNetwork(path="dataset/real_network/overlapping/fb_1912", name="fb1912",
                                mask_rate=[0.02, 0.0, 0.98])
    graph = ds[0]
    print(ds.num_classes)
    model = SMODGCN(graph, num_class=ds.num_classes, lr=0.01, lamda=0.5, alpha=1e-7, beta=1e-6)
    model.train(max_epoch=100)


def example2():
    ds = OverlappingSyntheticNetwork(path="dataset/synthetic/overlapping/SG2/", n=5000, mu=0.3, on=500, om=3,
                                     mask_rate=[0.02, 0., 0.98])
    graph = ds[0]
    model = SMODGCN(graph, num_class=ds.num_classes, lr=5e-3, lamda=0.5, alpha=1e-7, beta=1e-6)
    model.train(max_epoch=500, unsupervised=False)


if __name__ == '__main__':
    example2()
