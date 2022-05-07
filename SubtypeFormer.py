import argparse
import bisect
import csv
import os
from itertools import combinations
import random
from os.path import isfile
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import mixture, metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from math import sqrt
from utils import MyDataset
import warnings

warnings.filterwarnings('ignore')
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)


class MultiHeadSelfAttention(nn.Module):
    """
    the multiHead module of Subtype-Former
    """
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads  # 2
        dk = self.dim_k // nh  # dim_k of each head 1
        dv = self.dim_v // nh  # dim_v of each head 1

        q = self.linear_q(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk) 5.reshape(16,5,2)
        k = self.linear_k(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk)
        v = self.linear_v(x.reshape(batch, dim_in)).reshape(batch, nh, dv)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, self.dim_v)  # batch, n, dim_v
        return att


class Transformer(nn.Module):
    """
    Subtype-Former network
    """
    def __init__(self, n_input):
        super(Transformer, self).__init__()
        self.fc1 = nn.Linear(n_input, 256)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 1000)
        self.attn1 = MultiHeadSelfAttention(dim_in=256, dim_k=64, dim_v=64)
        self.drop1 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(10, 10)
        self.norm1 = nn.LayerNorm(100)
        self.norm2 = nn.LayerNorm(64)
        self.norm3 = nn.LayerNorm(n_input)
        self.norm4 = nn.LayerNorm(256)
        self.norm5 = nn.LayerNorm(1000)
        self.fc8 = nn.Linear(64, 256)
        self.fc9 = nn.Linear(100, 500)
        self.fc10 = nn.Linear(500, 500)
        self.fc11 = nn.Linear(500, 1000)
        self.fc12 = nn.Linear(256, n_input)
        self.fc13 = nn.Linear(1000, 100)

    def encoder(self, x):
        x1 = self.norm4(self.fc1(x))  # 9844-256
        x1 = F.relu(x1)
        x2 = self.norm2(self.attn1(x1))  # 256-64
        x2 = F.relu(x2)
        x3 = self.fc8(x2)  # 64-256
        x3 = self.norm4(x3 + x1)
        x5 = self.fc5(x3)  # 256-64
        x5 = self.norm2(x5)
        return x5

    def decoder(self, x):
        x6 = self.norm4(self.fc8(x))  # 64-256
        x6 = F.relu(x6)
        x7 = self.norm2(self.attn1(x6) + x)  # 256-64
        x7 = F.relu(x7)
        x8 = self.norm4(self.fc8(x7))  # 64-256
        x8 = F.relu(x8)
        x10 = self.norm3(self.fc12(x8))  # 256-9844
        return x10

    def forward(self, x):
        z = self.encoder(x)
        x_bar = self.decoder(z)
        return x_bar, z


class TransModel(nn.Module):

    def __init__(self,
                 n_input,
                 n_clusters):
        super(TransModel, self).__init__()
        print('n_input:', n_input)
        self.transformer = Transformer(
            n_input=n_input)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_clusters))
        print('cluster_layer', self.cluster_layer)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        # torch.nn.init.kaiming_normal_(self.cluster_layer.data)

    def pretrain(self):
        pretrain_transformer(self.transformer)

    def forward(self, x):
        x_bar, z = self.transformer(x)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


class ConsensusCluster:
    """
    Subtype-Former cc
    """
    def __init__(self, cluster, L, K, H, resample_proportion=0.8):
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = L
        self.K_ = K
        self.H_ = H
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    def _internal_resample(self, data, proportion):
        ids = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0] * proportion), replace=False)
        return ids, data[ids, :]

    def fit(self, data):
        Mk = np.zeros((self.K_ - self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],) * 2)
        for k in range(self.L_, self.K_):
            i_ = k - self.L_
            for h in range(self.H_):
                ids, dt = self._internal_resample(data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k).fit_predict(dt)
                ids_sorted = np.argsort(Mh)
                sorted_ = Mh[ids_sorted]
                for i in range(k):
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = ids_sorted[ia:ib]
                    ids_ = np.array(list(combinations(is_, 2))).T
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                ids_2 = np.array(list(combinations(ids, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is + 1e-8
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1
            Is.fill(0)
        self.Mk = Mk
        self.Ak = np.zeros(self.K_ - self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = sum(h * (b - a)
                             for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
        self.deltaK = np.array([(Ab - Aa) / Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_ - 1))])
        self.bestK = np.argmax(self.deltaK) + \
                     self.L_ if self.deltaK.size > 0 else self.L_

    def predict(self):
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1 - self.Mk[self.bestK - self.L_])

    def predict_data(self, data):
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            data)


def gmm(n_clusters=28):
    model = mixture.GaussianMixture(n_components=n_clusters, covariance_type='diag')
    return model


def kmeans(n_clusters=28):
    model = KMeans(n_clusters=n_clusters, random_state=0)
    return model


def pretrain_transformer(model):
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        total_loss = 0.
        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("epoch {} loss={:.6f}".format(epoch, total_loss / (batch_idx + 1)))
        print(loss.item())


def train():
    model = TransModel(
        n_input=dataset.__input__(),
        n_clusters=args.n_clusters).to(device)
    model.pretrain()
    data = dataset.x
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.transformer(data)
    print(hidden.shape)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=args.n_clusters)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    # gmm = GaussianMixture(n_components=args.n_clusters)
    # y_pred = gmm.fit_predict(hidden.data.cpu().numpy())
    print(data[-1])
    print('y_pred:', y_pred)

    cancer = args.cancer
    if args.dataset == 'dataset_1':
        res = pd.read_csv('./dataset/dataset_1/header/' + cancer + '_header.csv', header=0, index_col=0, sep=',',
                          encoding='utf-8')
        hidden = pd.DataFrame(hidden.data.cpu().numpy())
        hidden.index = res.index
        hidden.to_csv('./results/hidden/hidden_1/' + cancer + '.hidden', header=True, index=True, sep=',')
        res['subtype'] = y_pred
        print(res)
        res.to_csv('./results/subtype_1/' + cancer + '.' + args.method, header=True, index=True, sep='\t')
    elif args.dataset == 'dataset_2':
        res = pd.read_csv('./dataset/dataset_2/header/' + cancer + '_header.csv', header=0, index_col=None, sep=',',
                          encoding='utf-8')
        hidden = pd.DataFrame(hidden.data.cpu().numpy())
        r = res.to_numpy()
        t = []
        for i in r:
            for j in i:
                t.append(j)
        hidden.index = t
        hidden.to_csv('./results/hidden/hidden_2/' + cancer + '.hidden', header=True, index=True, sep=',')

        res['subtype'] = y_pred
        print(res)
        res.to_csv('./results/subtype_2/' + cancer + '.' + args.method, header=True, index=False, sep='\t')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l', '--lr', type=float, default=0.002)
    parser.add_argument('-n', '--n_clusters', default=5, type=int)
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-e', '--epoch', default=45, type=int)
    parser.add_argument('-c', '--cancer', default="BRCA", help="cancer type: BRCA, GBM")
    parser.add_argument('-d', '--dataset', default="dataset_1", help="dataset_1, dataset_2")
    parser.add_argument('-m', '--method', type=str, default='SubtypeFormer',
                        help="subtype method: SubtypeFormer, cc, nmiari, tsne")
    args = parser.parse_args()

    if args.method == 'SubtypeFormer':
        # Subtype task
        cancer_dict = {'BRCA': 5, 'UCEC': 4, 'HNSC': 4, 'THCA': 2, 'LUAD': 3, 'KIRC': 4, 'PRAD': 3,
                       'LUSC': 4, 'SKCM': 4, 'STAD': 3, 'ALL': 27}
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")
        args.n_clusters = cancer_dict[args.cancer]
        dataset = MyDataset(args.cancer, args.dataset)
        print(args)
        train()

    elif args.method == 'cc':
        # Consensus clustering task
        K1_dict = {'BRCA': 4, 'UCEC': 4, 'HNSC': 4, 'THCA': 3, 'LUAD': 3, 'KIRC': 3,
                   'PRAD': 3, 'LUSC': 3, 'SKCM': 3, 'STAD': 3}
        K2_dict = {'BRCA': 8, 'UCEC': 8, 'HNSC': 8, 'THCA': 6, 'LUAD': 6, 'KIRC': 6,
                   'PRAD': 6, 'LUSC': 6, 'SKCM': 6, 'STAD': 6}
        cancer_type = args.cancer
        fea_tmp_file = './results/hidden/hidden_1/' + cancer_type + '.hidden'
        fs = []
        cc_file = './results/cc/k.cc'
        fp = open(cc_file, 'a')
        if isfile(fea_tmp_file):
            X = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=',')
            cc = ConsensusCluster(gmm, K1_dict[cancer_type], K2_dict[cancer_type], 10)
            cc.fit(X.values)
            X['cc'] = gmm(cc.bestK).fit_predict(X.values)
            X = X.loc[:, ['cc']]
            print(X)
            out_file = './results/cc/' + cancer_type + '.cc'
            X.to_csv(out_file, header=True, index=True, sep='\t')
            fp.write("%s, k=%d\n" % (cancer_type, cc.bestK))
        else:
            print('file does not exist!')
        fp.close()

    elif args.method == 'nmiari':
        # nmi and ari task
        cancer = args.cancer
        print(cancer)
        mylabel = './mylabel/' + cancer + '_label.csv'
        myres = './results/subtype_2/' + cancer + '.SubtypeFormer'
        if isfile(mylabel):
            A = pd.read_csv(mylabel, delimiter=',', usecols=[1]).to_numpy()
            tempA = []
            B = pd.read_csv(myres, delimiter='\t', usecols=[1]).to_numpy()
            tempB = []
            for t in B:
                tempB.append(t[0] + 1)
            if cancer == 'BRCA':
                for cancer_subtype in A:
                    if cancer_subtype == 'Basal':
                        tempA.append(1)
                    elif cancer_subtype == 'Her2':
                        tempA.append(2)
                    elif cancer_subtype == 'LumA':
                        tempA.append(3)
                    elif cancer_subtype == 'LumB':
                        tempA.append(4)
                    elif cancer_subtype == 'Normal':
                        tempA.append(5)
            elif cancer == 'UCEC':
                for cancer_subtype in A:
                    if cancer_subtype == 'CN_HIGH':
                        tempA.append(1)
                    elif cancer_subtype == 'CN_LOW':
                        tempA.append(2)
                    elif cancer_subtype == 'POLE':
                        tempA.append(3)
                    elif cancer_subtype == 'MSI':
                        tempA.append(4)
            elif cancer == 'HNSC':
                for cancer_subtype in A:
                    if cancer_subtype == 'Mesenchymal':
                        tempA.append(1)
                    elif cancer_subtype == 'Basal':
                        tempA.append(2)
                    elif cancer_subtype == 'Classical':
                        tempA.append(3)
                    elif cancer_subtype == 'Atypical':
                        tempA.append(4)
            elif cancer == 'THCA':
                for t in A:
                    tempA.append(t[0])
            elif cancer == 'LUAD':
                for t in A:
                    tempA.append(t[0])
            elif cancer == 'KIRC':
                for t in A:
                    tempA.append(t[0])
            elif cancer == 'PRAD':
                for cancer_subtype in A:
                    cancer_subtype = cancer_subtype[0][0]
                    if cancer_subtype == '1':
                        tempA.append(1)
                    elif cancer_subtype == '2':
                        tempA.append(2)
                    elif cancer_subtype == '3':
                        tempA.append(3)
                    elif cancer_subtype == '4':
                        tempA.append(4)
                    elif cancer_subtype == '5':
                        tempA.append(5)
                    elif cancer_subtype == '6':
                        tempA.append(6)
                    elif cancer_subtype == '7':
                        tempA.append(7)
                    elif cancer_subtype == '8':
                        tempA.append(8)
            elif cancer == 'LUSC':
                for cancer_subtype in A:
                    if cancer_subtype == 'basal':
                        tempA.append(1)
                    elif cancer_subtype == 'classical':
                        tempA.append(2)
                    elif cancer_subtype == 'secretory':
                        tempA.append(3)
                    elif cancer_subtype == 'primitive':
                        tempA.append(4)
            elif cancer == 'SKCM':
                for cancer_subtype in A:
                    if cancer_subtype == 'BRAF_Hotspot_Mutants':
                        tempA.append(1)
                    elif cancer_subtype == 'RAS_Hotspot_Mutants':
                        tempA.append(2)
                    elif cancer_subtype == 'Triple_WT':
                        tempA.append(3)
                    elif cancer_subtype == 'NF1_Any_Mutants':
                        tempA.append(4)
            elif cancer == 'STAD':
                for cancer_subtype in A:
                    if cancer_subtype == 'CIN':
                        tempA.append(1)
                    elif cancer_subtype == 'GS':
                        tempA.append(2)
                    elif cancer_subtype == 'MSI':
                        tempA.append(3)
                    elif cancer_subtype == 'EBV':
                        tempA.append(4)
                    elif cancer_subtype == 'POLE':
                        tempA.append(5)
                    elif cancer_subtype == 'HM-SNV':
                        tempA.append(6)
            print(tempA)
            print(tempB)
            if cancer == 'BRCA' or cancer == 'THCA':
                p = [[], [], [], [], []]
                for i in p:
                    for j in range(5):
                        i.append(0)
            elif cancer == 'UCEC' or cancer == 'KIRC' or cancer == 'LUSC' or cancer == 'SKCM' or cancer == 'HNSC':
                p = [[], [], [], []]
                for i in p:
                    for j in range(4):
                        i.append(0)
            elif cancer == 'LUAD' or cancer == 'STAD':
                p = [[], [], [], [], [], []]
                for i in p:
                    for j in range(6):
                        i.append(0)
            elif cancer == 'PRAD':
                p = [[], [], [], [], [], [], [], []]
                for i in p:
                    for j in range(8):
                        i.append(0)
            print(p)
            for i in range(len(tempA)):
                p[tempA[i] - 1][tempB[i] - 1] += 1
            print(p)

            res1 = metrics.precision_score(tempA, tempB, average='micro')
            res2 = metrics.f1_score(tempA, tempB, average='micro')
            res3 = metrics.normalized_mutual_info_score(tempA, tempB)
            res4 = metrics.adjusted_rand_score(tempA, tempB)

            print('precision:', res1)
            print('f1score:', res2)
            print('NMI:', res3)
            print('ARI:', res4)
            save_dir = './results/nmi_ari/' + cancer + '.SubtypeFormer'
            with open(save_dir, 'w') as f:
                f.truncate()
                writer = csv.writer(f)
                data1 = ('precision', res1)
                data2 = ('f1score', res2)
                data3 = ('NMI', res3)
                data4 = ('ARI', res4)
                writer.writerow(data1)
                writer.writerow(data2)
                writer.writerow(data3)
                writer.writerow(data4)
        else:
            print("label does not exist!")

    elif args.method == 'tsne':
        # tsne task
        cancer = args.cancer
        fea_tmp_file = './results/hidden/hidden_1/' + cancer + '.hidden'
        out_file = './results/tsne/' + cancer + '.tsne'
        if isfile(fea_tmp_file):
            df = pd.read_csv(fea_tmp_file, header=0, index_col=0, sep=',')
            print(df)
            mat = df.values.astype(float)
            labels = TSNE(n_components=2).fit_transform(mat)
            print(labels.shape)
            df['x'] = labels[:, 0]
            df['y'] = labels[:, 1]
            df = df.loc[:, ['x', 'y']]
            print(df)
            subtype = pd.read_csv('./results/subtype_1/' + cancer + '.SubtypeFormer', header=0, index_col=0, sep='\t')
            print(subtype)
            df = df.join(subtype)
            print('df:', df)
            df.to_csv(out_file, header=True, index=True, sep='\t')
        else:
            print("hidden does not exist!")


