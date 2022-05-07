import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn import preprocessing
from torch.utils.data import Dataset
import os


def load_data(cancer, dataset):
    if dataset == 'dataset_1':
        root = './dataset/'
        root = root + 'dataset_1/fea/' + cancer + '/'
        print(root)

        dataset = pd.DataFrame([])
        frames = []
        minmax = preprocessing.MinMaxScaler()
        for idx, (_, _, n) in enumerate(os.walk(root)):
            for name in n:
                loc = root + name
                print(name)
                data = pd.read_csv(loc, header=0, index_col=0, sep=',')
                data = data.T
                print(data.shape)
                if cancer == 'ALL':
                    if name == 'miRNA.csv' or name == 'rna.csv':
                        data = np.log2(data + 1)
                data_minmax = minmax.fit_transform(data)
                data_minmax = DataFrame(data_minmax, index=data.index)
                frames.append(data_minmax)
            dataset = pd.concat(frames, axis=1)
        print(np.where(np.isnan(dataset)))
        print(dataset)
        x = torch.from_numpy(dataset.values).to(torch.float32)
        print(x.shape)
        return x

    elif dataset == 'dataset_2':
        root = './dataset/'
        root = root + 'dataset_2/fea/' + cancer + '/'
        print(root)

        dataset = pd.DataFrame([])
        frames = []
        minmax = preprocessing.MinMaxScaler()
        for idx, (_, _, n) in enumerate(os.walk(root)):
            for name in n:
                loc = root + name
                print(name)
                data = pd.read_csv(loc, header=0, index_col=0, sep=',')
                data = data.T
                print(data.shape)
                data_minmax = minmax.fit_transform(data)
                data_minmax = DataFrame(data_minmax, index=data.index)
                frames.append(data_minmax)
            dataset = pd.concat(frames, axis=1)
        print(np.where(np.isnan(dataset)))
        print(dataset)
        x = torch.from_numpy(dataset.values).to(torch.float32)
        print(x.shape)
        return x


class MyDataset(Dataset):
    """
    To import data
    """
    def __init__(self, cancer, dataset):
        self.x = load_data(cancer, dataset)

    def __len__(self):
        return self.x.shape[0]

    def __input__(self):
        print(self.x.shape[1])
        return self.x.shape[1]

    def __getitem__(self, idx):
        return self.x[idx]
