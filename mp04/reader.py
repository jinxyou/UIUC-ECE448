# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12@illinois.edu) for the fall 2021 semester
# Modified by Qinren Zhou (qinrenz2@illinois.edu) for the spring 2023 semester

"""
This file is responsible for providing functions for reading the files
"""
import numpy as np
import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader


def Load_dataset(filename):
    def unpickle(file):
        with open(file, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        return dict

    A = unpickle(filename)
    X = A[b"data"]
    Y = A[b"labels"].astype(np.int64)
    test_size = int(0.25 * len(X))  # set aside 25% for testing
    X_test = X[:test_size]
    Y_test = Y[:test_size]
    X = X[test_size:]
    Y = Y[test_size:]
    return X, Y, X_test, Y_test


def Preprocess(train_set, test_set):
    train_set = torch.tensor(train_set, dtype=torch.float32)
    test_set = torch.tensor(test_set, dtype=torch.float32)
    mu = train_set.mean(dim=0, keepdim=True)
    std = train_set.std(dim=0, keepdim=True)
    train_set = (train_set - mu) / std
    test_set = (test_set - mu) / std
    return train_set, test_set


def Get_DataLoaders(train_set, train_labels, test_set, test_labels, batch_size):
    train_dataset = MP_Dataset(train_set, train_labels)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_dataset = MP_Dataset(test_set, test_labels)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


class MP_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X, y):
        """
        Args:
            X [np.array]: features vector
            y [np.array]: labels vector
        """
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.data[idx, :]
        label = self.labels[idx]
        return features, label


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_accuracies(pred_labels, dev_labels):
    assert (
        pred_labels.dtype == int or pred_labels.dtype == np.int64
    ), "Your predicted labels have type {}, but they should have type np.int (consider using .astype(int) on your output)".format(
        pred_labels.dtype
    )
    if len(pred_labels) != len(dev_labels):
        print(
            "Lengths of predicted labels don't match length of actual labels",
            len(pred_labels),
            len(dev_labels),
        )
        return 0.0, None
    accuracy = np.mean(pred_labels == dev_labels)
    conf_m = np.zeros((len(np.unique(dev_labels)), len(np.unique(dev_labels))))
    for i, j in zip(dev_labels, pred_labels):
        conf_m[i, j] += 1
    return accuracy, conf_m
