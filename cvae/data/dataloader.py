"""
Data loading utilities for cVAE gene expression models.

Includes GEPcached dataset and setup_data_loaders for semi-supervised splits.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def split_sup_unsup_valid(X, y, sup_num, val_num=1000, rnd=1234, stratifyclasses=None):
    """
    Helper function for splitting the data into supervised, un-supervised and validation parts.
    Args:
        X: GEPs (1d float tensor) assume same order as y
        y: purity estimates (float) assume same order as X
        sup_num: what number of examples is supervised
        val_num: what number of last examples to use for validation
    Returns:
        Splits of data by sup_num number of supervised examples
    """
    data_size = len(X)
    sup_only = (data_size - val_num - sup_num) == 0
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=int(val_num), random_state=rnd, stratify=stratifyclasses)
    if sup_only:
        X_unsup, X_sup, y_unsup, y_sup = (pd.DataFrame(), X_train, pd.DataFrame(), y_train)
    else:
        X_unsup, X_sup, y_unsup, y_sup = train_test_split(X_train, y_train, test_size=int(sup_num), random_state=rnd)
    return X_sup, y_sup, X_unsup, y_unsup, X_valid, y_valid

class GEPcached(torch.utils.data.Dataset):
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    data_valid, labels_valid = None, None

    def __init__(self, input_GEP, input_classes, mode, sup_num, 
                 val_num, train_data_size, 
                 num_classes=1,
                 use_cuda=False,
                 stratifyclasses=None):
        self.mode = mode
        self.data = input_GEP
        self.num_classes = num_classes
        self.stratifyclasses = stratifyclasses

        # Transform targets
        self.targets = input_classes

        self.validation_num = val_num
        self.validation_size = val_num
        self.sup_num = sup_num
        self.sample_num = self.data.shape[0]
        self.train_data_size = self.sample_num - val_num
        self.gene_size = self.data.shape[1]
        self.use_cuda = use_cuda

        assert mode in [
            "sup",
            "unsup",
            "test",
            "valid",
            "full"
        ], "invalid train/test option values"

        if mode in ["sup", "unsup", "valid", "full"]:
            if self.train_data_sup is None:
                if sup_num is None:
                    assert mode == "unsup", "please provide sup_num"
                    self.train_data_unsup, self.train_labels_unsup = (
                        self.data,
                        self.targets,
                    )
                else:
                    (
                        self.train_data_sup,
                        self.train_labels_sup,
                        self.train_data_unsup,
                        self.train_labels_unsup,
                        self.data_valid,
                        self.labels_valid,
                    ) = split_sup_unsup_valid(self.data, self.targets, self.sup_num, self.validation_num, 
                                              stratifyclasses=self.stratifyclasses)
                    self.valid_indices = self.data_valid.index
                    self.train_sup_indices = self.train_data_sup.index

            if mode == "sup":
                self.data, self.targets = (
                    self.train_data_sup,
                    self.train_labels_sup,
                )
                if self.targets.shape[0] == 0 or self.num_classes == 0:
                    self.targets = torch.empty(self.data.shape[0], dtype=torch.float32).view(-1, 1) * np.nan

            elif mode == "unsup":
                self.data = self.train_data_unsup
                self.targets = (
                    torch.empty(self.train_labels_unsup.shape[0],
                                device=self.targets.device).
                                view(-1, 1)
                ) * np.nan

            elif mode == "full":
                self.data
                self.targets

            else:
                self.data, self.targets = (
                    self.data_valid,
                    self.labels_valid,
                )

        self.data = torch.tensor(self.data.values, dtype=torch.float32)
        if not isinstance(self.targets, torch.Tensor):
            self.targets = torch.tensor(self.targets.values, dtype=torch.float32)

        if torch.all(self.targets == 0).item() or self.num_classes == 0:
            self.targets = torch.tensor([]).reshape(self.data.shape[0], 0)

        if use_cuda:
            device = "mps" if torch.backends.mps.is_available() else "cuda"
            self.data = self.data.to(device)
            self.targets = self.targets.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index: Index or slice object
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.mode in ["sup", "unsup", "valid"]:
            dat, target = self.data[index], self.targets[index]
        elif self.mode == "test":
            dat, target = self.data[index], self.targets[index]
        elif self.mode == "full":
            dat, target = self.data[index], self.targets[index]
        else:
            assert False, "invalid mode: {}".format(self.mode)
        return dat, target

def setup_data_loaders(
    dataset, use_cuda, batch_size, sup_num=None, **kwargs
):
    """
    Helper function for setting up pytorch data loaders for a semi-supervised dataset.
    Args:
        dataset: the data to use
        use_cuda: use GPU(s) for training
        batch_size: size of a batch of data to output when iterating over the data loaders
        sup_num: number of supervised data examples
        kwargs: other params for the pytorch data loader
    Returns:
        dict: data loaders for "test", "sup", and "valid" splits
    """
    cached_data = {}
    loaders = {}
    for mode in ["test", "sup", "valid"]:
        if sup_num is None and mode == "sup":
            return loaders["unsup"], loaders["test"]
        cached_data[mode] = dataset(
            mode=mode, sup_num=sup_num, use_cuda=use_cuda, **kwargs 
        )
        loaders[mode] = DataLoader(
            cached_data[mode], batch_size=batch_size, shuffle=True, drop_last=True, 
        )
    return loaders