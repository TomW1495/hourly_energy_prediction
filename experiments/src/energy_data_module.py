from pathlib import Path
import torch
import numpy as np
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from .config import DATA_PATH
from typing import Callable, List, Tuple
from energy_dataset.data import BaseDataset


class TimeseriesDataset(Dataset):   
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return self.X.__len__() - 1

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


class EnergyDataModule(LightningDataModule):
    def __init__(self, lookback = 1, batch_size = 128, num_workers=0):
        super().__init__()
        self.lookback = lookback
        self.batch_size = batch_size
        self.num_workers = num_workers
        base_dataset = BaseDataset(DATA_PATH, lookback = self.lookback)
        self.X_train = base_dataset.train_x
        self.y_train = base_dataset.train_y
        self.X_val = base_dataset.val_x
        self.y_val = base_dataset.val_y
        self.X_test = base_dataset.test_x
        self.y_test = base_dataset.test_y


    def train_dataloader(self):
        train_dataset = TimeseriesDataset(self.X_train,
                                          self.y_train)
                                          
        train_loader = DataLoader(train_dataset,
                                  batch_size = self.batch_size,
                                  shuffle = False,
                                  num_workers = self.num_workers,
                                  drop_last = True)
        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val,
                                        self.y_val)
        val_loader = DataLoader(val_dataset,
                                batch_size = self.batch_size,
                                shuffle = False,
                                num_workers = self.num_workers,
                                drop_last = True)
        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test,
                                         self.y_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size = self.batch_size,
                                 shuffle = False,
                                 num_workers = self.num_workers,
                                 drop_last = True)
        return test_loader

        