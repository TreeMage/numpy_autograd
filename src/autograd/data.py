import random
from abc import abstractmethod
from typing import Tuple

import numpy as np

from autograd.tensor import Tensor


class Dataset:

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        pass


class DataloaderIterator:
    def __init__(self, dataloader):
        self.idx = 0
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        n = self.dataloader[self.idx]
        self.idx += 1
        return n


class Dataloader:

    def __init__(self, dataset: Dataset, batch_size=1, shuffle=True):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)

    def reshuffle(self):
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        return DataloaderIterator(self)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        start = idx * self.batch_size
        stop = start + self.batch_size
        data = [self.dataset[i] for i in self.indices[start:stop]]
        samples, labels = list(zip(*data))
        return Tensor(np.stack(samples)), Tensor(np.stack(labels))
