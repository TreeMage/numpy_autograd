from abc import abstractmethod

import numpy as np

from autograd.tensor import Tensor


class Metric:

    @abstractmethod
    def forward(self, y: Tensor, label: Tensor):
        pass

    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def reset(self):
        pass


class Accuracy(Metric):

    def __init__(self):
        self.hits = 0
        self.count = 0

    def forward(self, y: Tensor, label: Tensor):
        self.hits += (np.argmax(y.data, axis=-1) == np.argmax(label.data, axis=-1)).sum().item()
        self.count += np.prod(y.shape[:-1])

    def compute(self) -> float:
        return self.hits / self.count

    def name(self) -> str:
        return "Accuracy"

    def reset(self):
        self.hits = 0
        self.count = 0