from abc import abstractmethod
from typing import List

import numpy as np

from autograd.tensor import Tensor


class Optimizer:
    def __init__(self):
        self.parameters = None

    def bind(self, parameters: List[Tensor]):
        self.parameters = parameters

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class SGD(Optimizer):

    def __init__(self, lr: float = 4e-3, weight_decay: float = 0):
        super(SGD, self).__init__()
        self.weight_decay = weight_decay
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * (np.mean(param.grad, axis=0) + self.weight_decay * param.data)