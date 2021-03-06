import inspect
from abc import abstractmethod
from pathlib import Path
from typing import List

import numpy as np

from autograd.layer import Linear, ReLU, Softmax, Sigmoid, Layer, Conv2D
from autograd.tensor import Tensor


class Model:

    def parameters(self) -> List[Tensor]:
        params = []
        for name, value in inspect.getmembers(self):
            if issubclass(type(value), Layer):
                params.extend(value.parameters())

        return params

    def save(self, path: Path):
        parms = self.parameters()
        save_dict = {str(i): parms[i].data for i in range(len(parms))}
        with open(path, "wb") as f:
            np.savez(f, **save_dict)

    def load(self, path: Path):
        save_dict = np.load(str(path))
        for i, param in enumerate(self.parameters()):
            param.data = save_dict[str(i)]

    @abstractmethod
    def forward(self, *args: Tensor) -> Tensor:
        pass

    @abstractmethod
    def compute_loss(self, y: Tensor, target: Tensor) -> Tensor:
        pass

    def __call__(self, *args: Tensor) -> Tensor:
        return self.forward(*args)


class FCNModel(Model):
    def __init__(self):
        self.lin1 = Linear((28*28, 256))
        self.lin2 = Linear((256, 10))
        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = Tensor.reshape(x, (batch_size, 1, -1))
        y1 = self.lin1(x)
        r1 = self.relu(y1)
        y2 = self.lin2(r1)
        sm = self.softmax(y2)
        return sm

    def compute_loss(self, y: Tensor, target: Tensor) -> Tensor:
        return Tensor.cross_entropy(y, target)


class CNNModel(Model):
    def __init__(self):
        self.conv1 = Conv2D(input_shape=(28,28,1), kernel_shape=(3,3), num_filters=10)
        self.conv2 = Conv2D(input_shape=(26,26,10), kernel_shape=(3,3), num_filters=10)
        self.relu = ReLU()
        self.lin = Linear(shape=(24*24*10, 10))
        self.sm = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        c1 = self.conv1(x)
        r1 = self.relu(c1)
        c2 = self.conv2(r1)
        r2 = self.relu(c2)
        y = self.lin(Tensor.reshape(r2, (batch_size, 1, -1)))
        sm = self.sm(y)
        return sm

    def compute_loss(self, y: Tensor, target: Tensor) -> Tensor:
        return Tensor.cross_entropy(y, target)
