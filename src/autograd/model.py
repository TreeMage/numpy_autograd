import inspect
from abc import abstractmethod
from typing import List

import attr

from autograd.layer import Linear, ReLU, Softmax, Sigmoid, Layer
from autograd.tensor import Tensor


class Model:

    def parameters(self) -> List[Tensor]:
        params = []
        for name, value in inspect.getmembers(self):
            if issubclass(type(value), Layer):
                params.extend(value.parameters())

        return params

    @abstractmethod
    def forward(self, *args: Tensor) -> Tensor:
        pass


class FCNModel(Model):
    def __init__(self):
        self.lin1 = Linear((3, 3))
        self.lin2 = Linear((3, 2))
        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.lin1(x)
        r1 = self.relu(y1)
        y2 = self.lin2(r1)
        sm = self.softmax(y2)
        return sm