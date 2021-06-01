from abc import abstractmethod
from typing import List, Tuple

from autograd.tensor import Tensor


class Layer:

    @abstractmethod
    def parameters(self) -> List[Tensor]:
        pass

    def forward(self, *args: Tensor) -> Tensor:
        pass

    def __call__(self, *args) -> Tensor:
        return self.forward(*args)


class Linear(Layer):
    def __init__(self, shape: Tuple):
        self.shape = shape
        self.W = Tensor.rand(shape)
        self.b = Tensor.zeros((1, shape[1]))

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.W) + self.b

    def parameters(self) -> List[Tensor]:
        return [self.W, self.b]


class ReLU(Layer):

    def forward(self, x: Tensor) -> Tensor:
        return Tensor.relu(x)

    def parameters(self) -> List[Tensor]:
        return []


class Sigmoid(Layer):

    def forward(self, x: Tensor) -> Tensor:
        return Tensor.sigmoid(x)

    def parameters(self) -> List[Tensor]:
        return []


class Softmax(Layer):

    def __init__(self, axis: int = -1):
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return Tensor.softmax(x, axis=self.axis)

    def parameters(self) -> List[Tensor]:
        return []