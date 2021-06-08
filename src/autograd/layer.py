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


class Conv2D(Layer):

    def __init__(self, input_shape, kernel_shape, num_filters):
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters
        self.kernel = Tensor.rand((num_filters, *kernel_shape, input_shape[-1]))
        # A single bias per filter so we have to repeat it accordingly
        self.bias = Tensor.repeat(Tensor.repeat(Tensor.zeros((1, 1, num_filters)),
                                                axis=0, repeats=self.output_shape[0]),
                                  axis=1, repeats=self.output_shape[1])

    @property
    def output_shape(self):
        return (self.input_shape[0] - self.kernel_shape[0] + 1,
                self.input_shape[1] - self.kernel_shape[1] + 1,
                self.num_filters)

    def forward(self, x: Tensor) -> Tensor:
        return Tensor.conv2d(x, self.kernel) + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.kernel, self.bias]