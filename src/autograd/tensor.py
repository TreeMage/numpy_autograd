from abc import abstractmethod
from typing import Optional, List, Tuple

import numpy as np

EPSILON = 1e-8


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad=True, grad_fn: 'Function' = None):
        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)

    @classmethod
    def rand(cls, shape, requires_grad=True):
        return Tensor(np.random.rand(*shape) - 0.5, requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape, requires_grad=True):
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def neg(cls, this):
        return Neg.apply(this)

    @classmethod
    def mul(cls, this, that):
        return Mul.apply(this, that)

    @classmethod
    def div(cls, this, that):
        return Div.apply(this, that)

    @classmethod
    def add(cls, this, that):
        return Add.apply(this, that)

    @classmethod
    def sub(cls, this, that):
        return Sub.apply(this, that)

    @classmethod
    def exp(cls, this):
        return Exp.apply(this)

    @classmethod
    def pow(cls, this, exponent):
        return Pow.apply(this, exponent)

    @classmethod
    def dot(cls, this, that):
        return Tensor.sum(this * that, axis=-1)

    @classmethod
    def sigmoid(cls, this):
        return Sigmoid.apply(this)

    @classmethod
    def log(cls, this):
        return Log.apply(this)

    @classmethod
    def relu(cls, this) :
        return ReLU.apply(this)

    @classmethod
    def matmul(cls, this, that):
        return Matmul.apply(this, that)

    @classmethod
    def sum(cls, this, axis: int = None, keepdims = False):
        return Sum.apply(this, axis, keepdims)

    @classmethod
    def reshape(cls, this, shape):
        return Reshape.apply(this, shape)

    @classmethod
    def repeat(cls, this, axis: int, repeats: int):
        return Repeat.apply(this, axis, repeats)

    @classmethod
    def softmax(cls, this, axis: int):
        max = Tensor.max(this, axis=axis)
        adjusted = this - Tensor.repeat(max, axis=axis, repeats=this.shape[axis])
        exped = Tensor.exp(adjusted)
        sum = Tensor.sum(exped, axis=axis, keepdims=True)
        return exped / Tensor.repeat(sum, axis=axis, repeats=this.shape[axis])

    @classmethod
    def mean(cls, this, axis: int = None):
        return Mean.apply(this, axis)

    @classmethod
    def cross_entropy(cls, this, labels):
        return Tensor.mean(Tensor.sum(-labels * Tensor.log(this), axis=-1))

    @classmethod
    def max(cls, this, axis: int = None):
        return Max.apply(this, axis)

    @classmethod
    def conv2d(cls, inp, kernel):
        return Conv2D.apply(inp, kernel)

    def __neg__(self):
        return Tensor.neg(self)

    def __mul__(self, other):
        return Tensor.mul(self, other)

    def __truediv__(self, other):
        return Tensor.div(self, other)

    def __add__(self, other):
        return Tensor.add(self, other)

    def __sub__(self, other):
        return Tensor.sub(self, other)

    def __matmul__(self, other):
        return Tensor.matmul(self, other)

    def __pow__(self, power, modulo=None):
        return Tensor.pow(self, power)

    def __repr__(self) -> str:
        return f"Tensor[shape={self.shape},data={self.data},grad={self.grad}]"

    def _compute_graph(self):
        def _walk(node, visited, nodes):
            visited.add(node)
            if node.grad_fn is not None:
                for n in node.grad_fn.stored_tensors:
                    if n not in visited:
                        _walk(n, visited, nodes)
                nodes.append(node)
            return nodes
        return _walk(self, set(), [])

    def backward(self):
        if self.grad is None:
            assert self.size == 1, "Grad can only be implicitly created for leaf tensors."
            self.grad = np.ones(self.shape)

        tensors = self._compute_graph()
        for tensor in reversed(tensors):
            grads = tensor.grad_fn.backward(tensor.grad)
            prev = tensor.grad_fn.stored_tensors
            for t, g in zip(prev, grads):
                if t.grad is None:
                    t.grad = g
                else:
                    t.grad += g


class Function:
    def __init__(self):
        self.stored_tensors = []
        self.stored_metadata = []

    def save_tensors_for_backward(self, *args: Tensor):
        self.stored_tensors.extend(args)

    def save_metadata_for_backward(self, *args):
        self.stored_metadata.extend(args)

    @abstractmethod
    def forward(self, *args) -> Tensor:
        pass

    @abstractmethod
    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        pass

    @classmethod
    def apply(cls, *args) -> Tensor:
        return cls().forward(*args)


class Neg(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(-x.data, grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        return [-grad_in]


class Mul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.save_tensors_for_backward(x, y)
        return Tensor(x.data * y.data, grad_fn=self)

    def backward(self, grad_in) -> List[np.ndarray]:
        x, y = self.stored_tensors
        return [y.data * grad_in, x.data * grad_in]


class Add(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.save_tensors_for_backward(x, y)
        return Tensor(x.data + y.data, grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        return [grad_in, grad_in]


class Sub(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.save_tensors_for_backward(x, y)
        return Tensor(x.data - y.data, grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        return [grad_in, -grad_in]


class Div(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.save_tensors_for_backward(x, y)
        return Tensor(x.data / y.data, grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        x, y = self.stored_tensors
        return [grad_in / y.data, (-x.data * grad_in) / y.data ** 2]


class Pow(Function):
    def forward(self, x: Tensor, pow: float) -> Tensor:
        self.save_tensors_for_backward(x)
        self.save_metadata_for_backward(pow)
        return Tensor(x.data ** pow, grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        x, = self.stored_tensors
        pow, = self.stored_metadata
        return [pow * x.data ** (pow - 1) * grad_in]


class Log(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensors_for_backward(x)
        return Tensor(np.log(x.data + EPSILON), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        x, = self.stored_tensors
        return [grad_in / (x.data + EPSILON)]


class Matmul(Function):
    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        self.save_tensors_for_backward(x, w)
        return Tensor(x.data @ w.data, grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        x, w = self.stored_tensors
        grad_x = grad_in @ np.swapaxes(w.data, -1, -2)
        grad_w = np.swapaxes(x.data, -1, -2) @ grad_in
        return [grad_x, grad_w]


class Mean(Function):
    def forward(self, x: Tensor, axis: int = None) -> Tensor:
        self.save_tensors_for_backward(x)
        self.save_metadata_for_backward(axis,)
        return Tensor(np.mean(x.data, axis=axis, keepdims=True), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        x, = self.stored_tensors
        axis, = self.stored_metadata
        if axis is None:
            return [np.full(x.shape, grad_in.item() / x.size)]
        n = x.shape[axis]
        scaled = grad_in / n
        scaled = np.expand_dims(scaled, axis)
        return [np.repeat(scaled, n, axis=axis)]


class Exp(Function):
    def forward(self, x: Tensor) -> Tensor:
        self.save_tensors_for_backward(x)
        return Tensor(np.exp(x.data), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        x, = self.stored_tensors
        return [np.exp(x.data) * grad_in]


class Sigmoid(Function):

    def sigmoid(self, x: np.ndarray):
        return 1 / (1+ np.exp(-x))

    def forward(self, x: Tensor) -> Tensor:
        self.save_tensors_for_backward(x)
        return Tensor(self.sigmoid(x.data), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        x, = self.stored_tensors
        sig = self.sigmoid(x.data)
        return [sig * (1 - sig) * grad_in]


class Sum(Function):
    def forward(self, x: Tensor, axis: int = None, keepdims = False):
        self.save_tensors_for_backward(x)
        self.save_metadata_for_backward(axis, x.shape, keepdims)
        return Tensor(np.sum(x.data, axis=axis, keepdims=keepdims), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        axis, shape, keepdims = self.stored_metadata
        if axis is None:
            return [np.ones(shape) * grad_in]
        ext = grad_in if keepdims else np.expand_dims(grad_in, axis=axis)
        return [np.repeat(ext, shape[axis], axis=axis)]


class Reshape(Function):
    def forward(self, x: Tensor, new_shape: Tuple[int, ...]) -> Tensor:
        self.save_tensors_for_backward(x)
        self.save_metadata_for_backward(x.shape)
        return Tensor(np.reshape(x.data, new_shape), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        old_shape, = self.stored_metadata
        return [np.reshape(grad_in, old_shape)]


class Repeat(Function):
    def forward(self, x: Tensor, axis: int, repeats: int):
        self.save_tensors_for_backward(x)
        self.save_metadata_for_backward(axis)

        return Tensor(np.repeat(x.data, repeats, axis=axis), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        axis, = self.stored_metadata
        return [np.sum(grad_in, axis=axis, keepdims=True)]


class Max(Function):
    def forward(self, x: Tensor, axis: int = None) -> Tensor:
        self.save_tensors_for_backward(x)
        self.save_metadata_for_backward(axis, x.shape)
        return Tensor(np.max(x.data, axis=axis, keepdims=True), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        axis, shape = self.stored_metadata
        x, = self.stored_tensors
        mask = x.data == np.amax(x.data, axis=axis, keepdims=True)
        if axis is None:
            ext = np.full(shape,grad_in.item())
        else:
            ext = np.repeat(grad_in, shape[axis],axis=axis)

        return [ext * mask]


class ReLU(Function):

    def forward(self, x: Tensor) -> Tensor:
        self.save_tensors_for_backward(x)
        return Tensor(np.maximum(x.data, 0), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        x, = self.stored_tensors
        return [grad_in * (x.data >= 0)]


def im2col(x: np.ndarray, kernel_shape: Tuple[int, ...], stride=1):
    _, fH, fW, fC = kernel_shape
    bs, H, W, C = x.shape
    oH = (H - fH) // stride + 1
    oW = (W - fW) // stride + 1
    col = np.zeros((bs, oH*oW, fH*fW * C))
    for i in range(oH):
        for j in range(oW):
            cube = x[:, i*stride:i*stride+fH,j*stride:j*stride+fW, :]
            col[:, i * oH + j, :] = cube.reshape(bs, -1)
    return col


def col2im(col: np.ndarray, oH, oW):
    bs, _, oC = col.shape
    return col.reshape((bs, oH, oW, oC))


def conv2d(inp: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    b, iH, iW, iC = inp.shape
    nf, fH, fW, fC = kernel.shape
    oH = iH - fH + 1
    oW = iW - fW + 1
    col = im2col(inp, kernel.shape)
    reshaped_kernel = kernel.reshape(nf, -1)
    mul = np.dot(col, reshaped_kernel.T)
    return col2im(mul, oH, oW)
    #if len(inp.shape) == 3:
    #    # Implicit batch dim
    #    inp = np.expand_dims(inp, axis=0)
    #b, iH, iW, iC = inp.shape
    #nf, fH, fW, fC = kernel.shape
    #h = iH - fH + 1
    #w = iW - fW + 1
    #res = np.zeros((inp.shape[0], h, w, nf))
    #assert iC == fC, f"Input channels and kernel channels have to match. ({iC} != {fC})"
    #for y in range(h):
    #    for x in range(w):
    #        sub_img = inp[:, y:y + fH, x:x + fW, :]
    #        for k in range(nf):
    #            filter = np.repeat(np.expand_dims(kernel[k], axis=0), axis=0, repeats=b)
    #            res[:, y, x, k] = np.apply_over_axes(np.sum, filter * sub_img, axes=[1, 2, 3]).reshape(-1)
    #return res

class Conv2D(Function):
    def forward(self, inp: Tensor, kernel: Tensor) -> Tensor:
        self.save_tensors_for_backward(inp, kernel)
        return Tensor(conv2d(inp.data, kernel.data), grad_fn=self)

    def backward(self, grad_in: np.ndarray) -> List[np.ndarray]:
        inp, kernel = self.stored_tensors
        # batch size
        bs = inp.shape[0]
        kH, kW = kernel.shape[1], kernel.shape[2]
        # Gradient with respect to the input
        bk = np.rot90(np.transpose(kernel.data, axes=(3, 1, 2, 0)), k=2, axes=(1, 2))
        padded = np.pad(grad_in, ((0, 0), (kH - 1, kH - 1), (kW - 1, kW - 1), (0, 0)))
        inp_grad = conv2d(padded, bk)
        # Gradient with respect to the weights
        kernel_grad = np.zeros((bs, *kernel.shape))
        for co in range(kernel.shape[0]):
            sd = np.expand_dims(grad_in[:, :, :, co], axis=-1).sum(axis=0, keepdims=1)
            for ci in range(kernel.shape[3]):
                si = np.expand_dims(inp.data[:, :, :, ci], axis=-1)
                # Due to batch_size > 1 we somehow need to reduce the last dimension?
                kg = conv2d(si, sd)
                kernel_grad[:, co, :, :, ci] += kg.squeeze(axis=-1)

        return [inp_grad, kernel_grad]
    

if __name__ == "__main__":
   x = np.zeros((3,227,227,3))
   k = np.zeros((10,11,11,3))
   conv2d(x, k)
