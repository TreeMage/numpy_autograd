from pathlib import Path

import numpy as np

from autograd.data import Dataloader
from autograd.datasets.mnist import MNIST
from autograd.metric import Accuracy
from autograd.model import FCNModel
from autograd.optim import SGD
from autograd.tensor import Tensor
from autograd.layer import Linear
from autograd.trainer import Trainer


def fcn_example():
    x = Tensor(np.array([[0.4183, 0.5209, 0.0291]]), requires_grad=False)
    label = Tensor(np.array([[0.7095, 0.0942]]),requires_grad=False)
    W1 = Tensor(np.array([[-0.5058, 0.3987, -0.8943],
                          [0.3356, 0.1673, 0.8321],
                          [-0.3485, -0.4597, -0.1121]]))
    b1 = Tensor(np.array([[0.,0.,0.]]))
    W2 = Tensor(np.array([[0.4047, 0.9563],
                          [-0.8192, -0.1274],
                          [0.3662, -0.7252]]))
    b2 = Tensor(np.array([[0., 0.]]))
    y1 = (x @ W1) + b1
    sig1 = Tensor.sigmoid(y1)
    y2 = (sig1 @ W2) + b2
    sm = Tensor.softmax(y2, axis=-1)
    loss = Tensor.cross_entropy(sm, label)
    loss.backward()
    print(W1.grad)


if __name__ == "__main__":

    dataset = MNIST(Path("../../data/mnist_train.csv"))
    model_path = Path("../../data/model")
    dataloader = Dataloader(dataset, 8, True)
    model = FCNModel()
    optimizer = SGD(lr=0.01)
    trainer = Trainer(model, optimizer, 5)
    trainer.fit(dataloader, [Accuracy()])
    model.save(model_path)

