import argparse
import json
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=Path, dest="train_dataset", required=True)
    parser.add_argument("--test-dataset", type=Path, dest="test_dataset", required=True)
    parser.add_argument("--output-dir", type=Path, dest="output_dir", required=True)
    parser.add_argument("--lr", type=float, dest="lr", default=4e-3)
    parser.add_argument("--weight-decay", type=float, dest="weight_decay", default=0.)
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    parser.add_argument("--shuffle", type=bool, default=True, dest="shuffle")
    parser.add_argument("--epochs", type=int, dest="epochs", default=100)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = MNIST(args.train_dataset)
    train_dataloader = Dataloader(train_dataset, args.batch_size, args.shuffle)

    test_dataset = MNIST(args.test_dataset)
    test_dataloader = Dataloader(test_dataset, args.batch_size, False)

    model = FCNModel()

    optimizer = SGD(lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(model, optimizer, args.epochs)
    trainer.fit(train_dataloader, [Accuracy()])
    test_results = trainer.test(test_dataloader, [Accuracy()])
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(test_results, f)

    model.save(args.output_dir / "model")

