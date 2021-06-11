import argparse
import json
from pathlib import Path

import numpy as np

from autograd.data import Dataloader
from autograd.datasets.mnist import MNIST
from autograd.metric import Accuracy
from autograd.model import FCNModel, CNNModel
from autograd.optim import SGD
from autograd.trainer import Trainer


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
    parser.add_argument("--model", type=str, choices=["cnn", "fcn"], dest="model")
    parser.add_argument("--test-frequency", type=int, dest="run_test_every_n_epochs", default=1)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = MNIST(args.train_dataset)
    train_dataloader = Dataloader(train_dataset, args.batch_size, args.shuffle)

    test_dataset = MNIST(args.test_dataset)
    test_dataloader = Dataloader(test_dataset, args.batch_size, False)

    if args.model == "cnn":
        model = CNNModel()
    else:
        model = FCNModel()

    print(f"Training {args.model.upper()} for {args.epochs} epochs with lr={args.lr} and weight-decay={args.weight_decay}.")
    optimizer = SGD(lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(model, optimizer, args.epochs)
    test_results = trainer.fit(train_dataloader, test_dataloader,
                               metrics=[Accuracy()],
                               run_test_every_n_epochs=args.run_test_every_n_epochs)
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(test_results, f)

    model.save(args.output_dir / "model")

