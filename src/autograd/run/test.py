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
    parser.add_argument("--test-dataset", type=Path, dest="test_dataset", required=True)
    parser.add_argument("--output-dir", type=Path, dest="output_dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    parser.add_argument("--model", type=str, choices=["cnn", "fcn"], dest="model")
    parser.add_argument("--model-path", type=Path, dest="model_path")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = MNIST(args.test_dataset)
    test_dataloader = Dataloader(test_dataset, args.batch_size, False)

    if args.model == "cnn":
        model = CNNModel()
    else:
        model = FCNModel()

    model.load(args.model_path)
    trainer = Trainer(model, None)
    test_results = trainer.test(test_dataloader, metrics=[Accuracy()])
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(test_results, f)
