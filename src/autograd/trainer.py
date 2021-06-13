import math
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import tqdm

from autograd.data import Dataloader
from autograd.metric import Metric
from autograd.model import Model
from autograd.optim import Optimizer


class EpochFinishedCallback:

    def apply(self, model: Model, epoch: int, did_test: bool, test_metrics: Optional[Dict[str,Any]]):
        pass

    def __call__(self, model: Model, epoch: int, did_test: bool, test_metrics: Optional[Dict[str, Any]]):
        self.apply(model, epoch, did_test, test_metrics)


class CheckpointCallback(EpochFinishedCallback):

    CHECKPOINT_FORMAT_STRING = "checkpoint-epoch={}-{}={:.4f}.nn"

    def __init__(self, output_dir: Path, metric: str, k=1, mode="max"):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metric = metric
        self.k = k
        if mode not in ["min", "max"]:
            raise ValueError("Checkpointing mode has to be either 'min' or 'max'.")
        self.mode = mode
        self.current = {}

    def _determine_worst_value(self) -> float:
        if len(self.current) == 0:
            return math.inf if self.mode == "min" else -math.inf
        else:
            return max(self.current.keys()) if self.mode == "min" else min(self.current.keys())

    def apply(self, model: Model, epoch: int, did_test: bool, test_metrics: Optional[Dict[str,Any]]):
        if did_test:
            metric_value = test_metrics[self.metric]
            if len(self.current) >= self.k:
                worst = self._determine_worst_value()
                worst_path = self.current.pop(worst)
                os.remove(worst_path)

            path = self.output_dir / self.CHECKPOINT_FORMAT_STRING.format(epoch, self.metric, metric_value)
            model.save(path)
            self.current[metric_value] = path


class RunningAverage:
    def __init__(self, size: int):
        self.size = size
        self.values = []

    def append(self, value: float):
        self.values.append(value)
        if len(self.values) > self.size:
            self.values.pop(0)

    def compute(self) -> float:
        return sum(self.values) / len(self.values)


class Trainer:

    MOVING_AVG_WINDOW_SIZE = 100

    def __init__(self, model: Model, optimizer: Optimizer = None, num_epochs: int = 10):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def _run_train_epoch(self, train_dataloader: Dataloader, epoch: int, metrics: List[Metric] = None):
        avg = RunningAverage(self.MOVING_AVG_WINDOW_SIZE)
        [metric.reset() for metric in metrics]
        train_dataloader.reshuffle()
        pbar = tqdm.trange(len(train_dataloader))
        for batch_idx in pbar:
            self.optimizer.zero_grad()
            x, labels = train_dataloader[batch_idx]
            y = self.model(x)
            loss = self.model.compute_loss(y, labels)

            [metric.forward(y, labels) for metric in metrics]

            avg.append(loss.data.item())
            desc = f"Epoch: {epoch} Loss: {avg.compute():.3f} "
            metric_desc = ' '.join([f"{metric.name()}: {metric.compute():.3f}" for metric in metrics])
            pbar.set_description(desc + metric_desc)
            loss.backward()
            self.optimizer.step()

    def fit(self, train_dataloader: Dataloader, test_dataloader: Dataloader,
            metrics: List[Metric] = None,
            callbacks: List[EpochFinishedCallback] = None,
            run_test_every_n_epochs: int = 1) -> Dict[str, List[float]]:
        if self.optimizer is None:
            raise ValueError("An optimizer has to be provided for fitting the model.")
        self.optimizer.bind(self.model.parameters())
        test_results = defaultdict(list)
        epoch_result = {}
        for epoch in range(self.num_epochs):
            do_test = (epoch + 1) % run_test_every_n_epochs == 0
            self._run_train_epoch(train_dataloader, epoch, metrics)
            if do_test:
                epoch_result = self.test(test_dataloader, metrics)
                for metric, value in epoch_result.items():
                    test_results[metric].append(value)

            if callbacks is not None:
                for callback in callbacks:
                    callback(self.model, epoch, do_test, epoch_result if do_test else None)

        return test_results

    def test(self, test_dataloader: Dataloader, metrics: List[Metric] = None) -> Dict[str, float]:
        avg = RunningAverage(len(test_dataloader))
        [metric.reset() for metric in metrics]
        pbar = tqdm.trange(len(test_dataloader))
        for batch_idx in pbar:
            x, labels = test_dataloader[batch_idx]
            y = self.model(x)
            loss = self.model.compute_loss(y, labels)

            [metric.forward(y, labels) for metric in metrics]

            avg.append(loss.data.item())
            desc = f"[Testing] Loss: {avg.compute():.3f} "
            metric_desc = ' '.join([f"{metric.name()}: {metric.compute():.3f}" for metric in metrics])
            pbar.set_description(desc + metric_desc)

        return_dict = {metric.name(): metric.compute() for metric in metrics}
        return_dict["Test Loss"] = avg.compute()
        return return_dict


