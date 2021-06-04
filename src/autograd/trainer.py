from typing import List, Dict

import tqdm

from autograd.data import Dataloader
from autograd.metric import Metric
from autograd.model import Model
from autograd.optim import Optimizer


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

    def __init__(self, model: Model, optimizer: Optimizer, num_epochs: int):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs

    def fit(self, train_dataloader: Dataloader, metrics: List[Metric] = None):
        avg = RunningAverage(self.MOVING_AVG_WINDOW_SIZE)
        self.optimizer.bind(self.model.parameters())
        for epoch in range(self.num_epochs):
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


