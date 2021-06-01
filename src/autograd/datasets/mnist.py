import csv
from pathlib import Path

from autograd.data import Dataset


class MNIST(Dataset):

    def __init__(self, path: Path):
        self.path = path
        self.data, self.labels = self.load_data()

    def load_data(self):
        import numpy as np
        data, labels = [], []
        with open(self.path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                label = np.zeros((1,10))
                label[0, int(line[0])] = 1
                d = np.reshape(np.array([float(x) / 255 for x in line[1:]]), (1,28,28))
                data.append(d)
                labels.append(label)

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


