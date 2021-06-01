import numpy as np


def softmax(x: np.ndarray, axis: int):
    exped = np.exp(x - np.max(x, axis=axis))
    return exped / exped.sum(exped, axis=axis)

