import numpy as np

from autograd.tensor import Tensor

if __name__ == "__main__":
    w1 = Tensor(np.array([[-0.5057, 0.3987, -0.8943],
                          [0.3356, 0.1673 ,0.8321],
                          [-0.3485, -0.4597, -0.1121]]))
    b1 = Tensor.zeros((1, 3))
    w2 = Tensor(np.array([[0.4047, 0.9563],
                         [-0.8192, -0.1274],
                         [0.3662,-0.7252]]))
    b2 = Tensor.zeros((1, 2))
    x = Tensor(np.array([[0.4183, 0.5209, 0.0291]]))
    label = Tensor(np.array(([[0.7095, 0.0942]])))

    y1 = x @ w1 + b1
    s = Tensor.sigmoid(y1)
    y2 = s @ w2 + b2
    sm = Tensor.softmax(y2, axis=-1)
    loss = Tensor.cross_entropy(sm, label)
    loss.backward()

    print("Forward:")
    for t in [y1, s, y2, sm]:
        print(t.data)

    print("Backward:")
    for t in [y1, s, y2, sm]:
        print(t.grad)

    print("Delta Weights:")
    for i,t in enumerate([(w1, b1), (w2,b2)]):
        w, b = t
        print(f"FC {i+1} weight:")
        print(w.grad)
        print(f"FC {i+1} bias:")
        print(b.grad)