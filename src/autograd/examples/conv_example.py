import numpy as np

from autograd.tensor import Tensor

if __name__ == "__main__":
    # Reshape data using column-major indexing
    test_data = np.array([0.1, -0.2, 0.5, 0.6, 1.2, 1.4, 1.6, 2.2, 0.01, 0.2, -0.3, 4.0, 0.9, 0.3, 0.5, 0.65, 1.1, 0.7, 2.2, 4.4, 3.2, 1.7, 6.3, 8.2]).reshape((1, 4, 3, 2), order="F")
    test_im = Tensor(test_data)
    # First reshape the kernel to the right dimensions using column-major indexing and than transpose from (W, H, C, N) to (N, H, W, C)
    test_kernel_data = np.array([0.1, -0.2, 0.3, 0.4, 0.7, 0.6, 0.9, -1.1, 0.37, -0.9, 0.32, 0.17, 0.9, 0.3, 0.2, -0.7])\
        .reshape((2,2,2,2),order="F").transpose((3, 0, 1, 2))
    test_kernel = Tensor(test_kernel_data)
    res = Tensor.conv2d(test_im, test_kernel)
    print("Forward: ")
    print(res.data.reshape(-1))
    # Also reshape the deltas to be column-major
    deltas = np.array([0.1, 0.33, -0.6, -0.25, 1.3, 0.01, -0.5, 0.2, 0.1, -0.8, 0.81, 1.1]).reshape((1, 3, 2, 2),order="F")
    # Compute gradients w.r.t. the images and the kernel using deltas as the gradient from the next layer.
    gx, gk = res.grad_fn.backward(deltas)
    print("Backward: ")
    # Flatten image gradient with column-major indexing
    print(gx.reshape(-1, order="F"))
    print("Delta weights:")
    # Remove batch dimension, undo transpose back from (N, H, W, C) to (H, W, C, N) and flatten using column-major indexing
    print(gk.squeeze(axis=0).transpose((1, 2, 3, 0)).reshape(-1, order="F"))
