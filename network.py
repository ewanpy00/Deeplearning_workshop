"""A neural network with two hidden layers, 3Blue1Brown style: 784 -> 16 -> 16 -> 10.

No frameworks — just numpy. Every formula matches the video:

    z^l = W^l a^(l-1) + b^l          (weighted sum)
    a^l = sigma(z^l)                 (neuron activation)

Backpropagation (chapters 3-4):

    delta^L = grad_a C  (*)  sigma'(z^L)                 — error of the output layer
    delta^l = (W^(l+1))^T delta^(l+1)  (*)  sigma'(z^l)  — error pushed back a layer
    dC/dW^l = delta^l (a^(l-1))^T
    dC/db^l = delta^l

(*) is element-wise multiplication.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z):
    """Squashes any number into the range (0, 1) — "how brightly the neuron fires"."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_prime(z):
    """Derivative of the sigmoid: sigma'(z) = sigma(z) * (1 - sigma(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


class Network:
    """A fully connected network of any depth (784, 16, 16, 10 by default)."""

    def __init__(self, sizes=(784, 16, 16, 10), cost="mse", seed=None):
        self.sizes = list(sizes)
        self.num_layers = len(self.sizes)
        self.cost = cost  # "mse" as in the video, "cross-entropy" learns a little faster

        rng = np.random.default_rng(seed)
        # The weights of layer l form a (neurons in layer l) x (neurons in layer l-1) matrix.
        # We divide by sqrt(inputs) to keep z from saturating the sigmoid.
        self.weights = [
            rng.standard_normal((y, x)) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]

    # ------------------------------------------------------------- forward pass

    def feedforward(self, a):
        """a is a 784 x N matrix. Returns the activations of the output layer (10 x N)."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def forward_all(self, a):
        """Same, but keeps every intermediate z and a — needed for backprop and plots."""
        activations = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return zs, activations

    def predict(self, x):
        """The index of the brightest output neuron is the network's answer."""
        return np.argmax(self.feedforward(x), axis=0)

    # ------------------------------------------------------------ backpropagation

    def backprop(self, x, y):
        """Gradients of the cost for every weight and bias over a mini-batch (x, y).

        x: 784 x m, y: 10 x m. Returns dW and db lists averaged over the batch.
        """
        m = x.shape[1]
        zs, activations = self.forward_all(x)

        # Error of the last layer.
        if self.cost == "cross-entropy":
            # With cross-entropy and a sigmoid the sigma'(z) factor cancels out neatly.
            delta = activations[-1] - y
        else:
            # Quadratic cost C = 1/2 * sum (a - y)^2 — exactly as in the video.
            delta = (activations[-1] - y) * sigmoid_prime(zs[-1])

        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        grad_w[-1] = delta @ activations[-2].T / m
        grad_b[-1] = delta.sum(axis=1, keepdims=True) / m

        # Walk backwards through the layers: -2, -3, ... dragging the error through W^T.
        for l in range(2, self.num_layers):
            delta = (self.weights[-l + 1].T @ delta) * sigmoid_prime(zs[-l])
            grad_w[-l] = delta @ activations[-l - 1].T / m
            grad_b[-l] = delta.sum(axis=1, keepdims=True) / m

        return grad_w, grad_b

    # ------------------------------------------------------------------ training

    def sgd(self, x_train, y_train, epochs=30, batch_size=10, eta=3.0,
            test_data=None, seed=None, verbose=True):
        """Stochastic gradient descent: walking down the hilly landscape of the cost.

        Every epoch we shuffle the data, cut it into mini-batches and take one step
        on each:  W <- W - eta * dC/dW.
        """
        rng = np.random.default_rng(seed)
        n = x_train.shape[1]
        history = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                grad_w, grad_b = self.backprop(x_train[:, idx], y_train[:, idx])
                for i in range(len(self.weights)):
                    self.weights[i] -= eta * grad_w[i]
                    self.biases[i] -= eta * grad_b[i]

            if test_data is not None:
                acc = self.accuracy(*test_data)
                history.append(acc)
                if verbose:
                    print(f"Epoch {epoch:2d}/{epochs}: test accuracy {acc * 100:5.2f}%")
            elif verbose:
                print(f"Epoch {epoch:2d}/{epochs} done")

        return history

    # ------------------------------------------------------------------- metrics

    def accuracy(self, x, y_digits):
        return float(np.mean(self.predict(x) == y_digits))

    def loss(self, x, y):
        a = self.feedforward(x)
        if self.cost == "cross-entropy":
            eps = 1e-12
            return float(np.mean(-np.sum(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps), axis=0)))
        return float(np.mean(0.5 * np.sum((a - y) ** 2, axis=0)))

    # ----------------------------------------------------------- saving / loading

    def save(self, path):
        np.savez(
            path,
            sizes=np.array(self.sizes),
            cost=np.array(self.cost),
            **{f"w{i}": w for i, w in enumerate(self.weights)},
            **{f"b{i}": b for i, b in enumerate(self.biases)},
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=False)
        net = cls(sizes=tuple(data["sizes"].tolist()), cost=str(data["cost"]))
        net.weights = [data[f"w{i}"] for i in range(len(net.weights))]
        net.biases = [data[f"b{i}"] for i in range(len(net.biases))]
        return net
