"""Нейросеть с двумя скрытыми слоями «как у 3Blue1Brown»: 784 -> 16 -> 16 -> 10.

Никаких фреймворков — только numpy. Все формулы соответствуют видео:

    z^l = W^l a^(l-1) + b^l          (взвешенная сумма)
    a^l = sigma(z^l)                 (активация нейрона)

Обратное распространение (chapters 3-4):

    delta^L = grad_a C  (*)  sigma'(z^L)              — ошибка выходного слоя
    delta^l = (W^(l+1))^T delta^(l+1)  (*)  sigma'(z^l)  — ошибка проталкивается назад
    dC/dW^l = delta^l (a^(l-1))^T
    dC/db^l = delta^l

(*) — поэлементное умножение.
"""

from __future__ import annotations

import numpy as np


def sigmoid(z):
    """Сплющивает любое число в диапазон (0, 1) — «насколько сильно горит нейрон»."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_prime(z):
    """Производная сигмоиды: sigma'(z) = sigma(z) * (1 - sigma(z))."""
    s = sigmoid(z)
    return s * (1.0 - s)


class Network:
    """Полносвязная сеть произвольной глубины (по умолчанию — [784, 16, 16, 10])."""

    def __init__(self, sizes=(784, 16, 16, 10), cost="mse", seed=None):
        self.sizes = list(sizes)
        self.num_layers = len(self.sizes)
        self.cost = cost  # "mse" — как в видео, "cross-entropy" — учится быстрее

        rng = np.random.default_rng(seed)
        # Веса слоя l — матрица (нейронов в слое l) x (нейронов в слое l-1).
        # Делим на sqrt(входов), чтобы z не улетали в насыщение сигмоиды.
        self.weights = [
            rng.standard_normal((y, x)) / np.sqrt(x)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]
        self.biases = [np.zeros((y, 1)) for y in self.sizes[1:]]

    # ---------------------------------------------------------------- прямой ход

    def feedforward(self, a):
        """a — матрица 784 x N. Возвращает активации выходного слоя (10 x N)."""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def forward_all(self, a):
        """То же, но сохраняет все промежуточные z и a — нужно для backprop и визуализации."""
        activations = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
        return zs, activations

    def predict(self, x):
        """Номер самого «яркого» выходного нейрона — это и есть ответ сети."""
        return np.argmax(self.feedforward(x), axis=0)

    # ---------------------------------------------------- обратное распространение

    def backprop(self, x, y):
        """Градиенты стоимости по всем весам и смещениям для мини-батча (x, y).

        x: 784 x m, y: 10 x m. Возвращает списки dW и db, усреднённые по батчу.
        """
        m = x.shape[1]
        zs, activations = self.forward_all(x)

        # Ошибка последнего слоя.
        if self.cost == "cross-entropy":
            # Для кросс-энтропии с сигмоидой множитель sigma'(z) красиво сокращается.
            delta = activations[-1] - y
        else:
            # Квадратичная стоимость C = 1/2 * sum (a - y)^2 — ровно как в видео.
            delta = (activations[-1] - y) * sigmoid_prime(zs[-1])

        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        grad_w[-1] = delta @ activations[-2].T / m
        grad_b[-1] = delta.sum(axis=1, keepdims=True) / m

        # Идём назад по слоям: -2, -3, ... Каждый раз тянем ошибку через W^T.
        for l in range(2, self.num_layers):
            delta = (self.weights[-l + 1].T @ delta) * sigmoid_prime(zs[-l])
            grad_w[-l] = delta @ activations[-l - 1].T / m
            grad_b[-l] = delta.sum(axis=1, keepdims=True) / m

        return grad_w, grad_b

    # ------------------------------------------------------------------- обучение

    def sgd(self, x_train, y_train, epochs=30, batch_size=10, eta=3.0,
            test_data=None, seed=None, verbose=True):
        """Стохастический градиентный спуск: спускаемся по «холмистому ландшафту» стоимости.

        Каждую эпоху перемешиваем выборку, режем на мини-батчи и на каждом делаем
        один шаг:  W <- W - eta * dC/dW.
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
                    print(f"Эпоха {epoch:2d}/{epochs}: точность на тесте {acc * 100:5.2f}%")
            elif verbose:
                print(f"Эпоха {epoch:2d}/{epochs} завершена")

        return history

    # ------------------------------------------------------------------- метрики

    def accuracy(self, x, y_digits):
        return float(np.mean(self.predict(x) == y_digits))

    def loss(self, x, y):
        a = self.feedforward(x)
        if self.cost == "cross-entropy":
            eps = 1e-12
            return float(np.mean(-np.sum(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps), axis=0)))
        return float(np.mean(0.5 * np.sum((a - y) ** 2, axis=0)))

    # --------------------------------------------------------- сохранение/загрузка

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
