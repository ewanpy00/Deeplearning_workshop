"""Смотрим, что «ищет» каждый нейрон первого скрытого слоя.

В видео 3Blue1Brown есть ключевой момент: мы надеемся, что 16 нейронов первого
слоя выучат палочки, петельки и дуги. Рисуем веса каждого нейрона как картинку
28x28 (красное — положительный вес, синее — отрицательный) и видим, что на деле
получаются довольно шумные пятна. Сеть решает задачу, но не «по-человечески».

    python weights.py            # откроет окно
    python weights.py --save weights.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from network import Network


def main():
    p = argparse.ArgumentParser(description="Визуализация весов первого слоя")
    p.add_argument("--model", default="model.npz")
    p.add_argument("--layer", type=int, default=0, help="номер слоя (0 — первый скрытый)")
    p.add_argument("--save", help="сохранить картинку в файл вместо показа окна")
    args = p.parse_args()

    net = Network.load(args.model)
    w = net.weights[args.layer]

    if args.layer != 0:
        raise SystemExit("Как картинку 28x28 осмысленно рисовать только веса первого слоя")

    n = w.shape[0]
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    limit = np.abs(w).max()
    for i in range(n):
        ax = axes[i]
        ax.imshow(w[i].reshape(28, 28), cmap="bwr", vmin=-limit, vmax=limit)
        ax.set_title(f"нейрон {i}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Веса нейронов первого скрытого слоя\n"
                 "красное — положительный вес, синее — отрицательный", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=3.2)

    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"Сохранено в {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
