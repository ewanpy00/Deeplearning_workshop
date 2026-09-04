"""Обучение сети 784 -> 16 -> 16 -> 10 на MNIST.

Примеры:
    python train.py                              # настройки по умолчанию
    python train.py --epochs 50 --cost cross-entropy --eta 0.5
    python train.py --hidden 30 30 --out big.npz
"""

import argparse
import time

import mnist
from network import Network


def main():
    p = argparse.ArgumentParser(description="Обучение нейросети на MNIST")
    p.add_argument("--hidden", type=int, nargs="+", default=[16, 16],
                   help="размеры скрытых слоёв (по умолчанию 16 16 — как в видео)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=10, help="размер мини-батча")
    p.add_argument("--eta", type=float, default=3.0, help="скорость обучения")
    p.add_argument("--cost", choices=["mse", "cross-entropy"], default="mse",
                   help="mse — квадратичная стоимость из видео")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="model.npz", help="куда сохранить веса")
    args = p.parse_args()

    print("Загружаю MNIST ...")
    x_train, y_train, digits_train, x_test, y_test = mnist.load()
    print(f"Обучающая выборка: {x_train.shape[1]} картинок, тестовая: {x_test.shape[1]}\n")

    sizes = [784, *args.hidden, 10]
    net = Network(sizes=sizes, cost=args.cost, seed=args.seed)
    print("Архитектура:", " -> ".join(map(str, sizes)))
    print(f"Параметров: {sum(w.size for w in net.weights) + sum(b.size for b in net.biases):,}")
    print(f"Стоимость: {args.cost}, eta={args.eta}, батч={args.batch}\n")

    start = time.time()
    net.sgd(x_train, y_train, epochs=args.epochs, batch_size=args.batch,
            eta=args.eta, test_data=(x_test, y_test), seed=args.seed)
    elapsed = time.time() - start

    train_acc = net.accuracy(x_train, digits_train)
    test_acc = net.accuracy(x_test, y_test)
    print(f"\nОбучение заняло {elapsed:.1f} с")
    print(f"Точность на обучающей выборке: {train_acc * 100:.2f}%")
    print(f"Точность на тестовой выборке:  {test_acc * 100:.2f}%")

    net.save(args.out)
    print(f"Веса сохранены в {args.out}")


if __name__ == "__main__":
    main()
