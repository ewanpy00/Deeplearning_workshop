"""Trains a 784 -> 16 -> 16 -> 10 network on MNIST.

Examples:
    python train.py                              # default settings
    python train.py --epochs 50 --cost cross-entropy --eta 0.5
    python train.py --hidden 30 30 --out big.npz
"""

import argparse
import time

import mnist
from network import Network


def main():
    p = argparse.ArgumentParser(description="Train a neural network on MNIST")
    p.add_argument("--hidden", type=int, nargs="+", default=[16, 16],
                   help="sizes of the hidden layers (default 16 16, as in the video)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=10, help="mini-batch size")
    p.add_argument("--eta", type=float, default=3.0, help="learning rate")
    p.add_argument("--cost", choices=["mse", "cross-entropy"], default="mse",
                   help="mse is the quadratic cost from the video")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="model.npz", help="where to save the weights")
    args = p.parse_args()

    print("Loading MNIST ...")
    x_train, y_train, digits_train, x_test, y_test = mnist.load()
    print(f"Training set: {x_train.shape[1]} images, test set: {x_test.shape[1]}\n")

    sizes = [784, *args.hidden, 10]
    net = Network(sizes=sizes, cost=args.cost, seed=args.seed)
    print("Architecture:", " -> ".join(map(str, sizes)))
    print(f"Parameters: {sum(w.size for w in net.weights) + sum(b.size for b in net.biases):,}")
    print(f"Cost: {args.cost}, eta={args.eta}, batch={args.batch}\n")

    start = time.time()
    net.sgd(x_train, y_train, epochs=args.epochs, batch_size=args.batch,
            eta=args.eta, test_data=(x_test, y_test), seed=args.seed)
    elapsed = time.time() - start

    train_acc = net.accuracy(x_train, digits_train)
    test_acc = net.accuracy(x_test, y_test)
    print(f"\nTraining took {elapsed:.1f} s")
    print(f"Accuracy on the training set: {train_acc * 100:.2f}%")
    print(f"Accuracy on the test set:     {test_acc * 100:.2f}%")

    net.save(args.out)
    print(f"Weights saved to {args.out}")


if __name__ == "__main__":
    main()
