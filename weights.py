"""Looking at what each neuron of the first hidden layer is "looking for".

There is a key moment in the 3Blue1Brown video: we hope the 16 neurons of the first
layer will learn strokes, loops and arcs. We draw each neuron's weights as a 28x28
image (red for positive weights, blue for negative) and see that in practice they
come out as fairly noisy blobs. The network solves the task, but not "the human way".

    python weights.py            # opens a window
    python weights.py --save weights.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from network import Network


def main():
    p = argparse.ArgumentParser(description="Visualise the first-layer weights")
    p.add_argument("--model", default="model.npz")
    p.add_argument("--layer", type=int, default=0, help="layer index (0 is the first hidden layer)")
    p.add_argument("--save", help="save to a file instead of opening a window")
    args = p.parse_args()

    net = Network.load(args.model)
    w = net.weights[args.layer]

    if args.layer != 0:
        raise SystemExit("Only the first layer's weights make sense drawn as a 28x28 image")

    n = w.shape[0]
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    limit = np.abs(w).max()
    for i in range(n):
        ax = axes[i]
        ax.imshow(w[i].reshape(28, 28), cmap="bwr", vmin=-limit, vmax=limit)
        ax.set_title(f"neuron {i}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Weights of the first hidden layer's neurons\n"
                 "red is a positive weight, blue a negative one", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=3.2)

    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
