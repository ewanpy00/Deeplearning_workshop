"""Interactive demo: draw a digit with the mouse and watch the neurons light up.

    python draw.py

Controls:
    left mouse button  — draw
    right button       — erase
    c                  — clear the canvas
    n                  — drop in a random MNIST digit
    q                  — quit
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import mnist
from network import Network
from preprocess import box_downsample, resize_bilinear, to_mnist

SCALE = 10                      # canvas is 280x280 = 28*10, then averaged in blocks
CANVAS = 28 * SCALE
BRUSH_RADIUS = 13.0             # brush radius in canvas pixels


class DrawingDemo:
    def __init__(self, model_path="model.npz"):
        self.net = Network.load(model_path)
        self.canvas = np.zeros((CANVAS, CANVAS))
        self.drawing = None     # None | "ink" | "erase"
        self.mnist_test = None  # loaded lazily, only if "n" is pressed

        # Precompute the coordinate grid so the brush stays fast.
        self.yy, self.xx = np.mgrid[0:CANVAS, 0:CANVAS]

        self._build_figure()
        self._refresh()

    # -------------------------------------------------------------- interface

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 7))
        self.fig.canvas.manager.set_window_title("Neural network 784-16-16-10")
        gs = GridSpec(3, 3, width_ratios=[1.25, 0.75, 1.1], height_ratios=[1, 1, 1],
                      hspace=0.45, wspace=0.3, figure=self.fig)

        # The drawing canvas.
        self.ax_canvas = self.fig.add_subplot(gs[:, 0])
        self.im_canvas = self.ax_canvas.imshow(self.canvas, cmap="gray_r", vmin=0, vmax=1)
        self.ax_canvas.set_title("Draw a digit\n(left button draws, right erases, c clears)")
        self.ax_canvas.set_xticks([])
        self.ax_canvas.set_yticks([])

        # What the network actually sees — 784 input neurons.
        self.ax_input = self.fig.add_subplot(gs[0, 1])
        self.im_input = self.ax_input.imshow(np.zeros((28, 28)), cmap="gray_r", vmin=0, vmax=1)
        self.ax_input.set_title("Network input: 28×28 = 784", fontsize=9)
        self.ax_input.set_xticks([])
        self.ax_input.set_yticks([])

        # Hidden layers — 16 neurons each, shown as a 4x4 grid.
        self.ax_h1 = self.fig.add_subplot(gs[1, 1])
        self.im_h1 = self.ax_h1.imshow(np.zeros((4, 4)), cmap="viridis", vmin=0, vmax=1)
        self.ax_h1.set_title("Hidden layer 1 (16)", fontsize=9)
        self.ax_h1.set_xticks([])
        self.ax_h1.set_yticks([])

        self.ax_h2 = self.fig.add_subplot(gs[2, 1])
        self.im_h2 = self.ax_h2.imshow(np.zeros((4, 4)), cmap="viridis", vmin=0, vmax=1)
        self.ax_h2.set_title("Hidden layer 2 (16)", fontsize=9)
        self.ax_h2.set_xticks([])
        self.ax_h2.set_yticks([])

        # The output layer — 10 neurons.
        self.ax_out = self.fig.add_subplot(gs[:, 2])
        self.bars = self.ax_out.barh(np.arange(10), np.zeros(10), color="#4c72b0")
        self.ax_out.set_yticks(np.arange(10))
        self.ax_out.set_yticklabels([str(d) for d in range(10)], fontsize=13)
        self.ax_out.invert_yaxis()
        self.ax_out.set_xlim(0, 1)
        self.ax_out.set_xlabel("neuron activation")
        self.ax_out.set_title("Output layer", fontsize=11)
        self.value_labels = [
            self.ax_out.text(0.02, d, "", va="center", fontsize=9, color="#333")
            for d in range(10)
        ]
        self.verdict = self.fig.text(0.5, 0.965, "", ha="center", fontsize=15, weight="bold")

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # ----------------------------------------------------------------- events

    def on_press(self, event):
        if event.inaxes is not self.ax_canvas:
            return
        self.drawing = "erase" if event.button == 3 else "ink"
        self.paint(event.xdata, event.ydata)

    def on_release(self, event):
        self.drawing = None

    def on_motion(self, event):
        if self.drawing and event.inaxes is self.ax_canvas:
            self.paint(event.xdata, event.ydata)

    def on_key(self, event):
        if event.key == "c":
            self.canvas[:] = 0.0
            self._refresh()
        elif event.key == "n":
            self.load_random_digit()
        elif event.key == "q":
            plt.close(self.fig)

    # ---------------------------------------------------------------- drawing

    def paint(self, x, y):
        """A soft round brush: dense ink at the centre, fading out towards the edge."""
        if x is None or y is None:
            return
        dist = np.hypot(self.xx - x, self.yy - y)
        stamp = np.clip(1.0 - (dist - BRUSH_RADIUS * 0.55) / (BRUSH_RADIUS * 0.45), 0.0, 1.0)
        if self.drawing == "erase":
            self.canvas = np.clip(self.canvas - stamp * 1.5, 0.0, 1.0)
        else:
            self.canvas = np.maximum(self.canvas, stamp)
        self._refresh()

    def load_random_digit(self):
        """Drop in a real MNIST image — handy for comparing against a hand drawing."""
        if self.mnist_test is None:
            *_, x_test, y_test = mnist.load()
            self.mnist_test = (x_test, y_test)
        x_test, _ = self.mnist_test
        i = np.random.randint(x_test.shape[1])
        self.canvas = resize_bilinear(x_test[:, i].reshape(28, 28), CANVAS, CANVAS)
        self._refresh()

    # ------------------------------------------------- running it through the net

    def _refresh(self):
        if self.canvas.max() > 0.05:
            digit_img = to_mnist(self.canvas)
        else:
            digit_img = box_downsample(self.canvas, SCALE)

        _, activations = self.net.forward_all(digit_img.reshape(-1, 1))
        h1, h2 = activations[1].ravel(), activations[2].ravel()
        out = activations[-1].ravel()

        self.im_canvas.set_data(self.canvas)
        self.im_input.set_data(digit_img)
        self.im_h1.set_data(h1.reshape(4, 4))
        self.im_h2.set_data(h2.reshape(4, 4))

        guess = int(np.argmax(out))
        for d, (rect, label) in enumerate(zip(self.bars, self.value_labels)):
            rect.set_width(out[d])
            rect.set_color("#d1495b" if d == guess else "#4c72b0")
            label.set_text(f"{out[d]:.3f}")
            # keep the label just past the end of the bar so it does not cover it
            label.set_x(min(out[d] + 0.02, 0.86))

        if digit_img.sum() < 1e-6:
            self.verdict.set_text("Canvas is empty — draw a digit")
        else:
            self.verdict.set_text(f"The network sees: {guess}   ({out[guess] * 100:.1f}% sure)")

        self.fig.canvas.draw_idle()


def main():
    p = argparse.ArgumentParser(description="Interactive recognition of a hand-drawn digit")
    p.add_argument("--model", default="model.npz")
    args = p.parse_args()

    DrawingDemo(args.model)
    plt.show()


if __name__ == "__main__":
    main()
