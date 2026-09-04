"""Turning a hand-drawn digit into MNIST format.

MNIST images are not left as they came: the digit is cropped to its ink, fitted into
a 20x20 square and placed in a 28x28 field so that its centre of mass sits in the
middle. Skip that and a network trained on MNIST will misread even a carefully
drawn digit.
"""

import numpy as np


def resize_bilinear(img, out_h, out_w):
    """Bilinear resize of a 2D array — pure numpy, no SciPy or PIL."""
    h, w = img.shape
    if h == out_h and w == out_w:
        return img.copy()

    ys = np.linspace(0, h - 1, out_h)
    xs = np.linspace(0, w - 1, out_w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    top = img[np.ix_(y0, x0)] * (1 - wx) + img[np.ix_(y0, x1)] * wx
    bottom = img[np.ix_(y1, x0)] * (1 - wx) + img[np.ix_(y1, x1)] * wx
    return top * (1 - wy) + bottom * wy


def box_downsample(img, factor):
    """Average over factor x factor blocks — gives the soft edges MNIST has."""
    h, w = img.shape
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def to_mnist(canvas, box=20, size=28, threshold=0.05):
    """Turns an arbitrary canvas (values 0..1, ink = 1) into a 28x28 image.

    1) crop to the bounds of the ink;
    2) fit into a box x box square, keeping the aspect ratio;
    3) place in a size x size field so the centre of mass is at the centre.
    """
    ink = canvas > threshold
    if not ink.any():
        return np.zeros((size, size))

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    cropped = canvas[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Scale so that the longer side becomes box.
    h, w = cropped.shape
    scale = box / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    digit = resize_bilinear(cropped, new_h, new_w)

    # Line the digit's centre of mass up with the centre of the 28x28 field.
    yy, xx = np.mgrid[0:new_h, 0:new_w]
    total = digit.sum()
    com_y = (yy * digit).sum() / total
    com_x = (xx * digit).sum() / total

    out = np.zeros((size, size))
    top = int(round(size / 2 - com_y))
    left = int(round(size / 2 - com_x))
    top = np.clip(top, 0, size - new_h)
    left = np.clip(left, 0, size - new_w)
    out[top:top + new_h, left:left + new_w] = digit
    return np.clip(out, 0.0, 1.0)
