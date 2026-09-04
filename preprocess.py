"""Приведение нарисованной от руки цифры к формату MNIST.

Картинки MNIST сделаны не «как получилось»: цифру обрезают по краям чернил,
вписывают в квадрат 20x20 и кладут в поле 28x28 так, чтобы центр масс оказался
в середине. Если этого не сделать, сеть, обученная на MNIST, будет ошибаться
даже на аккуратно нарисованной цифре.
"""

import numpy as np


def resize_bilinear(img, out_h, out_w):
    """Билинейное масштабирование двумерного массива — чистый numpy, без SciPy/PIL."""
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
    """Усреднение блоками factor x factor — даёт мягкие края, как у MNIST."""
    h, w = img.shape
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def to_mnist(canvas, box=20, size=28, threshold=0.05):
    """Из произвольного холста (значения 0..1, чернила = 1) делает картинку 28x28.

    1) обрезаем по границам чернил;
    2) вписываем в квадрат box x box, сохраняя пропорции;
    3) кладём в поле size x size так, чтобы центр масс был в центре.
    """
    ink = canvas > threshold
    if not ink.any():
        return np.zeros((size, size))

    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    cropped = canvas[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Масштабируем так, чтобы длинная сторона стала равна box.
    h, w = cropped.shape
    scale = box / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    digit = resize_bilinear(cropped, new_h, new_w)

    # Центр масс цифры совмещаем с центром поля 28x28.
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
