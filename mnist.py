"""Loading MNIST without third-party libraries: we download the idx files and parse them ourselves.

The IDX format (described on Yann LeCun's site):
    [0:4]  magic number (0x00000803 for images, 0x00000801 for labels)
    [4:8]  number of items
    then for images: [8:12] rows, [12:16] columns, followed by the raw bytes 0..255
"""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(name: str) -> Path:
    """Download the file once and cache it in ./data."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / name
    if path.exists():
        return path

    last_error = None
    for mirror in MIRRORS:
        try:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(mirror + name, path)
            return path
        except Exception as exc:  # try the next mirror
            last_error = exc
    raise RuntimeError(f"Could not download {name}: {last_error}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic == 0x803:  # images
            rows, cols = struct.unpack(">II", f.read(8))
            buf = f.read(rows * cols * count)
            return np.frombuffer(buf, dtype=np.uint8).reshape(count, rows * cols)
        if magic == 0x801:  # labels
            buf = f.read(count)
            return np.frombuffer(buf, dtype=np.uint8)
        raise ValueError(f"Unknown magic number {magic:#x} in {path}")


def one_hot(labels: np.ndarray) -> np.ndarray:
    """Digit 3 -> a column vector with a 1 at position 3 (the network's target answer)."""
    y = np.zeros((10, labels.size))
    y[labels, np.arange(labels.size)] = 1.0
    return y


def load():
    """Returns (X_train, Y_train, y_train, X_test, y_test).

    X is a 784 x N matrix: one column per image, values scaled to [0, 1].
    Y is a 10 x N matrix of one-hot answers, y is just the digits 0..9.
    """
    raw = {k: _read_idx(_download(v)) for k, v in FILES.items()}

    x_train = raw["train_images"].T.astype(np.float64) / 255.0
    x_test = raw["test_images"].T.astype(np.float64) / 255.0
    y_train = raw["train_labels"].astype(np.int64)
    y_test = raw["test_labels"].astype(np.int64)

    return x_train, one_hot(y_train), y_train, x_test, y_test


if __name__ == "__main__":
    x_train, _, y_train, x_test, y_test = load()
    print("train:", x_train.shape, y_train.shape)
    print("test: ", x_test.shape, y_test.shape)
