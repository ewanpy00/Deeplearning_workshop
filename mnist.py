"""Загрузка MNIST без сторонних библиотек: скачиваем idx-файлы и читаем их сами.

Формат IDX (описан на сайте Яна Лекуна):
    [0:4]  магическое число (0x00000803 для картинок, 0x00000801 для меток)
    [4:8]  количество элементов
    далее для картинок: [8:12] строк, [12:16] столбцов, затем сами байты 0..255
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
    """Скачиваем файл один раз и кэшируем в ./data."""
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / name
    if path.exists():
        return path

    last_error = None
    for mirror in MIRRORS:
        try:
            print(f"Скачиваю {name} ...")
            urllib.request.urlretrieve(mirror + name, path)
            return path
        except Exception as exc:  # пробуем следующее зеркало
            last_error = exc
    raise RuntimeError(f"Не удалось скачать {name}: {last_error}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic == 0x803:  # картинки
            rows, cols = struct.unpack(">II", f.read(8))
            buf = f.read(rows * cols * count)
            return np.frombuffer(buf, dtype=np.uint8).reshape(count, rows * cols)
        if magic == 0x801:  # метки
            buf = f.read(count)
            return np.frombuffer(buf, dtype=np.uint8)
        raise ValueError(f"Неизвестное магическое число {magic:#x} в {path}")


def one_hot(labels: np.ndarray) -> np.ndarray:
    """Цифра 3 -> вектор-столбец с единицей на позиции 3 (это и есть «правильный ответ» сети)."""
    y = np.zeros((10, labels.size))
    y[labels, np.arange(labels.size)] = 1.0
    return y


def load():
    """Возвращает (X_train, Y_train, y_train, X_test, y_test).

    X — матрица 784 x N: один столбец = одна картинка, значения приведены к [0, 1].
    Y — матрица 10 x N с one-hot ответами, y — просто цифры 0..9.
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
