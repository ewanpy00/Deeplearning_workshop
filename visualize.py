"""Показывает картинку и «свечение» всех нейронов слой за слоем — как схема из видео.

    python visualize.py                 # случайная картинка из теста
    python visualize.py --index 42
    python visualize.py --count 5
    python visualize.py --errors        # только ошибки сети (самое интересное)
"""

import argparse

import numpy as np

import mnist
from network import Network

SHADES = " .:-=+*#%@"        # градации яркости для картинки 28x28
BAR_WIDTH = 24               # длина полоски активации

RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"
GREEN, RED, CYAN, YELLOW = "\033[32m", "\033[31m", "\033[36m", "\033[33m"


def image_lines(column):
    """28 строк ASCII-графики из вектора-столбца длиной 784."""
    img = column.reshape(28, 28)
    lines = []
    for row in img:
        # два символа на пиксель, иначе картинка выглядит сплющенной
        lines.append("".join(SHADES[min(int(v * len(SHADES)), len(SHADES) - 1)] * 2 for v in row))
    return lines


def bar(value, color=""):
    filled = int(round(value * BAR_WIDTH))
    return f"{color}{'█' * filled}{DIM}{'·' * (BAR_WIDTH - filled)}{RESET}"


def show(net, x, true_digit=None):
    _, activations = net.forward_all(x.reshape(-1, 1))
    guess = int(np.argmax(activations[-1]))

    print()
    for line in image_lines(x):
        print("  " + line)

    # Скрытые слои: 16 нейронов в строку, компактно.
    for layer_i, a in enumerate(activations[1:-1], start=1):
        vals = a.ravel()
        print(f"\n  {BOLD}Скрытый слой {layer_i}{RESET} ({vals.size} нейронов):")
        for start in range(0, vals.size, 8):
            chunk = vals[start:start + 8]
            cells = "  ".join(
                f"{DIM}{start + k:2d}{RESET} {CYAN}{'█' * int(round(v * 6)):<6}{RESET}{v:4.2f}"
                for k, v in enumerate(chunk)
            )
            print("   " + cells)

    print(f"\n  {BOLD}Выходной слой{RESET}:")
    out = activations[-1].ravel()
    for digit, value in enumerate(out):
        marker = ""
        color = ""
        if digit == guess:
            color = GREEN if (true_digit is None or digit == true_digit) else RED
            marker = "  <- ответ сети"
        if true_digit is not None and digit == true_digit and digit != guess:
            color, marker = YELLOW, "  <- верный ответ"
        print(f"   {BOLD}{digit}{RESET} {bar(value, color)} {value:5.3f}{color}{marker}{RESET}")

    if true_digit is None:
        print(f"\n  Сеть считает, что это {BOLD}{guess}{RESET} (уверенность {out[guess] * 100:.1f}%)")
    else:
        ok = guess == true_digit
        verdict = f"{GREEN}верно{RESET}" if ok else f"{RED}ошибка, на самом деле {true_digit}{RESET}"
        print(f"\n  Ответ сети: {BOLD}{guess}{RESET} ({out[guess] * 100:.1f}%) — {verdict}")
    print()


def main():
    p = argparse.ArgumentParser(description="Визуализация активаций сети")
    p.add_argument("--model", default="model.npz")
    p.add_argument("--index", type=int, help="номер картинки в тестовой выборке")
    p.add_argument("--count", type=int, default=1, help="сколько картинок показать")
    p.add_argument("--errors", action="store_true", help="показывать только ошибки сети")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    net = Network.load(args.model)
    *_, x_test, y_test = mnist.load()

    if args.index is not None:
        indices = [args.index]
    else:
        rng = np.random.default_rng(args.seed)
        pool = np.arange(x_test.shape[1])
        if args.errors:
            pool = pool[net.predict(x_test) != y_test]
            print(f"Сеть ошибается на {pool.size} из {y_test.size} тестовых картинок "
                  f"({pool.size / y_test.size * 100:.2f}%)")
        indices = rng.choice(pool, size=min(args.count, pool.size), replace=False)

    for i in indices:
        print(f"{DIM}--- тестовая картинка #{i} ---{RESET}")
        show(net, x_test[:, i], int(y_test[i]))


if __name__ == "__main__":
    main()
