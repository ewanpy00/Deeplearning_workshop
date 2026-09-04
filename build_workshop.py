"""Генератор workshop.ipynb и workshop_solution.ipynb."""
import json, sys

SETUP = r'''#@title Запусти эту ячейку и иди дальше (скачивание MNIST + проверки) { display-mode: "form" }
import gzip, struct, urllib.request, os, base64
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- данные MNIST
MIRRORS = ["https://storage.googleapis.com/cvdf-datasets/mnist/",
           "https://ossci-datasets.s3.amazonaws.com/mnist/"]
FILES = {"train_images": "train-images-idx3-ubyte.gz", "train_labels": "train-labels-idx1-ubyte.gz",
         "test_images": "t10k-images-idx3-ubyte.gz",  "test_labels": "t10k-labels-idx1-ubyte.gz"}

def _get(name):
    if not os.path.exists(name):
        for m in MIRRORS:
            try:
                urllib.request.urlretrieve(m + name, name); break
            except Exception: pass
    with gzip.open(name, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic == 0x803:
            rows, cols = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(rows*cols*count), np.uint8).reshape(count, rows*cols)
        return np.frombuffer(f.read(count), np.uint8)

def one_hot(labels):
    y = np.zeros((10, labels.size)); y[labels, np.arange(labels.size)] = 1.0; return y

print("Скачиваю MNIST ...")
_raw = {k: _get(v) for k, v in FILES.items()}
X_train = _raw["train_images"].T.astype(np.float64) / 255.0     # 784 x 60000
d_train = _raw["train_labels"].astype(np.int64)                 # цифры 0..9
Y_train = one_hot(d_train)                                      # 10 x 60000
X_test  = _raw["test_images"].T.astype(np.float64) / 255.0
y_test  = _raw["test_labels"].astype(np.int64)
print(f"Готово: {X_train.shape[1]} картинок для обучения, {X_test.shape[1]} для проверки")

# ------------------------------------------------- функция стоимости (дана готовой)
def cost(net, x, y):
    """C = среднее по картинкам от 1/2 * сумма (a - y)^2 — та самая из видео."""
    _, activations = forward(net, x)
    return float(np.mean(0.5 * np.sum((activations[-1] - y) ** 2, axis=0)))

# ------------------------------- приведение рисунка к формату MNIST (дано готовым)
def resize_bilinear(img, oh, ow):
    h, w = img.shape
    ys, xs = np.linspace(0, h-1, oh), np.linspace(0, w-1, ow)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0+1, h-1), np.minimum(x0+1, w-1)
    wy, wx = (ys-y0)[:, None], (xs-x0)[None, :]
    top = img[np.ix_(y0, x0)]*(1-wx) + img[np.ix_(y0, x1)]*wx
    bot = img[np.ix_(y1, x0)]*(1-wx) + img[np.ix_(y1, x1)]*wx
    return top*(1-wy) + bot*wy

def to_mnist(canvas, box=20, size=28):
    """Обрезает по чернилам, вписывает в 20x20 и центрирует по центру масс."""
    ink = canvas > 0.05
    if not ink.any(): return np.zeros((size, size))
    r, c = np.where(ink.any(1))[0], np.where(ink.any(0))[0]
    crop = canvas[r[0]:r[-1]+1, c[0]:c[-1]+1]
    h, w = crop.shape
    s = box / max(h, w)
    nh, nw = max(1, round(h*s)), max(1, round(w*s))
    digit = resize_bilinear(crop, nh, nw)
    yy, xx = np.mgrid[0:nh, 0:nw]
    cy, cx = (yy*digit).sum()/digit.sum(), (xx*digit).sum()/digit.sum()
    out = np.zeros((size, size))
    top  = int(np.clip(round(size/2 - cy), 0, size-nh))
    left = int(np.clip(round(size/2 - cx), 0, size-nw))
    out[top:top+nh, left:left+nw] = digit
    return np.clip(out, 0, 1)

# ------------------------------------------------------------------ показ ответа
def show_prediction(net, img, true_digit=None):
    _, acts = forward(net, img.reshape(-1, 1))
    out = acts[-1].ravel(); guess = int(np.argmax(out))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 2]})
    a1.imshow(img.reshape(28, 28), cmap="gray_r"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("вход сети")
    colors = ["#d1495b" if d == guess else "#4c72b0" for d in range(10)]
    a2.barh(np.arange(10), out, color=colors); a2.invert_yaxis()
    a2.set_yticks(range(10)); a2.set_xlim(0, 1); a2.set_title("выходной слой")
    t = f"сеть видит: {guess} ({out[guess]*100:.1f}%)"
    if true_digit is not None:
        t += "  — верно" if guess == true_digit else f"  — ошибка, это {true_digit}"
    fig.suptitle(t, fontsize=13); plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------- проверки
# Ниже нет ни одного готового ответа: проверки гоняют ТВОЙ код и смотрят,
# согласуется ли он сам с собой и с математикой. Подглядывать нечего :)
_OK, _NO = "✅", "❌"

def _need(name):
    f = globals().get(name)
    if f is None: raise AssertionError(f"функция {name}() ещё не определена — запусти ячейку с ней")
    try: return f
    except Exception: return f

def _check1():
    s = _need("sigmoid")
    assert abs(s(0.0) - 0.5) < 1e-12, "sigmoid(0) должна быть ровно 0.5"
    assert abs(s(2.0) - 0.8807970779778823) < 1e-9, "sigmoid(2) должна быть 0.8807970779778823"
    z = np.array([[-1.0, 0.0], [1.0, 2.0]])
    r = np.asarray(s(z))
    assert r.shape == z.shape, "функция должна работать с массивами поэлементно (без циклов)"
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        big = np.asarray(s(np.array([-800.0, 800.0])))
    assert np.isfinite(big).all(), "при больших по модулю z получается nan — обрежь z через np.clip(z, -500, 500)"
    assert abs(big[0]) < 1e-9 and abs(big[1] - 1) < 1e-9, "при z = -800 ожидается ~0, при z = 800 ~1"
    if any(issubclass(c.category, RuntimeWarning) for c in caught):
        return ("значения верные, но numpy ругается на переполнение в np.exp. "
                "Оберни аргумент: np.exp(-np.clip(z, -500, 500))")
    return "sigmoid работает и не взрывается на больших z"

def _check2():
    s, sp = _need("sigmoid"), _need("sigmoid_prime")
    z = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    num = (np.asarray(s(z + 1e-5)) - np.asarray(s(z - 1e-5))) / 2e-5
    err = np.max(np.abs(np.asarray(sp(z)) - num))
    assert err < 1e-6, f"не совпадает с численной производной твоей же sigmoid (расхождение {err:.2e})"
    return "производная совпадает с численной — формула верна"

def _check3():
    init = _need("init_network")
    net = init([784, 16, 16, 10], seed=0)
    assert isinstance(net, dict) and {"sizes", "weights", "biases"} <= set(net), \
        "верни словарь с ключами 'sizes', 'weights', 'biases'"
    W, B = net["weights"], net["biases"]
    assert len(W) == 3 and len(B) == 3, f"должно быть 3 набора весов (слоёв связей), а не {len(W)}"
    for i, (shape, w) in enumerate(zip([(16, 784), (16, 16), (10, 16)], W)):
        assert w.shape == shape, f"weights[{i}] должна быть {shape}, а не {w.shape}"
    for i, (shape, b) in enumerate(zip([(16, 1), (16, 1), (10, 1)], B)):
        assert b.shape == shape, f"biases[{i}] должен быть {shape} (столбец!), а не {b.shape}"
    assert np.allclose(W[0], init([784, 16, 16, 10], seed=0)["weights"][0]), \
        "с одинаковым seed должны получаться одинаковые веса — используй np.random.default_rng(seed)"
    std = W[0].std()
    assert std < 0.25, (f"веса слишком большие (std={std:.3f}): подели на np.sqrt(числа входов), "
                        "иначе z улетят в насыщение сигмоиды и сеть будет учиться еле-еле")
    return f"сеть 784-16-16-10 создана, {sum(w.size for w in W) + sum(b.size for b in B):,} параметров"

def _check4():
    init, fwd, s = _need("init_network"), _need("forward"), _need("sigmoid")
    net = init([6, 4, 3, 2], seed=1)
    x = np.random.default_rng(5).random((6, 7))
    zs, acts = fwd(net, x)
    assert len(zs) == 3, f"zs должен содержать 3 элемента (по одному на слой связей), а не {len(zs)}"
    assert len(acts) == 4, f"activations должен содержать 4 элемента (вход + 3 слоя), а не {len(acts)}"
    assert np.allclose(acts[0], x), "activations[0] — это сам вход x"
    assert acts[-1].shape == (2, 7), f"на выходе ожидается (2, 7), получено {acts[-1].shape}"
    for l in range(3):
        z_exp = net["weights"][l] @ acts[l] + net["biases"][l]
        assert np.allclose(zs[l], z_exp), f"zs[{l}] должен быть W @ a + b (проверь порядок множителей)"
        assert np.allclose(acts[l+1], np.asarray(s(zs[l]))), f"activations[{l+1}] должна быть sigmoid(zs[{l}])"
    return "прямой ход считается правильно, размерности сходятся"

def _check5():
    init, pred, acc = _need("init_network"), _need("predict"), _need("accuracy")
    net = init([4, 3, 2], seed=2)
    x = np.random.default_rng(9).random((4, 25))
    p = np.asarray(pred(net, x))
    assert p.shape == (25,), f"predict должен вернуть массив из 25 чисел, а не {p.shape}"
    assert p.dtype.kind in "iu", "predict должен вернуть номера нейронов (целые числа) — это np.argmax"
    assert np.array_equal(p, np.argmax(forward(net, x)[1][-1], axis=0)), \
        "ответ сети — номер самого активного нейрона выходного слоя"
    a = acc(net, x, p)
    assert abs(a - 1.0) < 1e-12, "если подать собственные ответы сети как правильные, точность должна быть 1.0"
    fake = np.zeros(25, dtype=int)
    assert abs(acc(net, x, fake) - np.mean(p == fake)) < 1e-12, "accuracy — это доля совпадений, число от 0 до 1"
    return "предсказание и подсчёт точности работают"

def _check6():
    init, bp = _need("init_network"), _need("backprop")
    net = init([5, 4, 3, 2], seed=3)
    rng = np.random.default_rng(11)
    x = rng.random((5, 6)); y = one_hot(rng.integers(0, 2, 6))[:2]
    gw, gb = bp(net, x, y)
    assert len(gw) == 3 and len(gb) == 3, "верни по 3 матрицы градиентов: для каждого слоя связей"
    for i in range(3):
        assert np.asarray(gw[i]).shape == net["weights"][i].shape, \
            f"grad_w[{i}] должна совпадать по форме с weights[{i}]"
        assert np.asarray(gb[i]).shape == net["biases"][i].shape, \
            f"grad_b[{i}] должен совпадать по форме с biases[{i}] — не забудь keepdims=True"
    # численная проверка: сдвигаем один вес на +-eps и смотрим, как меняется стоимость
    eps, worst, where, ratios = 1e-6, 0.0, "", []
    for l in range(3):
        for arr, g, tag in ((net["weights"][l], gw[l], "weights"), (net["biases"][l], gb[l], "biases")):
            for _ in range(12):
                i = rng.integers(arr.shape[0]); j = rng.integers(arr.shape[1])
                old = arr[i, j]
                arr[i, j] = old + eps; cp = cost(net, x, y)
                arr[i, j] = old - eps; cm = cost(net, x, y)
                arr[i, j] = old
                num = (cp - cm) / (2 * eps)
                mine = float(np.asarray(g)[i, j])
                rel = abs(num - mine) / max(1e-9, abs(num) + abs(mine))
                if rel > worst: worst, where = rel, f"{tag}[{l}]"
                if abs(num) > 1e-7: ratios.append(mine / num)
    # частый случай: формулы верны, но все градиенты отличаются в одно и то же число раз
    ratios = np.array(ratios)
    if worst > 1e-4 and ratios.size and abs(ratios.mean() - 1) > 0.02 \
            and np.all(np.abs(ratios - ratios.mean()) < 0.02 * abs(ratios.mean())):
        raise AssertionError(
            f"формулы верны, но каждый градиент ровно в {ratios.mean():.3g} раз отличается от численного. "
            f"Дело в делении на размер батча: в этой проверке m = {x.shape[1]}")
    if worst > 0.3:
        raise AssertionError(f"градиент по {where} не похож на численный (расхождение {worst:.1%}). "
                             "Проверь формулу delta и порядок множителей в W.T @ delta")
    if worst > 1e-4:
        raise AssertionError(f"градиент по {where} почти верный, но расходится на {worst:.2%}. "
                             "Скорее всего забыл поделить на размер батча m")
    return f"градиенты совпали с численными (расхождение {worst:.1e}) — backprop написан верно"

def _check7():
    init, apply_g = _need("init_network"), _need("apply_gradients")
    net = init([3, 2], seed=4)
    w0 = net["weights"][0].copy(); b0 = net["biases"][0].copy()
    gw = [np.ones_like(w0)]; gb = [np.ones_like(b0)]
    apply_g(net, gw, gb, 0.5)
    assert np.allclose(net["weights"][0], w0 - 0.5), \
        "шаг должен быть W = W - eta * grad (мы спускаемся, а не поднимаемся)"
    assert np.allclose(net["biases"][0], b0 - 0.5), "смещения обновляются по тому же правилу"
    return "шаг градиентного спуска сделан в правильную сторону"

def _check8():
    init, train = _need("init_network"), _need("train")
    net = init([784, 16, 16, 10], seed=7)
    before = accuracy(net, X_test[:, :2000], y_test[:2000])
    train(net, X_train[:, :6000], Y_train[:, :6000], epochs=2, batch_size=10, eta=3.0)
    after = accuracy(net, X_test[:, :2000], y_test[:2000])
    assert after > 0.55, (f"после 2 эпох точность {after:.1%} — обучение не идёт. "
                          "Проверь, что мини-батчи режутся из перемешанного порядка "
                          "и что apply_gradients вызывается на каждом батче")
    return f"за 2 эпохи на 6000 картинках точность выросла с {before:.1%} до {after:.1%}"

_CHECKS = {1: ("sigmoid", _check1), 2: ("sigmoid_prime", _check2), 3: ("init_network", _check3),
           4: ("forward", _check4), 5: ("predict / accuracy", _check5), 6: ("backprop", _check6),
           7: ("apply_gradients", _check7), 8: ("train", _check8)}
_DONE = set()

def check(step):
    name, fn = _CHECKS[step]
    try:
        msg = fn()
    except NotImplementedError:
        print(f"{_NO} Шаг {step} ({name}): функция ещё не написана — замени TODO своим кодом")
        return
    except AssertionError as e:
        print(f"{_NO} Шаг {step} ({name}): {e}")
        return
    except Exception as e:
        print(f"{_NO} Шаг {step} ({name}): код упал с ошибкой {type(e).__name__}: {e}")
        return
    _DONE.add(step)
    print(f"{_OK} Шаг {step} ({name}): {msg}")
    print(f"   Пройдено {len(_DONE)} из 8 " + "■" * len(_DONE) + "□" * (8 - len(_DONE)))
    if len(_DONE) == 8:
        print("\n\U0001f389 Все части на месте — сеть можно обучать!")

print("Проверки готовы. Пиши функции и вызывай check(1), check(2), ...")'''


EX = []

EX.append(dict(
    step=1,
    md=r'''## Шаг 1 из 8. Функция активации

Нейрон получает взвешенную сумму входов $z$ — любое число от $-\infty$ до $+\infty$. Но «насколько сильно нейрон горит» мы хотим измерять числом от 0 до 1. Для этого сумму пропускают через **сигмоиду**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Большие положительные $z$ она превращает почти в 1, большие отрицательные — почти в 0, а $\sigma(0) = 0.5$.

**Задание:** одна строка. Функция должна работать сразу со всем массивом (numpy делает это поэлементно, циклы не нужны).

*Подводный камень:* при $z = -800$ выражение `np.exp(800)` переполнится. Оберни аргумент в `np.clip(z, -500, 500)`.''',
    todo=r'''def sigmoid(z):
    """σ(z) = 1 / (1 + e^(-z)). Работает с числом и с массивом любой формы."""
    # TODO: одна строка. Понадобится np.exp и np.clip
    raise NotImplementedError''',
    sol=r'''def sigmoid(z):
    """σ(z) = 1 / (1 + e^(-z)). Работает с числом и с массивом любой формы."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))'''))

EX.append(dict(
    step=2,
    md=r'''## Шаг 2 из 8. Производная сигмоиды

Обучение — это спуск по склону: нам нужно знать, как меняется выход нейрона при малом изменении $z$. У сигмоиды производная выражается через неё же саму, и это очень удобно:

$$\sigma'(z) = \sigma(z)\,\bigl(1 - \sigma(z)\bigr)$$

**Задание:** две строки, используя уже написанную `sigmoid`.

Проверка сравнит твою формулу с численной производной $\dfrac{\sigma(z+\varepsilon) - \sigma(z-\varepsilon)}{2\varepsilon}$.''',
    todo=r'''def sigmoid_prime(z):
    """σ'(z) = σ(z) · (1 − σ(z))"""
    # TODO: вызови sigmoid(z) и составь произведение
    raise NotImplementedError''',
    sol=r'''def sigmoid_prime(z):
    """σ'(z) = σ(z) · (1 − σ(z))"""
    s = sigmoid(z)
    return s * (1.0 - s)'''))

EX.append(dict(
    step=3,
    md=r'''## Шаг 3 из 8. Создаём сеть

Наша сеть — **784 → 16 → 16 → 10**: 784 входных нейрона (пиксели 28×28), два скрытых слоя по 16 нейронов и 10 выходных (по одному на цифру).

Между соседними слоями — матрица весов. Если в слое $l$ стоит $n$ нейронов, а в предыдущем $m$, то `weights[l]` имеет форму **(n, m)**, а `biases[l]` — **(n, 1)**, столбец.

Сеть будем хранить обычным словарём:

```python
{"sizes": [784, 16, 16, 10], "weights": [W1, W2, W3], "biases": [b1, b2, b3]}
```

**Задание:** заполнить веса случайными числами, смещения — нулями.

*Важная деталь:* случайные числа надо поделить на `np.sqrt(число входов нейрона)`. Иначе сумма 784 слагаемых даст огромный $z$, сигмоида упрётся в 0 или 1, её производная станет почти нулевой — и сеть перестанет учиться. Это называется **насыщением**.''',
    todo=r'''def init_network(sizes, seed=0):
    """Создаёт сеть из списка размеров слоёв, например [784, 16, 16, 10].

    weights[l] — матрица (нейронов в слое l+1) × (нейронов в слое l),
                 случайные числа из rng.standard_normal, поделённые на sqrt(входов)
    biases[l]  — столбец нулей формы (нейронов в слое l+1, 1)
    """
    rng = np.random.default_rng(seed)
    # TODO: собери списки weights и biases (удобно через zip(sizes[:-1], sizes[1:]))
    raise NotImplementedError''',
    sol=r'''def init_network(sizes, seed=0):
    """Создаёт сеть из списка размеров слоёв, например [784, 16, 16, 10]."""
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal((n_out, n_in)) / np.sqrt(n_in)
               for n_in, n_out in zip(sizes[:-1], sizes[1:])]
    biases = [np.zeros((n_out, 1)) for n_out in sizes[1:]]
    return {"sizes": list(sizes), "weights": weights, "biases": biases}'''))

EX.append(dict(
    step=4,
    md=r'''## Шаг 4 из 8. Прямой ход

Вот и вся работа сети — одно матричное умножение на слой:

$$z^{l} = W^{l} a^{l-1} + b^{l} \qquad a^{l} = \sigma(z^{l})$$

Активации первого слоя $a^{0}$ — это яркости пикселей. Дальше каждый слой пересчитывает следующий, и активации последнего слоя — ответ сети.

Мы будем прогонять сразу **пачку картинок**: `x` имеет форму `(784, m)`, где столбец = картинка. Формула не меняется — numpy умножит матрицу на матрицу, а `b` формы `(n, 1)` сам «размножится» по всем столбцам.

**Задание:** вернуть **два списка** — все $z$ и все $a$. Они понадобятся для backprop, поэтому промежуточные значения надо сохранять, а не выбрасывать.

Обрати внимание: `activations` длиннее `zs` на один элемент, потому что в начало кладётся сам вход.''',
    todo=r'''def forward(net, a):
    """Прямой ход. a — матрица (784, m): столбец = картинка.

    Возвращает (zs, activations):
        zs[l]          — взвешенные суммы слоя l
        activations[0] — сам вход a
        activations[-1] — ответ сети, форма (10, m)
    """
    activations = [a]
    zs = []
    # TODO: пройди по слоям (zip(net["weights"], net["biases"])),
    #       посчитай z = W @ a + b, потом a = sigmoid(z), сохраняя оба списка
    raise NotImplementedError''',
    sol=r'''def forward(net, a):
    """Прямой ход. a — матрица (784, m): столбец = картинка."""
    activations = [a]
    zs = []
    for W, b in zip(net["weights"], net["biases"]):
        z = W @ a + b
        zs.append(z)
        a = sigmoid(z)
        activations.append(a)
    return zs, activations'''))

EX.append(dict(
    step=5,
    md=r'''## Шаг 5 из 8. Ответ сети и точность

Выходной слой — 10 чисел от 0 до 1. Ответом считаем номер самого «яркого» нейрона: если ярче всех горит нейрон номер 7, сеть говорит «это семёрка».

**Задание:** две коротких функции. `np.argmax(..., axis=0)` даст номер максимума в каждом столбце.

После этой ячейки уже можно измерить точность необученной сети. Она будет около **10%** — ровно как у случайного угадывания из десяти вариантов. Это наша точка отсчёта.''',
    todo=r'''def predict(net, x):
    """Номер самого активного выходного нейрона для каждой картинки. Форма ответа: (m,)"""
    # TODO: прогони через forward и возьми np.argmax по нужной оси
    raise NotImplementedError


def accuracy(net, x, digits):
    """Доля правильных ответов — число от 0 до 1."""
    # TODO: сравни predict(net, x) с digits и посчитай среднее
    raise NotImplementedError''',
    sol=r'''def predict(net, x):
    """Номер самого активного выходного нейрона для каждой картинки. Форма ответа: (m,)"""
    _, activations = forward(net, x)
    return np.argmax(activations[-1], axis=0)


def accuracy(net, x, digits):
    """Доля правильных ответов — число от 0 до 1."""
    return float(np.mean(predict(net, x) == digits))'''))

EX.append(dict(
    step=6,
    md=r'''## Шаг 6 из 8. Обратное распространение ошибки

Самая важная часть воркшопа. Мы хотим узнать, **как надо подкрутить каждый вес**, чтобы ошибка уменьшилась.

Стоимость (она уже определена в первой ячейке, функция `cost`):

$$C = \tfrac{1}{2}\sum_j \bigl(a^{L}_j - y_j\bigr)^2$$

где $y$ — правильный ответ: для картинки с цифрой 3 это столбец из нулей с единицей на позиции 3.

Введём $\delta^{l}$ — «насколько виноват» каждый нейрон слоя $l$. Тогда работают четыре формулы:

$$\delta^{L} = (a^{L} - y) \odot \sigma'(z^{L})$$
$$\delta^{l} = \bigl(W^{l+1}\bigr)^{T} \delta^{l+1} \odot \sigma'(z^{l})$$
$$\frac{\partial C}{\partial W^{l}} = \delta^{l} \bigl(a^{l-1}\bigr)^{T} \qquad \frac{\partial C}{\partial b^{l}} = \delta^{l}$$

Здесь $\odot$ — поэлементное умножение (в numpy просто `*`), а $W^T$ — транспонирование, `W.T`.

Смысл второй формулы: ошибка «протаскивается» назад через те же веса, по которым сигнал шёл вперёд.

**Задание:** посчитать градиенты для пачки из `m` картинок и **усреднить по батчу** — поделить на `m`. Для смещений сумма по столбцам: `delta.sum(axis=1, keepdims=True)` (без `keepdims` потеряется форма столбца).

Проверка сдвинет каждый вес на $\pm\varepsilon$, посмотрит, как изменилась стоимость, и сравнит с твоим градиентом. Если формулы верны — совпадёт до седьмого знака.''',
    todo=r'''def backprop(net, x, y):
    """Градиенты стоимости по всем весам и смещениям для пачки картинок.

    x — (784, m), y — (10, m) правильные ответы в виде столбцов из нулей с одной единицей.
    Возвращает (grad_w, grad_b) — списки той же длины и формы, что net["weights"] / net["biases"].
    """
    m = x.shape[1]
    zs, activations = forward(net, x)
    L = len(net["weights"])
    grad_w = [None] * L
    grad_b = [None] * L

    # TODO 1: ошибка последнего слоя
    #         delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    # TODO 2: grad_w[-1] и grad_b[-1] по формулам выше, не забудь поделить на m
    # TODO 3: цикл назад по слоям — for l in range(2, L + 1):
    #         delta = (net["weights"][-l + 1].T @ delta) * sigmoid_prime(zs[-l])
    #         и снова заполни grad_w[-l], grad_b[-l]
    raise NotImplementedError''',
    sol=r'''def backprop(net, x, y):
    """Градиенты стоимости по всем весам и смещениям для пачки картинок."""
    m = x.shape[1]
    zs, activations = forward(net, x)
    L = len(net["weights"])
    grad_w = [None] * L
    grad_b = [None] * L

    # ошибка выходного слоя
    delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    grad_w[-1] = delta @ activations[-2].T / m
    grad_b[-1] = delta.sum(axis=1, keepdims=True) / m

    # тянем ошибку назад: слой L-1, L-2, ...
    for l in range(2, L + 1):
        delta = (net["weights"][-l + 1].T @ delta) * sigmoid_prime(zs[-l])
        grad_w[-l] = delta @ activations[-l - 1].T / m
        grad_b[-l] = delta.sum(axis=1, keepdims=True) / m

    return grad_w, grad_b'''))

EX.append(dict(
    step=7,
    md=r'''## Шаг 7 из 8. Шаг градиентного спуска

Градиент показывает направление **самого быстрого роста** стоимости. Нам нужно вниз, поэтому идём в противоположную сторону:

$$W \leftarrow W - \eta \, \frac{\partial C}{\partial W} \qquad b \leftarrow b - \eta \, \frac{\partial C}{\partial b}$$

$\eta$ (эта) — скорость обучения, длина шага. Слишком маленькая — спуск займёт вечность, слишком большая — будем перепрыгивать через минимум.

**Задание:** три строки. Меняем содержимое `net` на месте.''',
    todo=r'''def apply_gradients(net, grad_w, grad_b, eta):
    """Шаг спуска: сдвигаем все веса и смещения против градиента."""
    # TODO: пройди по всем слоям и вычти eta * градиент
    raise NotImplementedError''',
    sol=r'''def apply_gradients(net, grad_w, grad_b, eta):
    """Шаг спуска: сдвигаем все веса и смещения против градиента."""
    for i in range(len(net["weights"])):
        net["weights"][i] -= eta * grad_w[i]
        net["biases"][i] -= eta * grad_b[i]'''))

EX.append(dict(
    step=8,
    md=r'''## Шаг 8 из 8. Цикл обучения

Считать градиент сразу по всем 60 000 картинкам честно, но медленно. Поэтому применяют **стохастический** градиентный спуск: перемешиваем выборку, режем на мини-батчи по 10 картинок и делаем шаг на каждом. Направление получается чуть неточным, зато шагов в тысячи раз больше — как спускаться с горы быстрыми неуверенными шагами вместо редких выверенных.

Один проход по всей выборке называется **эпохой**.

**Задание:** собрать цикл из уже готовых кубиков — `backprop` и `apply_gradients`.

*Подсказка по перемешиванию:* `order = rng.permutation(n)` даёт случайный порядок номеров, а `x[:, order[i:i+batch_size]]` — очередной мини-батч.''',
    todo=r'''def train(net, x, y, epochs=10, batch_size=10, eta=3.0, x_test=None, y_test_digits=None):
    """Стохастический градиентный спуск."""
    rng = np.random.default_rng(0)
    n = x.shape[1]
    for epoch in range(1, epochs + 1):
        # TODO 1: перемешать номера картинок
        # TODO 2: пройти по выборке шагами batch_size, для каждого мини-батча
        #         посчитать backprop и вызвать apply_gradients
        raise NotImplementedError

        if x_test is not None:
            print(f"Эпоха {epoch:2d}: точность на тесте {accuracy(net, x_test, y_test_digits)*100:5.2f}%")
    return net''',
    sol=r'''def train(net, x, y, epochs=10, batch_size=10, eta=3.0, x_test=None, y_test_digits=None):
    """Стохастический градиентный спуск."""
    rng = np.random.default_rng(0)
    n = x.shape[1]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            grad_w, grad_b = backprop(net, x[:, idx], y[:, idx])
            apply_gradients(net, grad_w, grad_b, eta)

        if x_test is not None:
            print(f"Эпоха {epoch:2d}: точность на тесте {accuracy(net, x_test, y_test_digits)*100:5.2f}%")
    return net'''))


FINAL = [
("md", r'''---
# Всё готово — запускаем

Ниже уже ничего писать не надо, только запускать ячейки.'''),

("code", r'''# Обучение настоящей сети: 30 эпох по всем 60 000 картинкам (около минуты)
net = init_network([784, 16, 16, 10], seed=42)
print(f"До обучения точность: {accuracy(net, X_test, y_test)*100:.2f}%  (случайное угадывание)\n")

train(net, X_train, Y_train, epochs=30, batch_size=10, eta=3.0,
      x_test=X_test, y_test_digits=y_test)

print(f"\nИтог: {accuracy(net, X_test, y_test)*100:.2f}% правильных ответов на 10 000 картинок,")
print(f"которых сеть никогда не видела.")'''),

("md", r'''## Смотрим на ответы

Каждый прогон показывает случайную картинку из тестовой выборки и активации выходного слоя.'''),

("code", r'''i = np.random.randint(X_test.shape[1])
show_prediction(net, X_test[:, i], int(y_test[i]))'''),

("md", r'''## Где сеть ошибается

Гораздо интереснее посмотреть на ошибки. Обрати внимание: в этих случаях активации обычно «размазаны» между двумя цифрами — сеть сомневается ровно там, где засомневался бы человек.'''),

("code", r'''errors = np.where(predict(net, X_test) != y_test)[0]
print(f"Сеть ошибается на {len(errors)} картинках из {len(y_test)}")
for i in np.random.choice(errors, 3, replace=False):
    show_prediction(net, X_test[:, i], int(y_test[i]))'''),

("md", r'''## Что выучил первый слой?

Мы надеялись, что 16 нейронов первого слоя научатся распознавать палочки, петельки и дуги — из которых потом складываются цифры. Нарисуем веса каждого нейрона как картинку 28×28: красное — положительный вес, синее — отрицательный.

Посмотри на результат и сравни с ожиданием. Это главный сюрприз из видео 3Blue1Brown.'''),

("code", r'''W = net["weights"][0]
lim = np.abs(W).max()
fig, axes = plt.subplots(2, 8, figsize=(14, 4.4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(W[i].reshape(28, 28), cmap="bwr", vmin=-lim, vmax=lim)
    ax.set_title(f"нейрон {i}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Веса нейронов первого скрытого слоя", fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.show()'''),

("md", r'''## Рисуем свою цифру

Рисуй мышкой в квадрате, потом нажми «Распознать».

Перед подачей в сеть рисунок проходит через `to_mnist`: обрезается по краям чернил, вписывается в квадрат 20×20 и центрируется по центру масс — именно так подготовлены картинки в MNIST. Без этой обработки точность на нарисованных от руки цифрах падает с ~93% до ~26%: сеть никогда не видела цифру, сдвинутую в угол.

Попробуй нарисовать цифру аккуратно, а потом — неряшливо или с непривычным наклоном. Хорошо видно, на чём именно сеть ломается.'''),

("code", r'''CANVAS_HTML = """
<canvas id="cnv" width="280" height="280"
        style="border:2px solid #444;border-radius:6px;background:#fff;touch-action:none;cursor:crosshair"></canvas>
<div style="margin-top:8px">
  <button id="btn_clear" style="padding:6px 14px">Очистить</button>
  <button id="btn_done"  style="padding:6px 14px;font-weight:bold">Распознать</button>
</div>
<script>
var cnv = document.getElementById('cnv'), ctx = cnv.getContext('2d');
ctx.lineWidth = 22; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#000';
var drawing = false;
function pos(e) { var r = cnv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
cnv.addEventListener('pointerdown', function(e) { drawing = true; var p = pos(e); ctx.beginPath(); ctx.moveTo(p[0], p[1]); });
cnv.addEventListener('pointermove', function(e) { if (!drawing) return; var p = pos(e); ctx.lineTo(p[0], p[1]); ctx.stroke(); });
window.addEventListener('pointerup', function() { drawing = false; });
document.getElementById('btn_clear').onclick = function() { ctx.clearRect(0, 0, 280, 280); };
var drawn_digit = new Promise(function(resolve) {
  document.getElementById('btn_done').onclick = function() {
    var d = ctx.getImageData(0, 0, 280, 280).data, s = '';
    for (var i = 0; i < 280 * 280; i++) { s += String.fromCharCode(d[i * 4 + 3]); }  // альфа-канал = чернила
    resolve(btoa(s));
  };
});
</script>
"""

try:
    from IPython.display import display, HTML
    from google.colab import output as _colab_output
    display(HTML(CANVAS_HTML))
    b64 = _colab_output.eval_js("drawn_digit", timeout_sec=600)
    canvas = np.frombuffer(base64.b64decode(b64), np.uint8).reshape(280, 280).astype(float) / 255.0
    show_prediction(net, to_mnist(canvas))
except ImportError:
    print("Холст для рисования работает только в Google Colab.")
    print("Показываю случайную цифру из MNIST вместо рисунка:")
    i = np.random.randint(X_test.shape[1])
    show_prediction(net, X_test[:, i], int(y_test[i]))'''),

("md", r'''---
## Что покрутить, если осталось время

Каждый эксперимент — это перезапуск двух ячеек: создать сеть заново и обучить.

```python
net = init_network([784, 30, 30, 10], seed=42)   # больше нейронов
net = init_network([784, 16, 10], seed=42)       # один скрытый слой вместо двух
net = init_network([784, 4, 4, 10], seed=42)     # всего 4 нейрона в слое — сколько выжмет?
train(net, X_train, Y_train, epochs=10, eta=0.01, x_test=X_test, y_test_digits=y_test)  # шаг слишком мал
train(net, X_train, Y_train, epochs=10, eta=30.0, x_test=X_test, y_test_digits=y_test)  # шаг слишком велик
```

Вопросы для обсуждения:

1. Два скрытых слоя по 16 нейронов дают около 94%. Сколько даёт **один** слой? А четыре нейрона вместо шестнадцати?
2. Точность на обучающей выборке заметно выше, чем на тестовой. Что это значит?
3. Веса первого слоя не похожи на палочки и петельки. Значит ли это, что сеть работает «неправильно»?
4. Что произойдёт, если подать сети картинку, на которой вообще нет цифры?'''),
]


def cell(kind, src):
    kind = {"md": "markdown"}.get(kind, kind)   # Colab не понимает "md" и отказывается открывать ноутбук
    assert kind in ("markdown", "code"), f"недопустимый тип ячейки: {kind}"
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def validate(nb, path):
    """Минимальная проверка формата: Colab молча падает на кривом ноутбуке."""
    assert nb["nbformat"] == 4, "ожидается nbformat 4"
    for i, c in enumerate(nb["cells"]):
        t = c["cell_type"]
        assert t in ("markdown", "code"), f"{path}: ячейка {i} имеет cell_type={t!r}"
        assert isinstance(c["source"], list), f"{path}: ячейка {i}: source должен быть списком строк"
        assert isinstance(c["metadata"], dict), f"{path}: ячейка {i}: нет metadata"
        if t == "code":
            assert "outputs" in c and "execution_count" in c, \
                f"{path}: ячейка {i}: у code-ячейки должны быть outputs и execution_count"
    print(f"  проверка формата пройдена: {len(nb['cells'])} ячеек")


def build(solution: bool):
    cells = [cell("markdown", HEADER_SOL if solution else HEADER)]
    cells.append(cell("code", SETUP))
    for ex in EX:
        cells.append(cell("markdown", ex["md"]))
        cells.append(cell("code", ex["sol"] if solution else ex["todo"]))
        cells.append(cell("code", f'check({ex["step"]})'))
    for kind, src in FINAL:
        cells.append(cell(kind, src))
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"},
                         "colab": {"provenance": [], "toc_visible": True}},
            "nbformat": 4, "nbformat_minor": 0}


HEADER = r'''# Собираем нейросеть, которая узнаёт цифры

Мы напишем с нуля сеть **784 → 16 → 16 → 10** — ту самую, что разбирается в видео [3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk). Без PyTorch и TensorFlow: только numpy и восемь функций, которые ты напишешь сам.

**Как это устроено.** Восемь шагов, в каждом — ячейка с `TODO` и ячейка с проверкой. Пиши код, запускай `check(N)`, получай зелёную галочку. Проверка не просто смотрит на результат: например, твой `backprop` она сверит с численной производной и подскажет, где именно ошибка.

Когда все восемь галочек собраны, тот же самый код обучает сеть до **~94%** правильных ответов и распознаёт цифру, которую ты нарисуешь мышкой.

**Что понадобится знать:** умножение матриц и производная. Всё остальное объясняется по дороге.

Начни с ячейки ниже — она скачает MNIST и включит проверки. Дальше иди по шагам сверху вниз.'''

HEADER_SOL = r'''# Собираем нейросеть, которая узнаёт цифры — РЕШЕНИЕ

Версия для ведущего: все восемь функций уже написаны. Ноутбук проходит целиком сверху вниз и даёт около 94% на тестовой выборке.

Раздавать участникам нужно `workshop.ipynb`.'''

for sol, path in ((False, sys.argv[1]), (True, sys.argv[2])):
    nb = build(sol)
    validate(nb, path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("написан", path)
