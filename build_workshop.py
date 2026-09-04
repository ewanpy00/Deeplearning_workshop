"""Generates all four workshop notebooks from a single source."""
import json, os, sys


def cell(kind, src):
    kind = {"md": "markdown"}.get(kind, kind)   # Colab refuses to open a notebook with "md"
    assert kind in ("markdown", "code"), f"invalid cell type: {kind}"
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def validate(nb, path):
    """Minimal format check: Colab fails silently on a malformed notebook."""
    assert nb["nbformat"] == 4, "expected nbformat 4"
    for i, c in enumerate(nb["cells"]):
        t = c["cell_type"]
        assert t in ("markdown", "code"), f"{path}: cell {i} has cell_type={t!r}"
        assert isinstance(c["source"], list), f"{path}: cell {i}: source must be a list of lines"
        assert isinstance(c["metadata"], dict), f"{path}: cell {i}: metadata missing"
        if t == "code":
            assert "outputs" in c and "execution_count" in c, \
                f"{path}: cell {i}: code cell needs outputs and execution_count"
    print(f"  format check passed: {len(nb['cells'])} cells")


NB_METADATA = {"kernelspec": {"display_name": "Python 3", "name": "python3"},
               "language_info": {"name": "python"},
               "colab": {"provenance": [], "toc_visible": True}}

DATA_LOADER = '''import gzip, struct, urllib.request, os, base64
import numpy as np
import matplotlib.pyplot as plt

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
'''

PREPROCESS = '''def resize_bilinear(img, oh, ow):
    h, w = img.shape
    ys, xs = np.linspace(0, h-1, oh), np.linspace(0, w-1, ow)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0+1, h-1), np.minimum(x0+1, w-1)
    wy, wx = (ys-y0)[:, None], (xs-x0)[None, :]
    top = img[np.ix_(y0, x0)]*(1-wx) + img[np.ix_(y0, x1)]*wx
    bot = img[np.ix_(y1, x0)]*(1-wx) + img[np.ix_(y1, x1)]*wx
    return top*(1-wy) + bot*wy

def to_mnist(canvas, box=20, size=28):
    """Crops to the ink, fits it into a 20x20 box and centres it by centre of mass."""
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
'''

CANVAS_CELL = r'''CANVAS_HTML = """
<canvas id="cnv" width="280" height="280"
        style="border:2px solid #444;border-radius:6px;background:#fff;touch-action:none;cursor:crosshair"></canvas>
<div style="margin-top:8px">
  <button id="btn_clear" style="padding:6px 14px">Clear</button>
  <button id="btn_done"  style="padding:6px 14px;font-weight:bold">Recognise</button>
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
    for (var i = 0; i < 280 * 280; i++) { s += String.fromCharCode(d[i * 4 + 3]); }  // alpha channel = ink
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
    print("The drawing canvas only works in Google Colab.")
    print("Showing a random MNIST digit instead:")
    i = np.random.randint(X_test.shape[1])
    show_prediction(net, X_test[:, i], int(y_test[i]))'''


# ============================================================================
#                    ADVANCED TRACK — write every function yourself
# ============================================================================

SETUP = '''#@title Run this cell and move on (downloads MNIST + sets up the checks) { display-mode: "form" }
''' + DATA_LOADER + '''
print("Downloading MNIST ...")
_raw = {k: _get(v) for k, v in FILES.items()}
X_train = _raw["train_images"].T.astype(np.float64) / 255.0     # 784 x 60000
d_train = _raw["train_labels"].astype(np.int64)                 # digits 0..9
Y_train = one_hot(d_train)                                      # 10 x 60000
X_test  = _raw["test_images"].T.astype(np.float64) / 255.0
y_test  = _raw["test_labels"].astype(np.int64)
print(f"Ready: {X_train.shape[1]} images to train on, {X_test.shape[1]} to test on")

# ------------------------------------------------------- cost function (given to you)
def cost(net, x, y):
    """C = average over images of 1/2 * sum (a - y)^2 — the one from the video."""
    _, activations = forward(net, x)
    return float(np.mean(0.5 * np.sum((activations[-1] - y) ** 2, axis=0)))

# ------------------------------- turning a drawing into MNIST format (given to you)
''' + PREPROCESS + '''
# ------------------------------------------------------------------- showing an answer
def show_prediction(net, img, true_digit=None):
    _, acts = forward(net, img.reshape(-1, 1))
    out = acts[-1].ravel(); guess = int(np.argmax(out))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 2]})
    a1.imshow(img.reshape(28, 28), cmap="gray_r"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("network input")
    colors = ["#d1495b" if d == guess else "#4c72b0" for d in range(10)]
    a2.barh(np.arange(10), out, color=colors); a2.invert_yaxis()
    a2.set_yticks(range(10)); a2.set_xlim(0, 1); a2.set_title("output layer")
    t = f"network says: {guess} ({out[guess]*100:.1f}%)"
    if true_digit is not None:
        t += "  — correct" if guess == true_digit else f"  — wrong, it is a {true_digit}"
    fig.suptitle(t, fontsize=13); plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------------- the checks
# There is not a single answer below: the checks run YOUR code and test whether it
# agrees with itself and with the maths. Nothing to peek at :)
_OK, _NO = "\\u2705", "\\u274c"

def _need(name):
    f = globals().get(name)
    if f is None: raise AssertionError(f"{name}() is not defined yet — run the cell that defines it")
    return f

def _check1():
    s = _need("sigmoid")
    assert abs(s(0.0) - 0.5) < 1e-12, "sigmoid(0) must be exactly 0.5"
    assert abs(s(2.0) - 0.8807970779778823) < 1e-9, "sigmoid(2) must be 0.8807970779778823"
    z = np.array([[-1.0, 0.0], [1.0, 2.0]])
    r = np.asarray(s(z))
    assert r.shape == z.shape, "the function must work element-wise on arrays (no loops)"
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        big = np.asarray(s(np.array([-800.0, 800.0])))
    assert np.isfinite(big).all(), "large |z| produces nan — clip z with np.clip(z, -500, 500)"
    assert abs(big[0]) < 1e-9 and abs(big[1] - 1) < 1e-9, "z = -800 should give ~0, z = 800 should give ~1"
    if any(issubclass(c.category, RuntimeWarning) for c in caught):
        return ("the values are right, but numpy warns about overflow in np.exp. "
                "Wrap the argument: np.exp(-np.clip(z, -500, 500))")
    return "sigmoid works and does not blow up on large z"

def _check2():
    s, sp = _need("sigmoid"), _need("sigmoid_prime")
    z = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    num = (np.asarray(s(z + 1e-5)) - np.asarray(s(z - 1e-5))) / 2e-5
    err = np.max(np.abs(np.asarray(sp(z)) - num))
    assert err < 1e-6, f"does not match the numerical derivative of your own sigmoid (off by {err:.2e})"
    return "the derivative matches the numerical one — your formula is right"

def _check3():
    init = _need("init_network")
    net = init([784, 16, 16, 10], seed=0)
    assert isinstance(net, dict) and {"sizes", "weights", "biases"} <= set(net), \\
        "return a dict with keys 'sizes', 'weights', 'biases'"
    W, B = net["weights"], net["biases"]
    assert len(W) == 3 and len(B) == 3, f"there should be 3 weight matrices (one per gap between layers), not {len(W)}"
    for i, (shape, w) in enumerate(zip([(16, 784), (16, 16), (10, 16)], W)):
        assert w.shape == shape, f"weights[{i}] should be {shape}, not {w.shape}"
    for i, (shape, b) in enumerate(zip([(16, 1), (16, 1), (10, 1)], B)):
        assert b.shape == shape, f"biases[{i}] should be {shape} (a column!), not {b.shape}"
    assert np.allclose(W[0], init([784, 16, 16, 10], seed=0)["weights"][0]), \\
        "the same seed must give the same weights — use np.random.default_rng(seed)"
    std = W[0].std()
    assert std < 0.25, (f"the weights are too large (std={std:.3f}): divide by np.sqrt(number of inputs), "
                        "otherwise z saturates the sigmoid and the network barely learns")
    return f"network 784-16-16-10 created, {sum(w.size for w in W) + sum(b.size for b in B):,} parameters"

def _check4():
    init, fwd, s = _need("init_network"), _need("forward"), _need("sigmoid")
    net = init([6, 4, 3, 2], seed=1)
    x = np.random.default_rng(5).random((6, 7))
    zs, acts = fwd(net, x)
    assert len(zs) == 3, f"zs should hold 3 entries (one per gap between layers), not {len(zs)}"
    assert len(acts) == 4, f"activations should hold 4 entries (input + 3 layers), not {len(acts)}"
    assert np.allclose(acts[0], x), "activations[0] is the input x itself"
    assert acts[-1].shape == (2, 7), f"the output should be (2, 7), got {acts[-1].shape}"
    for l in range(3):
        z_exp = net["weights"][l] @ acts[l] + net["biases"][l]
        assert np.allclose(zs[l], z_exp), f"zs[{l}] should be W @ a + b (check the order of the factors)"
        assert np.allclose(acts[l+1], np.asarray(s(zs[l]))), f"activations[{l+1}] should be sigmoid(zs[{l}])"
    return "the forward pass is correct and the shapes line up"

def _check5():
    init, pred, acc = _need("init_network"), _need("predict"), _need("accuracy")
    net = init([4, 3, 2], seed=2)
    x = np.random.default_rng(9).random((4, 25))
    p = np.asarray(pred(net, x))
    assert p.shape == (25,), f"predict should return 25 numbers, not {p.shape}"
    assert p.dtype.kind in "iu", "predict should return neuron indices (integers) — that is np.argmax"
    assert np.array_equal(p, np.argmax(forward(net, x)[1][-1], axis=0)), \\
        "the answer is the index of the most active output neuron"
    a = acc(net, x, p)
    assert abs(a - 1.0) < 1e-12, "feeding the network its own answers as truth should give an accuracy of 1.0"
    fake = np.zeros(25, dtype=int)
    assert abs(acc(net, x, fake) - np.mean(p == fake)) < 1e-12, "accuracy is the fraction of matches, between 0 and 1"
    return "prediction and accuracy both work"

def _check6():
    init, bp = _need("init_network"), _need("backprop")
    net = init([5, 4, 3, 2], seed=3)
    rng = np.random.default_rng(11)
    x = rng.random((5, 6)); y = one_hot(rng.integers(0, 2, 6))[:2]
    gw, gb = bp(net, x, y)
    assert len(gw) == 3 and len(gb) == 3, "return 3 gradient matrices, one per gap between layers"
    for i in range(3):
        assert np.asarray(gw[i]).shape == net["weights"][i].shape, \\
            f"grad_w[{i}] must have the same shape as weights[{i}]"
        assert np.asarray(gb[i]).shape == net["biases"][i].shape, \\
            f"grad_b[{i}] must have the same shape as biases[{i}] — remember keepdims=True"
    # numerical check: nudge one weight by +-eps and watch how the cost moves
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
    # common case: the formulas are right but every gradient is off by the same factor
    ratios = np.array(ratios)
    if worst > 1e-4 and ratios.size and abs(ratios.mean() - 1) > 0.02 \\
            and np.all(np.abs(ratios - ratios.mean()) < 0.02 * abs(ratios.mean())):
        raise AssertionError(
            f"the formulas are right, but every gradient is off by exactly {ratios.mean():.3g}x. "
            f"That is the division by the batch size: here m = {x.shape[1]}")
    if worst > 0.3:
        raise AssertionError(f"the gradient for {where} does not look numerical at all (off by {worst:.1%}). "
                             "Check the delta formula and the order of factors in W.T @ delta")
    if worst > 1e-4:
        raise AssertionError(f"the gradient for {where} is almost right but off by {worst:.2%}. "
                             "You probably forgot to divide by the batch size m")
    return f"gradients match the numerical ones (off by {worst:.1e}) — backprop is correct"

def _check7():
    init, apply_g = _need("init_network"), _need("apply_gradients")
    net = init([3, 2], seed=4)
    w0 = net["weights"][0].copy(); b0 = net["biases"][0].copy()
    gw = [np.ones_like(w0)]; gb = [np.ones_like(b0)]
    apply_g(net, gw, gb, 0.5)
    assert np.allclose(net["weights"][0], w0 - 0.5), \\
        "the step must be W = W - eta * grad (we are going down, not up)"
    assert np.allclose(net["biases"][0], b0 - 0.5), "biases follow the same rule"
    return "the gradient descent step goes in the right direction"

def _check8():
    init, train = _need("init_network"), _need("train")
    net = init([784, 16, 16, 10], seed=7)
    before = accuracy(net, X_test[:, :2000], y_test[:2000])
    train(net, X_train[:, :6000], Y_train[:, :6000], epochs=2, batch_size=10, eta=3.0)
    after = accuracy(net, X_test[:, :2000], y_test[:2000])
    assert after > 0.55, (f"after 2 epochs the accuracy is {after:.1%} — learning is not happening. "
                          "Check that mini-batches are cut from a shuffled order "
                          "and that apply_gradients runs on every batch")
    return f"2 epochs on 6000 images took the accuracy from {before:.1%} to {after:.1%}"

_CHECKS = {1: ("sigmoid", _check1), 2: ("sigmoid_prime", _check2), 3: ("init_network", _check3),
           4: ("forward", _check4), 5: ("predict / accuracy", _check5), 6: ("backprop", _check6),
           7: ("apply_gradients", _check7), 8: ("train", _check8)}
_DONE = set()

def check(step):
    name, fn = _CHECKS[step]
    try:
        msg = fn()
    except NotImplementedError:
        print(f"{_NO} Step {step} ({name}): not written yet — replace the TODO with your code")
        return
    except AssertionError as e:
        print(f"{_NO} Step {step} ({name}): {e}")
        return
    except Exception as e:
        print(f"{_NO} Step {step} ({name}): your code raised {type(e).__name__}: {e}")
        return
    _DONE.add(step)
    print(f"{_OK} Step {step} ({name}): {msg}")
    print(f"   {len(_DONE)} of 8 done " + "\\u25a0" * len(_DONE) + "\\u25a1" * (8 - len(_DONE)))
    if len(_DONE) == 8:
        print("\\n\\U0001f389 Every piece is in place — time to train the network!")

print("Checks are ready. Write the functions and call check(1), check(2), ...")'''


EX = []

EX.append(dict(
    step=1,
    md=r'''## Step 1 of 8. The activation function

A neuron receives a weighted sum of its inputs, $z$ — any number from $-\infty$ to $+\infty$. But "how brightly the neuron fires" should be a number between 0 and 1. So we squash the sum through a **sigmoid**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Large positive $z$ becomes almost 1, large negative $z$ almost 0, and $\sigma(0) = 0.5$.

**Your task:** one line. It has to work on a whole array at once (numpy does that element-wise, no loops needed).

*Gotcha:* at $z = -800$ the expression `np.exp(800)` overflows. Wrap the argument in `np.clip(z, -500, 500)`.''',
    todo=r'''def sigmoid(z):
    """sigma(z) = 1 / (1 + e^(-z)). Works on a number or an array of any shape."""
    # TODO: one line. You will need np.exp and np.clip
    raise NotImplementedError''',
    sol=r'''def sigmoid(z):
    """sigma(z) = 1 / (1 + e^(-z)). Works on a number or an array of any shape."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))'''))

EX.append(dict(
    step=2,
    md=r'''## Step 2 of 8. The derivative of the sigmoid

Learning is walking downhill, so we need to know how a neuron's output changes when $z$ changes a little. For the sigmoid the derivative is expressed through the sigmoid itself, which is very convenient:

$$\sigma'(z) = \sigma(z)\,\bigl(1 - \sigma(z)\bigr)$$

**Your task:** two lines, using the `sigmoid` you just wrote.

The check will compare your formula against the numerical derivative $\dfrac{\sigma(z+\varepsilon) - \sigma(z-\varepsilon)}{2\varepsilon}$.''',
    todo=r'''def sigmoid_prime(z):
    """sigma'(z) = sigma(z) * (1 - sigma(z))"""
    # TODO: call sigmoid(z) and build the product
    raise NotImplementedError''',
    sol=r'''def sigmoid_prime(z):
    """sigma'(z) = sigma(z) * (1 - sigma(z))"""
    s = sigmoid(z)
    return s * (1.0 - s)'''))

EX.append(dict(
    step=3,
    md=r'''## Step 3 of 8. Building the network

Our network is **784 → 16 → 16 → 10**: 784 input neurons (the 28×28 pixels), two hidden layers of 16 neurons each, and 10 outputs, one per digit.

Between neighbouring layers sits a weight matrix. If layer $l$ has $n$ neurons and the previous one has $m$, then `weights[l]` has shape **(n, m)** and `biases[l]` is a column of shape **(n, 1)**.

We will keep the network in a plain dictionary:

```python
{"sizes": [784, 16, 16, 10], "weights": [W1, W2, W3], "biases": [b1, b2, b3]}
```

**Your task:** fill the weights with random numbers and the biases with zeros.

*One detail that matters:* divide the random numbers by `np.sqrt(number of inputs to the neuron)`. Otherwise a sum of 784 terms gives a huge $z$, the sigmoid pins itself at 0 or 1, its derivative becomes nearly zero — and the network stops learning. This is called **saturation**.''',
    todo=r'''def init_network(sizes, seed=0):
    """Creates a network from a list of layer sizes, for example [784, 16, 16, 10].

    weights[l] — matrix (neurons in layer l+1) x (neurons in layer l),
                 random values from rng.standard_normal divided by sqrt(inputs)
    biases[l]  — column of zeros with shape (neurons in layer l+1, 1)
    """
    rng = np.random.default_rng(seed)
    # TODO: build the weights and biases lists (zip(sizes[:-1], sizes[1:]) helps)
    raise NotImplementedError''',
    sol=r'''def init_network(sizes, seed=0):
    """Creates a network from a list of layer sizes, for example [784, 16, 16, 10]."""
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal((n_out, n_in)) / np.sqrt(n_in)
               for n_in, n_out in zip(sizes[:-1], sizes[1:])]
    biases = [np.zeros((n_out, 1)) for n_out in sizes[1:]]
    return {"sizes": list(sizes), "weights": weights, "biases": biases}'''))

EX.append(dict(
    step=4,
    md=r'''## Step 4 of 8. The forward pass

This is all the network does — one matrix multiplication per layer:

$$z^{l} = W^{l} a^{l-1} + b^{l} \qquad a^{l} = \sigma(z^{l})$$

The activations of the first layer $a^{0}$ are the pixel brightnesses. Each layer computes the next one, and the activations of the last layer are the answer.

We will push a **batch of images** through at once: `x` has shape `(784, m)`, one column per image. The formula does not change — numpy multiplies matrix by matrix, and `b` of shape `(n, 1)` broadcasts across all columns by itself.

**Your task:** return **two lists** — all the $z$ values and all the $a$ values. Backprop needs them, so the intermediate results have to be kept rather than thrown away.

Note that `activations` is one entry longer than `zs`, because the input itself goes in front.''',
    todo=r'''def forward(net, a):
    """Forward pass. a is a (784, m) matrix: one column per image.

    Returns (zs, activations):
        zs[l]           — weighted sums of layer l
        activations[0]  — the input a itself
        activations[-1] — the network's answer, shape (10, m)
    """
    activations = [a]
    zs = []
    # TODO: walk the layers with zip(net["weights"], net["biases"]),
    #       compute z = W @ a + b, then a = sigmoid(z), keeping both lists
    raise NotImplementedError''',
    sol=r'''def forward(net, a):
    """Forward pass. a is a (784, m) matrix: one column per image."""
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
    md=r'''## Step 5 of 8. The answer and the accuracy

The output layer is ten numbers between 0 and 1. The answer is the index of the brightest neuron: if neuron number 7 fires hardest, the network says "seven".

**Your task:** two short functions. `np.argmax(..., axis=0)` gives the index of the maximum in each column.

After this cell you can already measure the accuracy of an untrained network. It will be around **10%** — exactly what random guessing between ten options gives. That is our baseline.''',
    todo=r'''def predict(net, x):
    """Index of the most active output neuron for each image. Shape of the result: (m,)"""
    # TODO: run the forward pass and take np.argmax along the right axis
    raise NotImplementedError


def accuracy(net, x, digits):
    """Fraction of correct answers — a number between 0 and 1."""
    # TODO: compare predict(net, x) with digits and take the mean
    raise NotImplementedError''',
    sol=r'''def predict(net, x):
    """Index of the most active output neuron for each image. Shape of the result: (m,)"""
    _, activations = forward(net, x)
    return np.argmax(activations[-1], axis=0)


def accuracy(net, x, digits):
    """Fraction of correct answers — a number between 0 and 1."""
    return float(np.mean(predict(net, x) == digits))'''))

EX.append(dict(
    step=6,
    md=r'''## Step 6 of 8. Backpropagation

The most important part of the workshop. We want to know **how each weight should be nudged** so that the error gets smaller.

The cost (already defined in the first cell as `cost`):

$$C = \tfrac{1}{2}\sum_j \bigl(a^{L}_j - y_j\bigr)^2$$

where $y$ is the right answer: for an image of a 3 it is a column of zeros with a single 1 at position 3.

Let $\delta^{l}$ be "how much each neuron of layer $l$ is to blame". Then four formulas do the work:

$$\delta^{L} = (a^{L} - y) \odot \sigma'(z^{L})$$
$$\delta^{l} = \bigl(W^{l+1}\bigr)^{T} \delta^{l+1} \odot \sigma'(z^{l})$$
$$\frac{\partial C}{\partial W^{l}} = \delta^{l} \bigl(a^{l-1}\bigr)^{T} \qquad \frac{\partial C}{\partial b^{l}} = \delta^{l}$$

Here $\odot$ is element-wise multiplication (plain `*` in numpy) and $W^T$ is the transpose, `W.T`.

What the second formula means: the error is dragged backwards through the very same weights the signal travelled forward along.

**Your task:** compute the gradients for a batch of `m` images and **average over the batch** — divide by `m`. For the biases that means summing over columns: `delta.sum(axis=1, keepdims=True)` (without `keepdims` you lose the column shape).

The check will nudge each weight by $\pm\varepsilon$, see how the cost moved, and compare that against your gradient. If the formulas are right they agree to seven decimal places.''',
    todo=r'''def backprop(net, x, y):
    """Gradients of the cost for a batch of images.

    x is (784, m), y is (10, m): the right answers as columns of zeros with a single 1.
    Returns (grad_w, grad_b) — lists matching net["weights"] / net["biases"] in shape.
    """
    m = x.shape[1]
    zs, activations = forward(net, x)
    L = len(net["weights"])
    grad_w = [None] * L
    grad_b = [None] * L

    # TODO 1: the error of the output layer
    #         delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    # TODO 2: grad_w[-1] and grad_b[-1] from the formulas above, remember to divide by m
    # TODO 3: walk backwards — for l in range(2, L + 1):
    #         delta = (net["weights"][-l + 1].T @ delta) * sigmoid_prime(zs[-l])
    #         and fill grad_w[-l], grad_b[-l] again
    raise NotImplementedError''',
    sol=r'''def backprop(net, x, y):
    """Gradients of the cost for a batch of images."""
    m = x.shape[1]
    zs, activations = forward(net, x)
    L = len(net["weights"])
    grad_w = [None] * L
    grad_b = [None] * L

    # error of the output layer
    delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    grad_w[-1] = delta @ activations[-2].T / m
    grad_b[-1] = delta.sum(axis=1, keepdims=True) / m

    # drag the error backwards: layer L-1, L-2, ...
    for l in range(2, L + 1):
        delta = (net["weights"][-l + 1].T @ delta) * sigmoid_prime(zs[-l])
        grad_w[-l] = delta @ activations[-l - 1].T / m
        grad_b[-l] = delta.sum(axis=1, keepdims=True) / m

    return grad_w, grad_b'''))

EX.append(dict(
    step=7,
    md=r'''## Step 7 of 8. One step of gradient descent

The gradient points in the direction of **steepest increase** of the cost. We want to go down, so we move the opposite way:

$$W \leftarrow W - \eta \, \frac{\partial C}{\partial W} \qquad b \leftarrow b - \eta \, \frac{\partial C}{\partial b}$$

$\eta$ (eta) is the learning rate, the length of the step. Too small and the descent takes forever; too large and we jump straight over the minimum.

**Your task:** three lines. Change the contents of `net` in place.''',
    todo=r'''def apply_gradients(net, grad_w, grad_b, eta):
    """One descent step: move all weights and biases against the gradient."""
    # TODO: walk the layers and subtract eta * gradient
    raise NotImplementedError''',
    sol=r'''def apply_gradients(net, grad_w, grad_b, eta):
    """One descent step: move all weights and biases against the gradient."""
    for i in range(len(net["weights"])):
        net["weights"][i] -= eta * grad_w[i]
        net["biases"][i] -= eta * grad_b[i]'''))

EX.append(dict(
    step=8,
    md=r'''## Step 8 of 8. The training loop

Computing the gradient over all 60,000 images at once is honest but slow. So people use **stochastic** gradient descent instead: shuffle the data, cut it into mini-batches of 10 images and take a step on each one. Each direction is slightly off, but there are thousands of times more steps — like walking down a hill in quick uncertain steps instead of rare carefully measured ones.

One pass over the whole dataset is called an **epoch**.

**Your task:** assemble the loop from the pieces you already have — `backprop` and `apply_gradients`.

*Shuffling hint:* `order = rng.permutation(n)` gives a random ordering of indices, and `x[:, order[i:i+batch_size]]` is the next mini-batch.''',
    todo=r'''def train(net, x, y, epochs=10, batch_size=10, eta=3.0, x_test=None, y_test_digits=None):
    """Stochastic gradient descent."""
    rng = np.random.default_rng(0)
    n = x.shape[1]
    for epoch in range(1, epochs + 1):
        # TODO 1: shuffle the image indices
        # TODO 2: walk the data in steps of batch_size, and for every mini-batch
        #         call backprop and then apply_gradients
        raise NotImplementedError

        if x_test is not None:
            print(f"Epoch {epoch:2d}: test accuracy {accuracy(net, x_test, y_test_digits)*100:5.2f}%")
    return net''',
    sol=r'''def train(net, x, y, epochs=10, batch_size=10, eta=3.0, x_test=None, y_test_digits=None):
    """Stochastic gradient descent."""
    rng = np.random.default_rng(0)
    n = x.shape[1]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            grad_w, grad_b = backprop(net, x[:, idx], y[:, idx])
            apply_gradients(net, grad_w, grad_b, eta)

        if x_test is not None:
            print(f"Epoch {epoch:2d}: test accuracy {accuracy(net, x_test, y_test_digits)*100:5.2f}%")
    return net'''))


FINAL = [
("md", r'''---
# Everything is ready — let's run it

Nothing left to write below, just run the cells.'''),

("code", r'''# Training the real network: 30 epochs over all 60,000 images (about a minute)
net = init_network([784, 16, 16, 10], seed=42)
print(f"Before training the accuracy is {accuracy(net, X_test, y_test)*100:.2f}%  (random guessing)\n")

train(net, X_train, Y_train, epochs=30, batch_size=10, eta=3.0,
      x_test=X_test, y_test_digits=y_test)

print(f"\nResult: {accuracy(net, X_test, y_test)*100:.2f}% correct on 10,000 images")
print("the network has never seen before.")'''),

("md", r'''## Looking at the answers

Each run shows a random image from the test set together with the activations of the output layer.'''),

("code", r'''i = np.random.randint(X_test.shape[1])
show_prediction(net, X_test[:, i], int(y_test[i]))'''),

("md", r'''## Where the network gets it wrong

The mistakes are far more interesting. Notice that in these cases the activations are usually smeared between two digits — the network hesitates exactly where a human would.'''),

("code", r'''errors = np.where(predict(net, X_test) != y_test)[0]
print(f"The network is wrong on {len(errors)} of {len(y_test)} images")
for i in np.random.choice(errors, 3, replace=False):
    show_prediction(net, X_test[:, i], int(y_test[i]))'''),

("md", r'''## What did the first layer learn?

We hoped the 16 neurons of the first layer would learn to spot strokes, loops and arcs — the pieces digits are made of. Let's draw each neuron's weights as a 28×28 image: red is a positive weight, blue a negative one.

Look at the result and compare it with that expectation. This is the big surprise from the 3Blue1Brown video.'''),

("code", r'''W = net["weights"][0]
lim = np.abs(W).max()
fig, axes = plt.subplots(2, 8, figsize=(14, 4.4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(W[i].reshape(28, 28), cmap="bwr", vmin=-lim, vmax=lim)
    ax.set_title(f"neuron {i}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Weights of the first hidden layer", fontsize=13)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.show()'''),

("md", r'''## Draw your own digit

Draw with the mouse, then press **Recognise**.

Before the drawing reaches the network it goes through `to_mnist`: cropped to the ink, fitted into a 20×20 box and centred by centre of mass — exactly how the MNIST images were prepared. Without that step the accuracy on hand-drawn digits falls from about 93% to about 26%: the network has never seen a digit pushed into a corner.

Try drawing a digit neatly, then sloppily or at an unusual slant. It is very clear what breaks the network.'''),

("code", CANVAS_CELL),

("md", r'''---
## Things to try if there is time left

Each experiment is a rerun of two cells: build the network again and train it.

```python
net = init_network([784, 30, 30, 10], seed=42)   # more neurons
net = init_network([784, 16, 10], seed=42)       # one hidden layer instead of two
net = init_network([784, 4, 4, 10], seed=42)     # only 4 neurons — how far does that get?
train(net, X_train, Y_train, epochs=10, eta=0.01, x_test=X_test, y_test_digits=y_test)  # step too small
train(net, X_train, Y_train, epochs=10, eta=30.0, x_test=X_test, y_test_digits=y_test)  # step too large
```

Questions worth discussing:

1. Two hidden layers of 16 neurons give about 94%. What does **one** layer give? What about four neurons instead of sixteen?
2. The training accuracy is noticeably higher than the test accuracy. What does that mean?
3. The first-layer weights look nothing like strokes and loops. Does that mean the network works "incorrectly"?
4. What happens if you feed the network an image with no digit on it at all?'''),
]


HEADER = r'''# Building a neural network from scratch — the long version

We will write a **784 → 16 → 16 → 10** network from scratch — the one explained in the [3Blue1Brown video](https://www.youtube.com/watch?v=aircAruvnKk). No PyTorch, no TensorFlow: just numpy and eight functions you write yourself.

If this is your first time here, start with [workshop.ipynb](https://colab.research.google.com/github/ewanpy00/Deeplearning_workshop/blob/main/workshop.ipynb) instead — the same thing in 30 minutes.

**How this works.** Eight steps, each with a `TODO` cell and a check cell. Write the code, run `check(N)`, collect a green tick. The checks do more than look at the result: your `backprop`, for instance, is compared against a numerical derivative, and the message tells you where the mistake is.

Once all eight ticks are collected, the very same code trains the network to about **94%** and recognises a digit you draw with the mouse.

**What you need to know:** matrix multiplication and derivatives. Everything else is explained along the way.

Start with the cell below — it downloads MNIST and switches the checks on. Then work top to bottom.'''

HEADER_SOL = r'''# Building a neural network from scratch — the long version, SOLUTIONS

The instructor's copy: all eight functions are filled in. The notebook runs top to bottom and reaches about 94% on the test set.

Hand out `workshop_advanced.ipynb` to participants.'''


def build(solution: bool):
    cells = [cell("markdown", HEADER_SOL if solution else HEADER)]
    cells.append(cell("code", SETUP))
    for ex in EX:
        cells.append(cell("markdown", ex["md"]))
        cells.append(cell("code", ex["sol"] if solution else ex["todo"]))
        cells.append(cell("code", f'check({ex["step"]})'))
    for kind, src in FINAL:
        cells.append(cell(kind, src))
    return {"cells": cells, "metadata": NB_METADATA, "nbformat": 4, "nbformat_minor": 0}


# ============================================================================
#                      LIGHT TRACK — 30 minutes, no prior ML
# ============================================================================
# The student writes no maths: every function is ready. They set the architecture,
# assemble the training loop from two calls, run it and experiment.

def _sol(name):
    """Pull a finished implementation out of the long version so the two tracks cannot drift."""
    for ex in EX:
        if ex["todo"].split("def ")[1].split("(")[0] == name:
            return ex["sol"]
    raise KeyError(name)


LIGHT_SETUP = '''#@title Run this cell — it downloads the data and sets everything up { display-mode: "form" }
''' + DATA_LOADER + '''
print("Downloading images of handwritten digits ...")
_raw = {k: _get(v) for k, v in FILES.items()}
X_train = _raw["train_images"].T.astype(np.float64) / 255.0
d_train = _raw["train_labels"].astype(np.int64)
Y_train = one_hot(d_train)
X_test  = _raw["test_images"].T.astype(np.float64) / 255.0
y_test  = _raw["test_labels"].astype(np.int64)
print(f"Ready: {X_train.shape[1]} images to learn from and {X_test.shape[1]} to test on")

# ------------------------------------------- the ready-made pieces the network is built from
''' + _sol("sigmoid") + '''


''' + _sol("sigmoid_prime") + '''


class _Blank:
    """The ___ placeholder that stands where a number has to be filled in."""
    def __repr__(self): return "___"
___ = _Blank()


def init_network(sizes, seed=0):
    """Creates the network: random weights between neighbouring layers, zero biases."""
    for s in sizes:
        if isinstance(s, _Blank):
            raise ValueError("There is still a ___ placeholder in sizes — replace it with a number")
        if not isinstance(s, (int, np.integer)):
            raise ValueError(f"A layer size must be a whole number, not {s!r}")
    rng = np.random.default_rng(seed)
    weights = [rng.standard_normal((n_out, n_in)) / np.sqrt(n_in)
               for n_in, n_out in zip(sizes[:-1], sizes[1:])]
    biases = [np.zeros((n_out, 1)) for n_out in sizes[1:]]
    return {"sizes": list(sizes), "weights": weights, "biases": biases}


def count_parameters(net):
    return sum(w.size for w in net["weights"]) + sum(b.size for b in net["biases"])


''' + _sol("forward") + '''


''' + _sol("predict") + '''


''' + _sol("backprop") + '''


''' + _sol("apply_gradients") + '''


def FILL_ME(*args, **kwargs):
    raise NotImplementedError(
        "There is still a FILL_ME placeholder here — replace it with the name of the right "
        "function (they are listed in the comment above the line)")

# ------------------------------------------------- drawing pictures and preparing a sketch
''' + PREPROCESS + '''
def show_pixels(column, label=None):
    """Picture on the left, the same patch as numbers on the right — an image really is numbers."""
    img = column.reshape(28, 28)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
    a1.imshow(img, cmap="gray_r"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("what we see" + (f" (this is a {label})" if label is not None else ""))
    piece = img[9:17, 9:17]
    a2.imshow(piece, cmap="gray_r", vmin=0, vmax=1)
    for r in range(8):
        for c in range(8):
            v = piece[r, c]
            a2.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 else "#666")
    a2.set_xticks([]); a2.set_yticks([])
    a2.set_title("what the computer sees (8x8 patch from the middle)")
    plt.tight_layout(); plt.show()

def show_prediction(net, img, true_digit=None):
    _, acts = forward(net, img.reshape(-1, 1))
    out = acts[-1].ravel(); guess = int(np.argmax(out))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 2]})
    a1.imshow(img.reshape(28, 28), cmap="gray_r"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("the image")
    colors = ["#d1495b" if d == guess else "#4c72b0" for d in range(10)]
    a2.barh(np.arange(10), out, color=colors); a2.invert_yaxis()
    a2.set_yticks(range(10)); a2.set_xlim(0, 1)
    a2.set_title("how brightly each of the 10 output neurons fires")
    t = f"the network says: {guess} ({out[guess]*100:.0f}% sure)"
    if true_digit is not None:
        t += "  — correct" if guess == true_digit else f"  — wrong, it is a {true_digit}"
    fig.suptitle(t, fontsize=13); plt.tight_layout(); plt.show()

# ------------------------------------------------------------------------ the two checks
def _need(name):
    f = globals().get(name)
    if f is None: raise AssertionError(f"there is no {name} yet — run the cell above")
    return f

def _lcheck1():
    sizes = _need("sizes")
    assert not any(isinstance(s, _Blank) for s in sizes), "sizes still contains ___ placeholders"
    assert sizes[0] == 784, (f"the input is {sizes[0]}, but it should be 784: a 28x28 image is "
                             "28 * 28 = 784 numbers, one per pixel")
    assert sizes[-1] == 10, (f"the output is {sizes[-1]}, but it should be 10: one neuron for "
                             "each digit from 0 to 9")
    net = _need("net")
    return (f"network {' -> '.join(map(str, sizes))} created, "
            f"with {count_parameters(net):,} adjustable numbers in it")

def _lcheck2():
    train = _need("train")
    probe = init_network([784, 16, 16, 10], seed=7)
    before = accuracy(probe, X_test[:, :2000], y_test[:2000])
    train(probe, X_train[:, :6000], Y_train[:, :6000],
          epochs=2, batch_size=10, eta=3.0, verbose=False)
    after = accuracy(probe, X_test[:, :2000], y_test[:2000])
    assert after > 0.55, (f"the accuracy barely moved ({before:.0%} -> {after:.0%}). "
                          "Check the order: backprop works out where to move the weights first, "
                          "then apply_gradients moves them")
    return f"training works: on a trial run the accuracy went from {before:.0%} to {after:.0%}"

_LCHECKS = {1: ("network built", _lcheck1), 2: ("training loop", _lcheck2)}

def check(step):
    name, fn = _LCHECKS[step]
    try:
        msg = fn()
    except NotImplementedError as e:
        print(f"\\u274c Check {step} ({name}): {e}"); return
    except (AssertionError, ValueError) as e:
        print(f"\\u274c Check {step} ({name}): {e}"); return
    except Exception as e:
        print(f"\\u274c Check {step} ({name}): your code raised {type(e).__name__}: {e}"); return
    print(f"\\u2705 Check {step} ({name}): {msg}")

print("All set. Scroll down and run the cells in order.")'''


LIGHT_TRAIN_TODO = r'''def train(net, x, y, epochs=10, batch_size=10, eta=3.0, verbose=True):
    """The training loop. You need to fill in two calls in place of FILL_ME.

    Functions available to you:
        backprop(net, images, answers)  ->  grad_w, grad_b
            looks at the error and works out which way to move every weight
        apply_gradients(net, grad_w, grad_b, eta)
            moves the weights that way; eta sets how big the step is
    """
    rng = np.random.default_rng(0)
    n = x.shape[1]

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)                      # shuffle the images
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_x = x[:, idx]                         # 10 images
            batch_y = y[:, idx]                         # 10 correct answers

            # STEP 1: work out which way to move the weights
            grad_w, grad_b = FILL_ME(net, batch_x, batch_y)

            # STEP 2: move the weights that way
            FILL_ME(net, grad_w, grad_b, eta)

        if verbose:
            print(f"Epoch {epoch:2d}: the network recognises {accuracy(net, X_test, y_test)*100:5.2f}% of digits")
    return net'''

LIGHT_TRAIN_SOL = LIGHT_TRAIN_TODO \
    .replace("grad_w, grad_b = FILL_ME(net, batch_x, batch_y)",
             "grad_w, grad_b = backprop(net, batch_x, batch_y)") \
    .replace("FILL_ME(net, grad_w, grad_b, eta)",
             "apply_gradients(net, grad_w, grad_b, eta)") \
    .replace('"""The training loop. You need to fill in two calls in place of FILL_ME.',
             '"""The training loop.')


LIGHT = [
("md", r'''# Your first neural network — in 30 minutes

By the end of this notebook you will have a working program that recognises handwritten digits. Right at the end you will draw a digit with your mouse and it will read it.

No machine learning experience needed. Almost all the code is written already — your job is to assemble a working network out of the ready pieces, train it and see what happens. There are only a few things to fill in, marked with a pencil.

**How to work:** run the cells from top to bottom with the play button on the left of each cell (or `Shift+Enter`).

**Do this right now:** `File → Save a copy in Drive`. Otherwise your edits will not be saved.'''),

("code", "LIGHT_SETUP"),

("md", r'''---
## Part 1. An image is just numbers

A computer does not see "a two". It sees a 28 × 28 square of pixels, and every pixel is a number between 0 (white) and 1 (black).

That makes **28 · 28 = 784 numbers** per image. That is everything our network gets as input.

Run the cell below. On the left is the picture, on the right is a patch from its middle written out as numbers. Change `i` and run it again to look at other digits.'''),

("code", r'''i = 0     # pick any number from 0 to 59999 and run the cell again

show_pixels(X_train[:, i], d_train[i])'''),

("md", r'''---
## Part 2. Building the network

A network is a few layers of neurons. The numbers from the image go in on the left, pass through the layers and come out as an answer on the right.

We need to decide how many neurons each layer has:

- **First layer (input).** This is where the pixels land. How many are there in a 28 × 28 image?
- **Two middle layers.** Here the network looks for something of its own, something intermediate. Let's take 16 neurons each — just a sensible number, and we will change it later.
- **Last layer (output).** One neuron per possible answer. How many digits are there?

**Fill in the two numbers in place of `___`.**'''),

{"todo": r'''sizes = [___, 16, 16, ___]
#         ^              ^
#   how many pixels   how many different
#   in a 28x28 image  digits can come out

net = init_network(sizes)

print("Network layers:", " -> ".join(str(s) for s in net["sizes"]))
print("Numbers the network will tune:", f"{count_parameters(net):,}")''',
 "sol": r'''sizes = [784, 16, 16, 10]
#         ^              ^
#   how many pixels   how many different
#   in a 28x28 image  digits can come out

net = init_network(sizes)

print("Network layers:", " -> ".join(str(s) for s in net["sizes"]))
print("Numbers the network will tune:", f"{count_parameters(net):,}")'''},

("code", "check(1)"),

("md", r'''---
## Part 3. Right now the network can do nothing

All 13 thousand of those numbers are still random. Let's see what the network answers.

The ten bars on the right are the ten output neurons, one per digit. The network answers with whichever digit has the brightest neuron.'''),

("code", r'''print(f"The untrained network gets {accuracy(net, X_test, y_test)*100:.1f}% of digits right")
print("That is about the same as guessing: ten options, so you hit one in ten.\n")

show_prediction(net, X_test[:, 0], int(y_test[0]))'''),

("md", r'''---
## Part 4. How the network learns

The idea behind learning is simple, and it fits into two actions:

1. The network looks at 10 images and gives its answers. We compare them with the correct ones and get the error.
2. Then we need to work out **which way to move each of those 13 thousand numbers** so that next time the error is a little smaller. That is what `backprop` computes.
3. And finally move them — that is `apply_gradients`.

Then take the next 10 images and repeat. One pass over all 60,000 images makes 6000 such little steps. Each one makes the network slightly better.

**Fill in the names of the two functions in place of `FILL_ME`.** The order matters: first work out where to move, then move.'''),

{"todo": "LIGHT_TRAIN_TODO", "sol": "LIGHT_TRAIN_SOL"},

("code", "check(2)"),

("md", r'''---
## Part 5. Training

Everything is ready now. Run the cell and watch the accuracy climb — it takes less than a minute.

Keep an eye on the first epoch: in a single pass the network jumps from 10% to almost 90%.'''),

("code", r'''net = init_network(sizes)          # start from a fresh network

train(net, X_train, Y_train, epochs=10, batch_size=10, eta=3.0)

print(f"\nDone. Your network recognises {accuracy(net, X_test, y_test)*100:.2f}% of digits,")
print("and these are images it has never seen before.")'''),

("md", r'''---
## Part 6. Looking at the result

Run it a few times — each run picks a random image.'''),

("code", r'''i = np.random.randint(X_test.shape[1])
show_prediction(net, X_test[:, i], int(y_test[i]))'''),

("md", r'''## Where it gets things wrong

The mistakes are the interesting part. Look at the bars: in these cases the network is usually torn between two digits, and often they are exactly the two a human would confuse as well.'''),

("code", r'''errors = np.where(predict(net, X_test) != y_test)[0]
print(f"The network is wrong on {len(errors)} images out of {len(y_test)}\n")

for i in np.random.choice(errors, 2, replace=False):
    show_prediction(net, X_test[:, i], int(y_test[i]))'''),

("md", r'''---
## Part 7. Draw your own digit

Draw with the mouse in the white square, then press **Recognise**.

Before showing your drawing to the network the program adjusts it: crops it to the ink, shrinks it and puts it in the centre. Every image the network learned from looks exactly like that, and without the adjustment it gets confused — try drawing a digit in the corner, for example.'''),

("code", CANVAS_CELL),

("md", r'''---
## Part 8. Break it

Now the most useful part: seeing what the result actually depends on. Change one number, run the cell below and see what happened to the accuracy.

| What to change | What to try | What should happen |
|---|---|---|
| size of the middle layers | `[784, 4, 4, 10]` | too few neurons to tell all the digits apart |
| number of layers | `[784, 16, 10]` | one middle layer instead of two — how much worse? |
| step size `eta` | `0.01` | tiny steps, learning almost stands still |
| step size `eta` | `30.0` | huge steps, the network jumps over the right spot |
| number of passes | `epochs=1` | how far does a single pass get you? |'''),

("code", r'''# change the numbers here and run the whole cell

my_sizes = [784, 16, 16, 10]
my_eta = 3.0
my_epochs = 5

experiment = init_network(my_sizes)
train(experiment, X_train, Y_train, epochs=my_epochs, batch_size=10, eta=my_eta)
print(f"\nResult: {accuracy(experiment, X_test, y_test)*100:.2f}%")'''),

("md", r'''---
## What you have now

A working neural network built without a single machine learning library — just numpy, which is ordinary arithmetic on arrays of numbers. There is no PyTorch and no TensorFlow here.

And more importantly: what you just assembled is not a toy diagram, it is how real neural networks work. The big models differ in size, in the shape of their layers and in the tricks used to train them, but the two actions inside the loop stay the same: **work out which way to move the weights, and move them**.

**Want to write all the maths yourself?** There is a longer version of this same notebook where `backprop` and the rest are written from scratch from the formulas — [workshop_advanced.ipynb](https://colab.research.google.com/github/ewanpy00/Deeplearning_workshop/blob/main/workshop_advanced.ipynb).

**Where all this comes from:** the [3Blue1Brown video on neural networks](https://www.youtube.com/watch?v=aircAruvnKk) — the best visual explanation of what happens inside.'''),
]


LIGHT_HEADER_SOL = r'''# Your first neural network — in 30 minutes (SOLUTIONS)

The instructor's copy: both blanks are filled in. The notebook runs top to bottom and reaches about 94%.

Hand out `workshop.ipynb` to participants.'''


def build_light(solution: bool):
    named = {"LIGHT_SETUP": LIGHT_SETUP,
             "LIGHT_TRAIN_TODO": LIGHT_TRAIN_TODO,
             "LIGHT_TRAIN_SOL": LIGHT_TRAIN_SOL}
    cells = []
    for item in LIGHT:
        if isinstance(item, dict):
            src = item["sol"] if solution else item["todo"]
            cells.append(cell("code", named.get(src, src)))
        else:
            kind, src = item
            src = named.get(src, src)
            if solution and kind == "md" and src.startswith("# Your first neural network"):
                src = LIGHT_HEADER_SOL
            cells.append(cell(kind, src))
    return {"cells": cells, "metadata": NB_METADATA, "nbformat": 4, "nbformat_minor": 0}


TARGETS = [
    ("workshop.ipynb",                   lambda: build_light(False)),
    ("workshop_solution.ipynb",          lambda: build_light(True)),
    ("workshop_advanced.ipynb",          lambda: build(False)),
    ("workshop_advanced_solution.ipynb", lambda: build(True)),
]

if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    for name, make in TARGETS:
        path = os.path.join(out_dir, name)
        nb = make()
        validate(nb, path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", path)
