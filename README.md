# A neural network that recognises 28×28 digits — 3Blue1Brown style

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ewanpy00/Deeplearning_workshop/blob/main/workshop.ipynb)

A fully connected **784 → 16 → 16 → 10** network with two hidden layers, written from
scratch in numpy. No PyTorch, no TensorFlow: the forward pass, backpropagation and
gradient descent are all implemented by hand, straight from the formulas in the
[3Blue1Brown video](https://www.youtube.com/watch?v=aircAruvnKk).

Accuracy on the MNIST test set: **94.5%**, and training takes about 15 seconds.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py          # downloads MNIST and trains the network (~15 s)
python draw.py           # draw a digit with the mouse and watch the neurons
```

## What is where

| File | Purpose |
|---|---|
| [network.py](network.py) | The network itself: `feedforward`, `backprop`, `sgd`. The main file. |
| [mnist.py](mnist.py) | Downloads and parses the MNIST IDX files (60,000 + 10,000 images). |
| [train.py](train.py) | Training, saving the weights to `model.npz`. |
| [draw.py](draw.py) | Interactive: draw a digit, see every layer's activations. |
| [visualize.py](visualize.py) | The same thing in the terminal, as ASCII art. |
| [weights.py](weights.py) | First-layer weights drawn as 28×28 images. |
| [preprocess.py](preprocess.py) | Turning a drawing into MNIST format. |
| [workshop.ipynb](workshop.ipynb) | **The handout notebook**: 30 minutes, no ML background needed. |
| [workshop_solution.ipynb](workshop_solution.ipynb) | The same with answers — for the instructor. |
| [workshop_advanced.ipynb](workshop_advanced.ipynb) | The long version: 8 exercises, all the maths written by hand. |
| [workshop_advanced_solution.ipynb](workshop_advanced_solution.ipynb) | The same with answers. |
| [build_workshop.py](build_workshop.py) | Generates all four notebooks from a single source. |

## The maths that is actually in the code

The forward pass — one matrix multiplication per layer ([network.py:53](network.py#L53)):

```
z^l = W^l · a^(l-1) + b^l          weighted sum of the inputs
a^l = σ(z^l)                       σ(x) = 1 / (1 + e^(-x))
```

The cost is quadratic, as in the video: `C = ½ · Σ (a^L − y)²`.
For an image of a 3 the target `y` is a column of zeros with a single 1 at position 3.

Backpropagation ([network.py:76](network.py#L76)) — four formulas:

```
δ^L = (a^L − y) ⊙ σ'(z^L)              error of the output layer
δ^l = (W^(l+1))^T · δ^(l+1) ⊙ σ'(z^l)  error pushed one layer back
∂C/∂W^l = δ^l · (a^(l-1))^T
∂C/∂b^l = δ^l
```

Gradient descent ([network.py:107](network.py#L107)): shuffle the data, cut it into
mini-batches of 10 images and take a step `W ← W − η · ∂C/∂W` on each.

The gradients are verified numerically: nudge one weight by ±ε, compute
`(C(w+ε) − C(w−ε)) / 2ε`, and you get the same number backprop produces
(they agree to 1e-8).

## Demos

**1. Watching the neurons fire.** `python visualize.py --index 0` prints the image as
ASCII art along with the activations of all four layers. `--count 5` shows several random images.

**2. Where the network fails.** `python visualize.py --errors --count 5` shows only the
misread digits. It is very visible that the network fails where a human would hesitate too,
and that in those cases the output activations are smeared across two candidates.

**3. What the first layer learned.** `python weights.py` draws the weights of all 16
first-layer neurons as 28×28 images. This is the key moment from the video: we expect
detectors for strokes and loops and instead get noisy blobs. The network solves the task,
but not the way a human would.

**4. Why preprocessing matters.** Before feeding a drawing to the network, `draw.py`
crops it, fits it into a 20×20 box and centres it by centre of mass — exactly how the
MNIST images were made. On artificially shifted and rescaled digits:

| | accuracy |
|---|---|
| fed in "as drawn" | 25.7% |
| after `to_mnist()` | 92.7% |

A good excuse to talk about data quality mattering as much as architecture.

## Knobs to turn

```bash
python train.py --hidden 30 30           # more neurons in the hidden layers
python train.py --hidden 16              # one hidden layer instead of two
python train.py --eta 0.01               # step far too small — barely learns
python train.py --eta 30                 # step far too large — jumps over the minimum
python train.py --cost cross-entropy --eta 0.5
```

On the cost function: the quadratic one (`mse`, the default) gives 94.5% and cross-entropy
with a tuned η gives 94.2%, so on a network this small there is hardly any difference.
Cross-entropy does make one thing vivid, though: the σ'(z) factor in the output-layer error
cancels out, so saturated neurons stop holding learning back.

## `draw.py` controls

| Key / button | Action |
|---|---|
| left mouse button | draw |
| right button | erase |
| `c` | clear the canvas |
| `n` | drop in a random MNIST digit |
| `q` | quit |

---

# Running the workshop

Two tracks out of one repository.

**The main one — [workshop.ipynb](workshop.ipynb), 30 minutes.** Aimed at a student who
knows nothing about machine learning. All the code is written already; the student sets the
architecture, assembles the training loop from two calls, runs it and experiments. There are
four things to fill in, everything else is running cells and watching. They walk away with
their own working network that reads a digit drawn with the mouse.

**The extra one — [workshop_advanced.ipynb](workshop_advanced.ipynb), about an hour.**
For people comfortable with Python: eight functions from scratch, `backprop` from the formulas
included. It is linked at the end of the light notebook, so the fast students find it themselves.

## Handing it out

The notebooks are already published — give participants this link:

**https://colab.research.google.com/github/ewanpy00/Deeplearning_workshop/blob/main/workshop.ipynb**

Colab opens the file straight from the public repository, so nothing needs installing.
Participants click **Copy to Drive** and work in their own copy — they cannot change the
original, so the same link works for any number of groups.

The long version sits at the same link with `workshop_advanced.ipynb` on the end.

If the link shows an old version after an edit, that is Colab's cache: hard-refresh with
`Cmd+Shift+R`. Make sure the change is pushed to `main`.

Fallbacks if GitHub is not reachable from the room:

- upload `workshop.ipynb` to Google Drive and share the link;
- `File → Upload notebook` in Colab itself, if you hand the file out over chat.

## Timing of the light version

| Part | What the student does | Min |
|---|---|---|
| — | Runs the setup cell, data downloads | 2 |
| 1. An image is numbers | Changes the image index, sees pixels as numbers | 3 |
| 2. Building the network | Fills 784 and 10 into `sizes` | 4 |
| 3. It can do nothing yet | Runs it, sees 10% — the guessing baseline | 2 |
| 4. How the network learns | Fills `backprop` and `apply_gradients` into the loop | 6 |
| 5. Training | Runs it, watches the accuracy climb | 4 |
| 6. Looking at the result | Studies answers and mistakes | 3 |
| 7. Draws a digit | Draws with the mouse | 3 |
| 8. Breaks it | Turns the neuron count, `eta`, the number of epochs | 5 |

About 30 minutes in total. Ten epochs of training give **94%** and take under a minute.

Strong moments to narrate: the jump from 10% to 90% in a single epoch (part 5), the mistakes
where the network wavers between two digits (part 6), and part 8, where you see first-hand
what too large a learning rate does.

## What the student fills in

Four things, each about understanding rather than syntax:

1. **`sizes = [___, 16, 16, ___]`** — how many numbers come in and how many answers there
   can be. Get it wrong and the check explains: 28 · 28 = 784 pixels, 10 digits from 0 to 9.
2. **Two calls in the training loop** — `backprop` works out which way to move the weights,
   `apply_gradients` moves them. That is the whole of learning, which is why the blanks are there.

An unfilled blank gives a readable message rather than a Python traceback:
`There is still a ___ placeholder in sizes — replace it with a number`.

Each block is followed by `check(1)` / `check(2)` — a short check with an explanation.
The second one actually trains a trial network on 6000 images and confirms the accuracy rises.

## The long version: eight steps

| Step | Function | What they write | Min |
|---|---|---|---|
| 1 | `sigmoid` | the formula $\sigma(z) = 1/(1+e^{-z})$, one line | 3 |
| 2 | `sigmoid_prime` | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | 3 |
| 3 | `init_network` | weight matrix shapes, division by $\sqrt{n}$ | 6 |
| 4 | `forward` | $z = Wa + b$, $a = \sigma(z)$ layer by layer | 8 |
| 5 | `predict`, `accuracy` | `argmax` and the fraction of matches | 4 |
| 6 | `backprop` | the four backpropagation formulas | 15 |
| 7 | `apply_gradients` | the step $W \leftarrow W - \eta\,\partial C/\partial W$ | 3 |
| 8 | `train` | the loop over epochs and mini-batches | 8 |

32 lines get written by hand; the rest is prepared scaffolding.

**The checks contain no reference implementations** — there is nothing to peek at. They run
the participant's code and test whether it agrees with itself and with the maths.

- `sigmoid_prime` is compared against the numerical derivative of *your own* `sigmoid`;
- `forward` is checked layer by layer: is `zs[l]` really `W @ activations[l] + b`?
- `backprop` is checked by numerical differentiation: every weight is nudged by $\pm\varepsilon$
  and the change in cost is compared against the computed gradient;
- `train` is checked behaviourally: two epochs must push the accuracy past 55%.

The messages point at the cause rather than just saying "wrong". Tested against common mistakes:

| What the participant did | What the check says |
|---|---|
| forgot to divide gradients by `m` | "the formulas are right, but every gradient is off by exactly 6x. That is the division by the batch size" |
| `delta.sum(axis=1)` without `keepdims` | "grad_b[0] must have the same shape as biases[0] — remember keepdims=True" |
| `W += eta * grad` instead of `-=` | "the step must be W = W − eta · grad (we are going down, not up)" |
| weights without dividing by $\sqrt{n}$ | "the weights are too large (std=0.997) … otherwise z saturates the sigmoid" |
| `sigmoid` without `np.clip` | a tick plus a note about overflow in `np.exp` |

## Editing the material

The text, hints and solutions for both versions live in [build_workshop.py](build_workshop.py).
Edit it and rebuild:

```bash
python build_workshop.py
```

That rewrites all four `.ipynb` files. The ready-made functions in the light version are taken
from the solutions of the long one, so the two tracks cannot drift apart. Each notebook is
checked against the format before being written — without that Colab silently refuses to open it.

Do not edit the `.ipynb` files directly: the next rebuild will overwrite your changes.

## If something goes wrong

- **A student falls behind** — open `workshop_solution.ipynb` and show the filled-in cell.
  The checks are stateless and the parts are independent.
- **MNIST will not download** — the notebook tries two mirrors in turn. If both are blocked,
  hand out the four files from `data/` and have people upload them through Colab's Files panel.
- **Training takes too long** — set `epochs=5`, that still gives about 93%.
- **The canvas does not draw** — it only works in Colab; outside it the cell falls back to
  showing a random MNIST digit.
- **`NameError: name 'net' is not defined`** — the student skipped a cell. The notebook has
  to be run top to bottom.
