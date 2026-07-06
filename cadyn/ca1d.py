"""
One-dimensional cellular automata (replaces the 1997 program ``CA1dim``).

Additive CA are specified by a convolution kernel applied with periodic
(wrap-around) boundary conditions, all arithmetic mod k.  Non-additive
rules are supplied as functions taking the current state array and
returning the next.  Space-time diagrams are drawn with one generation
per row (or per column, as in Figure 2 of Chapter 3).
"""

import numpy as np
import matplotlib.pyplot as plt

from .palettes import rainbow_cmap


# ---------------------------------------------------------------------- #
# additive rules: convolution kernels
# ---------------------------------------------------------------------- #
def convolve_step(state, kernel, k):
    """One generation of the additive CA with the given kernel, mod k,
    with periodic boundary conditions.  The kernel is centered: for
    kernel [a b c], new[i] = a*x[i-1] + b*x[i] + c*x[i+1]  (mod k)."""
    state = np.asarray(state, dtype=np.int64)
    kernel = np.asarray(kernel, dtype=np.int64)
    m = len(kernel) // 2
    out = np.zeros_like(state)
    for j, coef in enumerate(kernel):
        if coef:
            out += coef * np.roll(state, m - j)
    return np.mod(out, k)


def run_additive(initial, kernel, k, generations):
    """Evolve an additive CA; returns array of shape (generations+1, n)."""
    hist = np.empty((generations + 1, len(initial)), dtype=np.int64)
    hist[0] = np.mod(initial, k)
    for t in range(generations):
        hist[t + 1] = convolve_step(hist[t], kernel, k)
    return hist


def run_rule(initial, step, generations):
    """Evolve a CA given an arbitrary step function state -> state."""
    hist = [np.array(initial, dtype=np.int64)]
    for _ in range(generations):
        hist.append(step(hist[-1]))
    return np.array(hist)


# ---------------------------------------------------------------------- #
# Wolfram's elementary cellular automata (Section 3.8)
# ---------------------------------------------------------------------- #
def elementary_step(rule):
    """One of Wolfram's 256 elementary CA: k = 2, neighborhood
    (left, center, right).  The rule number's binary digits give the
    outputs for the eight neighborhood configurations, with
    (1,1,1) -> bit 7 down to (0,0,0) -> bit 0.  E.g. rule 90 is the
    additive CA with kernel [1 0 1] mod 2; rule 30 is Wolfram's
    pseudo-random rule; rule 110 is computationally universal."""
    if not 0 <= rule <= 255:
        raise ValueError("rule must be in 0..255")
    table = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int64)

    def step(x):
        left, right = np.roll(x, 1), np.roll(x, -1)
        return table[4 * left + 2 * x + right]
    return step


def run_elementary(initial, rule, generations):
    """Evolve an elementary CA; returns array (generations+1, n)."""
    return run_rule(initial, elementary_step(rule), generations)


# ---------------------------------------------------------------------- #
# non-additive rules from Chapter 3
# ---------------------------------------------------------------------- #
def sum_of_squares_step(k):
    """Section 3.6:  C_i' = C_{i-1}^2 + 2 C_i^2 + C_{i+1}^2  (mod k)."""
    def step(x):
        left, right = np.roll(x, 1), np.roll(x, -1)
        return np.mod(left**2 + 2 * x**2 + right**2, k)
    return step


def life_1d_step(rng=None, n_colors=256):
    """Section 3.7: a 1-dimensional Game of Life.  The neighborhood of
    cell i is {i-4, i-3, i-1, i+1, i+3, i+4}; a dead cell is born with
    exactly 3 live neighbors, and a live cell survives with 2 or 3.
    Nonzero values are random 'colors'; 0 means dead."""
    rng = rng or np.random.default_rng()

    def step(x):
        alive = (x != 0).astype(np.int64)
        L = sum(np.roll(alive, s) for s in (-4, -3, -1, 1, 3, 4))
        born = (alive == 0) & (L == 3)
        survive = (alive == 1) & ((L == 2) | (L == 3))
        out = np.where(survive, x, 0)
        out[born] = rng.integers(1, n_colors, born.sum())
        return out
    return step


# ---------------------------------------------------------------------- #
# elementary CA: Wolfram's rule numbering (k = 2, nearest neighbors)
# ---------------------------------------------------------------------- #
def elementary_step(rule):
    """One-step function for the elementary CA with the given Wolfram
    rule number (0..255).

    An elementary CA has k = 2 and neighborhood (left, self, right).
    The 8 possible neighborhoods, read as 3-bit numbers
    111, 110, ..., 001, 000  (i.e. 7 down to 0), are mapped to the
    corresponding bits of the rule number.  E.g. Rule 90 = 01011010_2
    is the additive kernel [1 0 1] mod 2; Rule 30 is Wolfram's chaotic
    workhorse; Rule 110 is computationally universal (Cook 2004).
    """
    if not 0 <= rule <= 255:
        raise ValueError("rule number must be in 0..255")
    table = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int64)

    def step(x):
        idx = 4 * np.roll(x, 1) + 2 * x + np.roll(x, -1)
        return table[idx]
    return step


def run_elementary(initial, rule, generations):
    """Evolve an elementary CA; returns array (generations+1, n)."""
    return run_rule(np.asarray(initial) & 1, elementary_step(rule),
                    generations)


# ---------------------------------------------------------------------- #
# entropy of 1-D CA states (extends Section 2.2 to one dimension)
# ---------------------------------------------------------------------- #
def spatial_entropy(history, k):
    """Shannon entropy (bits) of the distribution of cell values across
    the lattice, one value per generation.

    For each generation t, p_t(v) is the fraction of the n cells that
    hold value v; H_t = -sum p log2 p.  This is the natural 1-D
    analogue of the 0-dimensional entropy of Section 2.2: it measures
    how evenly the k cell values populate the lattice.  H = 0 for a
    uniform (e.g. all-zero) state and log2(k) when all values are
    equally common."""
    history = np.asarray(history)
    H = np.empty(len(history))
    n = history.shape[1]
    for t, row in enumerate(history):
        counts = np.bincount(row, minlength=k)
        p = counts[counts > 0] / n
        H[t] = float(-(p * np.log2(p)).sum())
    return H


# ---------------------------------------------------------------------- #
# polynomial representation of additive CA (Section 3.5)
# ---------------------------------------------------------------------- #
def kernel_to_poly(kernel, n):
    """Coefficient vector (length n) of the characteristic polynomial of a
    centered kernel, in the ring  Z_k[x] / (x^n - 1).  Entry j is the
    coefficient of x^j.  E.g. [1 0 1] -> x + x^{n-1}  ( = x + x^{-1} )."""
    m = len(kernel) // 2
    poly = np.zeros(n, dtype=np.int64)
    for j, coef in enumerate(kernel):
        poly[(m - j) % n] += coef
    return poly


def poly_step(state_poly, kernel_poly, k):
    """Multiply the state polynomial by the transition polynomial in
    Z_k[x]/(x^n - 1): this is exactly one CA generation (Section 3.5).
    Implemented as a circular convolution."""
    n = len(state_poly)
    out = np.zeros(n, dtype=np.int64)
    nz = np.flatnonzero(kernel_poly)
    for j in nz:
        out += kernel_poly[j] * np.roll(state_poly, j)
    return np.mod(out, k)


# ---------------------------------------------------------------------- #
# plotting
# ---------------------------------------------------------------------- #
def plot_spacetime(history, k, ax=None, zero_color="white",
                   time_axis="down", title=None, max_inches=9.0):
    """Space-time diagram with square pixels (true data aspect).
    time_axis='down' puts generation 0 at the top (the Sierpinski
    figures); 'right' puts generation 0 at the left (Figure 2 of
    Chapter 3)."""
    img = history if time_axis == "down" else history.T
    rows, cols = img.shape
    if ax is None:
        scale = max_inches / max(rows, cols)
        _, ax = plt.subplots(figsize=(cols * scale, rows * scale))
    ax.imshow(img, cmap=rainbow_cmap(k, zero_color), vmin=0, vmax=k - 1,
              interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def single_seed(n, value=1):
    """Initial state: all zeros except one central cell."""
    x = np.zeros(n, dtype=np.int64)
    x[n // 2] = value
    return x
