"""
Two-dimensional cellular automata (replaces the 1997 program ``CA2dim``).

Additive CA are given by 2-D convolution kernels applied on a torus
(periodic boundary conditions in both directions), arithmetic mod k.
Also included: Conway's Game of Life with age coloring, and the
non-additive Euler-totient rule of Section 4.6.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from .palettes import rainbow_cmap, age_cmap
from .numtheory import totient_table


# ---------------------------------------------------------------------- #
# additive rules
# ---------------------------------------------------------------------- #
KERNEL_VON_NEUMANN = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])          # Example 4.2

KERNEL_4_3 = np.array([[-1,  2, -1],
                       [ 2, -4,  2],
                       [-1,  2, -1]])               # Example 4.3


def convolve_step2d(state, kernel, k):
    """One generation of the additive CA on the torus, mod k."""
    out = convolve2d(state, kernel[::-1, ::-1], mode="same", boundary="wrap")
    return np.mod(out, k)


def run_additive(initial, kernel, k, generations, sample_at=()):
    """Evolve, optionally keeping snapshots at the listed generations.
    Returns (final_state, {gen: snapshot})."""
    state = np.mod(np.asarray(initial, dtype=np.int64), k)
    snaps = {0: state.copy()} if 0 in sample_at else {}
    for t in range(1, generations + 1):
        state = convolve_step2d(state, kernel, k)
        if t in sample_at:
            snaps[t] = state.copy()
    return state, snaps


def single_seed(n, m=None, value=1):
    m = m or n
    grid = np.zeros((n, m), dtype=np.int64)
    grid[n // 2, m // 2] = value
    return grid


def seed_grid(n, m=None, values=range(0, 256), spacing=30):
    """Initial state of many seeds on a regular grid (Ch. 4).

    The default plants all 256 seeds, values 0..255; the value-0 seed is
    a valid seed but infertile (indistinguishable from the background)."""
    m = m or n
    grid = np.zeros((n, m), dtype=np.int64)
    vals = list(values)
    per_row = max(1, n // spacing)
    for idx, v in enumerate(vals):
        i = (idx // per_row) * spacing + spacing // 2
        j = (idx % per_row) * spacing + spacing // 2
        if i < n and j < m:
            grid[i, j] = v
    return grid


# ---------------------------------------------------------------------- #
# Game of Life (Section 4.5), with age coloring
# ---------------------------------------------------------------------- #
_LIFE_KERNEL = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]])


def life_step(alive):
    """One generation of Conway's Life on the torus. alive: bool array."""
    nbrs = convolve2d(alive.astype(np.int64), _LIFE_KERNEL,
                      mode="same", boundary="wrap")
    return (nbrs == 3) | (alive & (nbrs == 2))


def run_life(initial_alive, generations, max_age=256):
    """Evolve Life, tracking cell ages.  Returns (alive, age) where age is
    0 for dead cells and min(age, max_age) for live ones."""
    alive = initial_alive.astype(bool)
    age = alive.astype(np.int64)
    for _ in range(generations):
        new = life_step(alive)
        age = np.where(new, np.minimum(np.where(alive, age + 1, 1), max_age), 0)
        alive = new
    return alive, age


def random_life(n, m=None, p=0.3, rng=None):
    rng = rng or np.random.default_rng()
    m = m or n
    return rng.random((n, m)) < p


# ---------------------------------------------------------------------- #
# a small bestiary of famous Life patterns (Section 4.5)
# ---------------------------------------------------------------------- #
# patterns are lists of (row, col) offsets of live cells
BLOCK = [(0, 0), (0, 1), (1, 0), (1, 1)]
BLINKER = [(0, 0), (0, 1), (0, 2)]
GLIDER = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
R_PENTOMINO = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)]
GOSPER_GLIDER_GUN = [
    (4, 0), (5, 0), (4, 1), (5, 1),
    (4, 10), (5, 10), (6, 10), (3, 11), (7, 11), (2, 12), (8, 12),
    (2, 13), (8, 13), (5, 14), (3, 15), (7, 15), (4, 16), (5, 16),
    (6, 16), (5, 17),
    (2, 20), (3, 20), (4, 20), (2, 21), (3, 21), (4, 21), (1, 22),
    (5, 22), (0, 24), (1, 24), (5, 24), (6, 24),
    (2, 34), (3, 34), (2, 35), (3, 35),
]


def place(grid, pattern, top, left):
    """Set the cells of a pattern (list of (row, col) offsets) to alive
    on a boolean grid, with the pattern's bounding box anchored at
    (top, left).  Returns the grid for chaining."""
    for r, c in pattern:
        grid[(top + r) % grid.shape[0], (left + c) % grid.shape[1]] = True
    return grid


def pattern_grid(pattern, n, m=None, top=None, left=None):
    """A fresh n x m boolean grid containing one copy of the pattern,
    centered by default."""
    m = m or n
    rows = [r for r, _ in pattern]; cols = [c for _, c in pattern]
    h, w = max(rows) + 1, max(cols) + 1
    top = (n - h) // 2 if top is None else top
    left = (m - w) // 2 if left is None else left
    return place(np.zeros((n, m), dtype=bool), pattern, top, left)


# ---------------------------------------------------------------------- #
# Euler totient rule (Section 4.6)
# ---------------------------------------------------------------------- #
def totient_step(k=256):
    """C'_{ij} = sum of phi(C) over the 3x3 neighborhood, mod k
    (with phi(0) taken to be 0)."""
    phi = totient_table(k)
    ones = np.ones((3, 3), dtype=np.int64)

    def step(state):
        return np.mod(convolve2d(phi[state], ones,
                                 mode="same", boundary="wrap"), k)
    return step


def run_rule(initial, step, generations):
    state = np.asarray(initial, dtype=np.int64)
    for _ in range(generations):
        state = step(state)
    return state


# ---------------------------------------------------------------------- #
# image processing (Section 4.1's kernels applied as actual filters)
# ---------------------------------------------------------------------- #
EDGE_KERNEL = np.array([[ 0, -1,  0],
                        [-1,  4, -1],
                        [ 0, -1,  0]])       # Laplacian edge extractor

BLUR_KERNEL = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16.0   # Gaussian-like smoothing


def filter_image(img, kernel):
    """Ordinary (non-modular) 2-D convolution with periodic boundaries:
    exactly one generation of the corresponding CA, minus the mod-k
    reduction.  This is the image-processing use of the same
    machinery."""
    return convolve2d(img, kernel[::-1, ::-1], mode="same", boundary="wrap")


def demo_image(n=256, rng=None):
    """A synthetic grayscale test image (values 0..255): a gradient
    background with geometric shapes and mild noise -- enough structure
    to exhibit edge extraction and smoothing without external data."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    img = 60 + 80 * xx / n                                # gradient
    img[(yy - 0.32*n)**2 + (xx - 0.30*n)**2 < (0.16*n)**2] = 210   # disk
    img[int(0.55*n):int(0.85*n), int(0.55*n):int(0.85*n)] = 30     # square
    tri = (yy > 0.60*n) & (xx > 0.10*n) & (yy - xx < 0.52*n) \
          & (yy + xx < 1.15*n)
    img[tri] = 150                                                 # triangle
    img += rng.normal(0, 4, (n, n))                               # noise
    return np.clip(img, 0, 255)


def show_gray(img, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------- #
# plotting
# ---------------------------------------------------------------------- #
def show_state(state, k, ax=None, zero_color="white", title=None, crop=None):
    """Display a 2-D CA state.  crop=(size) shows only the central
    size x size window (the 'magnified detail' figures)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    img = state
    if crop:
        n, m = state.shape
        i0, j0 = (n - crop) // 2, (m - crop) // 2
        img = state[i0:i0 + crop, j0:j0 + crop]
    ax.imshow(img, cmap=rainbow_cmap(k, zero_color), vmin=0, vmax=k - 1,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def show_life(age, max_age=256, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.minimum(age, max_age), cmap=age_cmap(max_age),
              vmin=0, vmax=max_age, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
