"""
Evolutionary strategy dynamics: the mean-field model of Chapter 7.

This module implements the first (mean-field) experiment of Espericueta,
"Veridical Perception & Evolutionary Dynamics" (UCSC), which asks
Hoffman's question --- does natural selection favor veridical (accurate)
perception? --- in a deliberately minimal setting.

A single species carries several competing *strategies*, each with a
fixed probability of success p_i (its fitness).  The population
proportion vector s(t) (on the probability simplex) evolves by a
time-varying, row-stochastic transition built from those fitnesses.  A
cell keeps its own strategy with probability p_i, otherwise adopts a
neighbor's strategy; treating the current proportions as the
neighbor-type probabilities gives the update

    s(t+1) = W(t) s(t),   W(t)_{ij} = p_i [i=j] + (1 - p_j) s_i .

The central results reproduced and checked here:

  * the Lemma: W(t) s(t) has the closed Hadamard-product form
        s(t+1) = ( p (.) I  +  1 ((1 - p) (.) s)^T ) (.) s
    valid in any dimension (`step`, `step_matrix`, and
    `verify_lemma` all agree);
  * the Theorem (empirical): if one strategy has the strictly greatest
    fitness, its proportion tends to 1 and all others to 0
    (`dominant_strategy_wins` tests this over random simplex starts).

Here "(.)" denotes the elementwise (Hadamard) product.
"""

import numpy as np


# ---------------------------------------------------------------------- #
# the transition, three equivalent ways
# ---------------------------------------------------------------------- #
def transition_matrix(p, s):
    """The time-varying transition matrix W(t) for fitness vector p and
    current proportions s.  Column j sums to 1 (column-stochastic), so
    that s(t+1) = W s keeps the proportions on the simplex.

        W_{ij} = p_i        if i == j
                 (1 - p_j) s_i   otherwise... 

    written compactly so the diagonal carries the 'keep your strategy'
    term and every column j distributes its 'switch' mass (1 - p_j)
    across strategies in proportion to s."""
    p = np.asarray(p, float)
    s = np.asarray(s, float).ravel()
    n = len(p)
    W = np.outer(s, (1.0 - p))            # W_{ij} = s_i (1 - p_j)
    W[np.diag_indices(n)] = p + s * (1.0 - p)   # diagonal: p_i + s_i(1-p_i)
    return W


def step_matrix(p, s):
    """One generation via the explicit matrix product W(t) s(t)."""
    return transition_matrix(p, s) @ np.asarray(s, float).ravel()


def step(p, s):
    """One generation via the closed-form Hadamard expression of the
    Lemma --- no n-by-n matrix is formed:

        s(t+1) = ( p (.) I + 1 ((1-p) (.) s)^T ) (.) s
               = p (.) s  +  s * < (1 - p), s > .

    (The bracket is the scalar dot product of (1-p) with s.)"""
    p = np.asarray(p, float)
    s = np.asarray(s, float).ravel()
    switch_mass = float(np.dot(1.0 - p, s))     # total 'switching' weight
    return p * s + s * switch_mass


def run(p, s0, generations, renormalize=True):
    """Iterate the mean-field dynamics; return array (generations+1, n)
    of population-proportion vectors.

    The update is exactly column-stochastic on the simplex, so the true
    dynamics conserve the total proportion.  Finite-precision iteration,
    however, drifts slightly off the simplex, and the map amplifies that
    drift (a vertex of the simplex is an unstable-in-scale fixed point of
    the raw arithmetic).  We therefore renormalize each generation, which
    is what 'population *proportions*' means in any case.  Set
    renormalize=False to see the raw map."""
    p = np.asarray(p, float)
    s = np.asarray(s0, float).ravel()
    s = s / s.sum()
    hist = np.empty((generations + 1, len(s)))
    hist[0] = s
    for t in range(generations):
        s = step(p, s)
        if renormalize:
            tot = s.sum()
            if tot > 0:
                s = s / tot
        hist[t + 1] = s
    return hist


# ---------------------------------------------------------------------- #
# the Lemma and the Theorem, made checkable
# ---------------------------------------------------------------------- #
def verify_lemma(p, s, tol=1e-12):
    """Confirm the three formulations agree for given p, s:
    matrix product, Hadamard closed form, and a literal transcription of
    the Lemma's right-hand side  (p (.) I + 1 ((1-p)(.)s)^T) (.) s."""
    p = np.asarray(p, float)
    s = np.asarray(s, float).ravel()
    n = len(p)
    a = step_matrix(p, s)
    b = step(p, s)
    # literal Lemma RHS:  ( p (.) I  +  1 ((1-p)(.)s)^T ) (.) s .
    # The bracketed matrix M has diagonal p_i and every row equal to the
    # row vector ((1-p) (.) s); Hadamard-multiplying by the column s
    # scales row i by s_i, and the implicit matrix-vector sum is the row
    # sum.  (Broadcasting s down the columns is the crux.)
    row = (1.0 - p) * s
    M = np.outer(np.ones(n), row)
    M[np.diag_indices(n)] += p
    c = (M * s[:, np.newaxis]).sum(axis=1)
    return np.allclose(a, b, atol=tol) and np.allclose(a, c, atol=tol)


def dominant_strategy_wins(p, s0, generations=2000, tol=1e-6):
    """Return True if, starting from s0, the strategy of strictly
    greatest fitness ends with proportion within tol of 1.  (An empirical
    test of the Theorem for one initial condition.)"""
    p = np.asarray(p, float)
    order = np.argsort(p)
    if p[order[-1]] - p[order[-2]] <= 0:
        raise ValueError("Theorem assumes a unique maximum-fitness strategy")
    final = run(p, s0, generations)[-1]
    return final[order[-1]] > 1.0 - tol


def nash_proportions(p):
    """The interior fixed point predicted for the *combined* two-species
    (bird/cat) subsystem is estimated from simulation elsewhere; for the
    single-species model the only fixed points on the simplex are the
    vertices e_k, so this returns the winning vertex for a unique max."""
    p = np.asarray(p, float)
    e = np.zeros_like(p)
    e[int(np.argmax(p))] = 1.0
    return e


# ---------------------------------------------------------------------- #
# the paper's worked example
# ---------------------------------------------------------------------- #
#: fitness (probability-of-success) vector used throughout Chapter 7:
#: the R (random), P (probabilistic) and V (veridical) strategies.
FITNESS_RPV = np.array([0.6, 0.75, 0.9])

STRATEGY_LABELS = ("R (random)", "P (probabilistic)", "V (veridical)")


def plot_run(p, s0, generations, labels=None, ax=None, title=None):
    """Population-proportion curves over time (Figure 1 style)."""
    import matplotlib.pyplot as plt
    hist = run(p, s0, generations)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    labels = labels or [f"strategy {i}" for i in range(hist.shape[1])]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown"]
    for i in range(hist.shape[1]):
        ax.plot(hist[:, i], color=colors[i % len(colors)], lw=2,
                label=labels[i])
    ax.set_xlabel("generation"); ax.set_ylabel("population proportion")
    ax.set_ylim(-0.03, 1.03); ax.legend(loc="center right")
    if title:
        ax.set_title(title)
    return ax


def plot_simplex_trajectory(p, starts, generations, labels=None, ax=None):
    """Project one or more trajectories onto the 2-simplex (a triangle),
    the natural home of a 3-strategy proportion vector.  Vertices are the
    pure strategies; the interior is every mixture."""
    import matplotlib.pyplot as plt
    # corners of an equilateral triangle for strategies 0,1,2
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    to_xy = lambda s: s @ corners
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.4))
    tri = np.vstack([corners, corners[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="0.6", lw=1)
    lab = labels or STRATEGY_LABELS
    for i, c in enumerate(corners):
        dx = -0.04 if c[0] < 0.5 else (0.04 if c[0] > 0.5 else 0)
        ax.text(c[0] + dx, c[1] + (0.03 if c[1] > 0.1 else -0.06), lab[i],
                ha="center", fontsize=9)
    for s0 in starts:
        xy = np.array([to_xy(row) for row in run(p, s0, generations)])
        ax.plot(xy[:, 0], xy[:, 1], lw=1.4)
        ax.plot(*xy[0], "o", ms=5, color="0.3")
        ax.plot(*xy[-1], "*", ms=13, color="crimson")
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title("Strategy simplex: every start flows to the fittest vertex")
    return ax
