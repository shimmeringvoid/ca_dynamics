"""
Three-species CA ecosystem (replaces the 1997 program ``CAecosystem``).

Cells on a torus hold one of four states:

    0 EMPTY  (purple in the original Figure 2 -- 'devoid of life')
    1 PLANT  (green)
    2 PREY   (yellow)
    3 PREDATOR (red)

Stochastic, synchronous update rules (all neighbor counts use the
8-cell Moore neighborhood):

  * a PLANT with at least one PREY neighbor is eaten -- it becomes PREY
    with probability 1 - (1 - b_prey)^(#prey neighbors);
  * a PREY with at least one PREDATOR neighbor is eaten -- it becomes
    PREDATOR with probability 1 - (1 - b_pred)^(#predator neighbors);
  * PREY die of other causes with probability d_prey (-> EMPTY);
  * PREDATORS starve/die with probability d_pred (-> EMPTY);
  * an EMPTY cell sprouts a PLANT with probability
    1 - (1 - g)^(#plant neighbors), or spontaneously with prob g0.

This produces the classic out-of-phase predator/prey oscillations that
generalize the Volterra-Lotka dynamics, damping toward a noisy
equilibrium -- and, with imprudent parameter choices, extinction.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d

EMPTY, PLANT, PREY, PREDATOR = 0, 1, 2, 3

_MOORE = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

ECO_CMAP = ListedColormap([(0.55, 0.0, 0.6),    # empty  -> purple
                           (0.0, 0.85, 0.0),    # plant  -> green
                           (1.0, 1.0, 0.0),     # prey   -> yellow
                           (1.0, 0.0, 0.0)])    # predator -> red


class Ecosystem:
    def __init__(self, n=200, m=None, p_prey=0.212, p_pred=0.029,
                 b_prey=0.30, b_pred=0.30, d_prey=0.06, d_pred=0.10,
                 g=0.35, g0=0.0005, rng=None):
        self.rng = rng or np.random.default_rng(0)
        m = m or n
        r = self.rng.random((n, m))
        self.grid = np.full((n, m), PLANT, dtype=np.int8)
        self.grid[r < p_prey + p_pred] = PREY
        self.grid[r < p_pred] = PREDATOR
        self.params = dict(b_prey=b_prey, b_pred=b_pred,
                           d_prey=d_prey, d_pred=d_pred, g=g, g0=g0)
        self.history = [self.populations()]

    # ------------------------------------------------------------------ #
    def populations(self):
        g = self.grid
        return (int((g == PLANT).sum()), int((g == PREY).sum()),
                int((g == PREDATOR).sum()))

    def extinct(self):
        """Return the list of animal species (of 'prey', 'predator')
        with zero population."""
        _, prey, pred = self.populations()
        out = []
        if prey == 0:
            out.append("prey")
        if pred == 0:
            out.append("predator")
        return out

    def run_until(self, generations, stop_on_extinction=True):
        """Run for at most `generations` steps, optionally stopping as
        soon as any animal species goes extinct.  Returns the number of
        generations actually run."""
        for t in range(generations):
            if stop_on_extinction and self.extinct():
                return t
            self.step()
        return generations

    def _count(self, species):
        return convolve2d((self.grid == species).astype(np.int64), _MOORE,
                          mode="same", boundary="wrap")

    def step(self):
        p, rng, g = self.params, self.rng, self.grid
        n_plant = self._count(PLANT)
        n_prey = self._count(PREY)
        n_pred = self._count(PREDATOR)
        r = rng.random(g.shape)
        new = g.copy()

        # predation / grazing (state promotions)
        eat_prey = (g == PREY) & (r < 1 - (1 - p["b_pred"]) ** n_pred)
        eat_plant = (g == PLANT) & (r < 1 - (1 - p["b_prey"]) ** n_prey)
        new[eat_prey] = PREDATOR
        new[eat_plant] = PREY

        # deaths (evaluated with fresh randomness)
        r2 = rng.random(g.shape)
        new[(new == PREY) & (r2 < p["d_prey"])] = EMPTY
        new[(new == PREDATOR) & (r2 < p["d_pred"])] = EMPTY

        # plant growth into empty cells
        r3 = rng.random(g.shape)
        sprout = (new == EMPTY) & (
            (r3 < 1 - (1 - p["g"]) ** n_plant) | (r3 < p["g0"]))
        new[sprout] = PLANT

        self.grid = new
        self.history.append(self.populations())

    def run(self, generations, sample_every=1):
        for t in range(generations):
            self.step()
        return np.array(self.history[::sample_every])

    # ------------------------------------------------------------------ #
    def show(self, ax=None, title=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(self.grid, cmap=ECO_CMAP, vmin=0, vmax=3,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title)
        return ax

    def plot_populations(self, ax=None, sample_every=4):
        """Normalized population curves (Figure 3 / Figure 5 style)."""
        H = np.array(self.history)[::sample_every].astype(float)
        H /= H.max(axis=0, keepdims=True).clip(min=1)
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        t = np.arange(len(H)) * sample_every
        ax.plot(t, H[:, 0], color="green", lw=2, label="plants")
        ax.plot(t, H[:, 1], color="gold", lw=2, label="prey")
        ax.plot(t, H[:, 2], color="red", lw=2, label="predators")
        ax.set_xlabel("generation"); ax.set_ylabel("normalized population")
        ax.legend(loc="upper right")
        return ax

    def plot_phase_portrait(self, ax=None, sample_every=4,
                            elev=22, azim=-55):
        """3-D phase portrait (Figure 4 style): one point per sample,
        coordinates = the three population counts."""
        H = np.array(self.history)[::sample_every].astype(float)
        if ax is None:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(projection="3d")
        ax.plot(H[:, 0], H[:, 1], H[:, 2], color="black", lw=0.6)
        ax.set_xlabel("plants", color="green")
        ax.set_ylabel("prey", color="goldenrod")
        ax.set_zlabel("predators", color="red")
        ax.view_init(elev=elev, azim=azim)
        return ax
