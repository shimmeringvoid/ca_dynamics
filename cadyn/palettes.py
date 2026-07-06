"""Color maps for displaying CA states, matching the spirit of the
original book figures: state 0 is white (or black), all other states
are drawn from a bright rainbow."""

import numpy as np
from matplotlib.colors import ListedColormap, hsv_to_rgb


def rainbow_cmap(k, zero_color="white"):
    """Colormap with k entries: index 0 -> zero_color, indices 1..k-1
    sweep the rainbow (violet ... red)."""
    colors = np.ones((k, 3))
    if k > 1:
        h = np.linspace(0.83, 0.0, k - 1)          # violet -> red
        s = np.full(k - 1, 0.95)
        v = np.full(k - 1, 0.95)
        colors[1:] = hsv_to_rgb(np.stack([h, s, v], axis=1))
    colors[0] = {"white": (1, 1, 1), "black": (0, 0, 0)}[zero_color]
    return ListedColormap(colors)


def age_cmap(max_age=256):
    """Game-of-Life age coloring: dead = white; newborn cerise, then red,
    orange, yellow, green, blue, and finally violet for 'ancient' cells."""
    anchors = [
        (0.00, (1.00, 0.10, 0.55)),   # cerise (newborn)
        (0.05, (1.00, 0.00, 0.00)),   # red
        (0.15, (1.00, 0.55, 0.00)),   # orange
        (0.30, (1.00, 1.00, 0.00)),   # yellow
        (0.50, (0.00, 0.80, 0.00)),   # green
        (0.75, (0.00, 0.20, 1.00)),   # blue
        (1.00, (0.60, 0.00, 0.80)),   # violet (ancient)
    ]
    xs = np.linspace(0, 1, max_age)
    pts = np.array([a[0] for a in anchors])
    cols = np.array([a[1] for a in anchors])
    rgb = np.stack([np.interp(xs, pts, cols[:, c]) for c in range(3)], axis=1)
    table = np.vstack([[1.0, 1.0, 1.0], rgb])      # index 0 = dead = white
    return ListedColormap(table)
