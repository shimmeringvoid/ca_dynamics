"""Generate the geometric Sierpinski construction (S0..S3) and the
Pascal-triangle-mod-2 figure for the appendices."""
import sys, os
sys.path.insert(0, "/home/claude/ca_dynamics")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

FIG = "/home/claude/ca_dynamics/figures"
os.makedirs(FIG, exist_ok=True)


def sierpinski_triangles(level):
    """Return list of upward triangles (as vertex arrays) for S_level."""
    base = [np.array([(0, 0), (1, 0), (0.5, np.sqrt(3)/2)])]
    for _ in range(level):
        new = []
        for tri in base:
            a, b, c = tri
            ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
            new += [np.array([a, ab, ca]),
                    np.array([ab, b, bc]),
                    np.array([ca, bc, c])]
        base = new
    return base


fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, lvl in zip(axes, range(4)):
    for tri in sierpinski_triangles(lvl):
        ax.add_patch(Polygon(tri, closed=True, facecolor="black",
                             edgecolor="none"))
    ax.set_aspect("equal"); ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, np.sqrt(3)/2 + 0.02); ax.axis("off")
    ax.set_title(f"$S_{lvl}$")
fig.savefig(f"{FIG}/appC_sierpinski_construction.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print("appC_sierpinski_construction.png")


# Pascal's triangle mod 2 as a Sierpinski approximation
def pascal_mod2(rows):
    grid = np.zeros((rows, rows), dtype=int)
    grid[0, 0] = 1
    for i in range(1, rows):
        grid[i, 0] = 1
        for j in range(1, i + 1):
            grid[i, j] = (grid[i-1, j-1] + grid[i-1, j]) % 2
    # center each row for the triangular look
    img = np.zeros((rows, 2 * rows))
    for i in range(rows):
        for j in range(i + 1):
            img[i, rows - i + 2 * j] = grid[i, j]
    return img


fig, ax = plt.subplots(figsize=(7, 4))
ax.imshow(pascal_mod2(128), cmap="binary", interpolation="nearest")
ax.axis("off")
ax.set_title("Pascal's triangle mod 2 (128 rows)")
fig.savefig(f"{FIG}/appB_pascal_mod2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("appB_pascal_mod2.png")


# Euler totient function plot
from cadyn.numtheory import totient_table
phi = totient_table(200)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(range(1, 200), phi[1:200], ".", ms=3, color="navy")
ax.plot(range(1, 200), range(1, 200), "-", lw=0.5, color="0.7",
        label="$n-1$ (primes lie here)")
ax.set_xlabel("$n$"); ax.set_ylabel(r"$\varphi(n)$")
ax.set_title("Euler's totient function")
ax.legend()
fig.savefig(f"{FIG}/appD_totient.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("appD_totient.png")
