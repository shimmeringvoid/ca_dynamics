"""Regenerate the book's figures with the cadyn package.
Outputs PNG files into ../figures/ (also used by the LaTeX build)."""
import sys, os, time
sys.path.insert(0, "/home/claude/ca_dynamics")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cadyn import fds, ca1d, ca2d
from cadyn.ecosystem import Ecosystem

FIG = "/home/claude/ca_dynamics/figures"
os.makedirs(FIG, exist_ok=True)
t0 = time.time()

def save(name, fig=None, dpi=150):
    (fig or plt.gcf()).savefig(f"{FIG}/{name}.png", dpi=dpi,
                               bbox_inches="tight")
    plt.close("all")
    print(f"  {name}.png  ({time.time()-t0:5.1f}s)")

# ===================== Chapter 2 =========================================
print("Chapter 2: finite dynamical systems")
for name, sys_ in [("ch2_basin_mod17", fds.example_2_1()),
                   ("ch2_basin_mod37", fds.example_2_3()),
                   ("ch2_basin_cubic_mod28", fds.example_2_5()),
                   ("ch2_basin_ulam_mod20", fds.ulam_mod(20)),
                   ("ch2_basin_ulam_mod40", fds.ulam_mod(40)),
                   ("ch2_basin_ulam_mod47", fds.ulam_mod(47))]:
    sys_.plot_basin_field()
    save(name)

fig, ax = plt.subplots(figsize=(13, 11))
fds.example_2_4().plot_basin_field(ax=ax, node_size=70, font_size=7)
save("ch2_basin_mod221", dpi=200)

for name, sys_, tmax in [("ch2_entropy_mod37", fds.example_2_3(), 14),
                         ("ch2_entropy_mod221", fds.example_2_4(), 5),
                         ("ch2_entropy_cubic_mod28", fds.example_2_5(), 4),
                         ("ch2_entropy_ulam_mod47", fds.ulam_mod(47), 16)]:
    sys_.plot_entropy(tmax)
    save(name)

# ===================== Chapter 3 =========================================
print("Chapter 3: 1-dimensional CA")
init = np.zeros(21, dtype=int); init[10] = 1
hist = ca1d.run_additive(init, [1, 1, 1], 5, 136)
ca1d.plot_spacetime(hist, 5, zero_color="black", time_axis="right",
                    title="kernel [1 1 1], $k=5$, $n=21$; time left to right")
save("ch3_k5_n21_kernel111")

hist = ca1d.run_additive(ca1d.single_seed(512), [1, 0, 1], 2, 255)
ca1d.plot_spacetime(hist, 2, title="Sierpinski gasket: kernel [1 0 1], $k=2$")
save("ch3_sierpinski_k2")

for k in (3, 4, 5, 15, 17, 29, 30):
    hist = ca1d.run_additive(ca1d.single_seed(512), [1, 0, 1], k, 255)
    ca1d.plot_spacetime(hist, k, title=f"kernel [1 0 1], $k={k}$")
    save(f"ch3_kernel101_k{k}")

for k in (7, 8, 9, 56, 128):
    step = ca1d.sum_of_squares_step(k)
    hist = ca1d.run_rule(ca1d.single_seed(200), step, 100)
    ca1d.plot_spacetime(hist, k, title=f"sum of squares rule, $k={k}$")
    save(f"ch3_sumsq_k{k}")

rng = np.random.default_rng(7)
x0 = np.where(rng.random(512) < 0.5,
              rng.integers(1, 256, 512), 0)
hist = ca1d.run_rule(x0, ca1d.life_1d_step(rng), 300)
ca1d.plot_spacetime(hist, 256, time_axis="right",
                    title="1-D Game of Life (neighborhood $\\{\\pm1,\\pm3,\\pm4\\}$)")
save("ch3_life1d")

# ===================== Chapter 4 =========================================
print("Chapter 4: 2-dimensional CA")
final, snaps = ca2d.run_additive(ca2d.single_seed(500),
                                 ca2d.KERNEL_VON_NEUMANN, 256, 250,
                                 sample_at=(100, 250))
ca2d.show_state(snaps[100], 256, zero_color="black", crop=260,
                title="von Neumann kernel, generation 100")
save("ch4_vonneumann_gen100")
ca2d.show_state(snaps[250], 256, zero_color="black",
                title="von Neumann kernel, generation 250")
save("ch4_vonneumann_gen250")

seeds = ca2d.seed_grid(500, values=range(0, 256), spacing=30)
final, _ = ca2d.run_additive(seeds, ca2d.KERNEL_VON_NEUMANN, 256, 13)
ca2d.show_state(final, 256, zero_color="black",
                title="256 seeds (values 0-255), generation 13")
save("ch4_vonneumann_seeds_gen13")

final, snaps = ca2d.run_additive(ca2d.single_seed(500),
                                 ca2d.KERNEL_4_3, 256, 250,
                                 sample_at=(250,))
ca2d.show_state(snaps[250], 256, title="kernel of Example 4.3, generation 250")
save("ch4_kernel43_gen250")
ca2d.show_state(snaps[250], 256, crop=150,
                title="Example 4.3, generation 250 (detail)")
save("ch4_kernel43_gen250_detail")

# Game of Life
alive0 = ca2d.random_life(300, p=0.3, rng=np.random.default_rng(3))
ca2d.show_life(alive0.astype(int), title="Life: initial random state, $p=0.3$")
save("ch4_life_initial")
alive, age = ca2d.run_life(alive0, 500)
ca2d.show_life(age, title="Life after 500 generations (age-colored)")
save("ch4_life_gen500")

row = np.zeros((220, 220), dtype=bool)
row[110, 75:146] = True                       # a row of 71 live cells
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
gens = [1, 50, 100, 150, 200, 250, 300, 400]
a, g_prev = row.copy(), 0
age = row.astype(int)
for ax, g in zip(axes.flat, gens):
    a2, age = ca2d.run_life(a, g - g_prev)
    a, g_prev = a2, g
    ca2d.show_life(age, ax=ax, title=f"generation {g}")
fig.suptitle("Life from an initial row of 71 live cells")
save("ch4_life_row71", fig)

state = ca2d.run_rule(ca2d.single_seed(500), ca2d.totient_step(256), 250)

# A small bestiary: glider translating, and the Gosper glider gun
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
g = ca2d.pattern_grid(ca2d.GLIDER, 14)
a, age = g.copy(), g.astype(int)
for ax, gen in zip(axes, [0, 1, 2, 3, 4]):
    if gen:
        a, age = ca2d.run_life(a, 1)
    ca2d.show_life(age, ax=ax, title=f"gen {gen}")
fig.suptitle("A glider: it returns to its shape, shifted by (1,1), every 4 generations")
save("ch4_glider", fig)

gun = ca2d.pattern_grid(ca2d.GOSPER_GLIDER_GUN, 90, 130, top=5, left=5)
_, age = ca2d.run_life(gun, 120)
ca2d.show_life(age, title="Gosper glider gun after 120 generations")
save("ch4_gosper_gun")
ca2d.show_state(state, 256, title="Euler totient rule, generation 250")
save("ch4_totient_gen250")
ca2d.show_state(state, 256, crop=150,
                title="Euler totient rule, generation 250 (detail)")
save("ch4_totient_gen250_detail")

# ===================== Chapter 5 =========================================
print("Chapter 5: ecosystem")
eco = Ecosystem(n=200, p_prey=0.212, p_pred=0.029,
                rng=np.random.default_rng(11))
eco.show(title="Initial ecosystem state (green=plants, yellow=prey, red=predators)")
save("ch5_eco_initial")
eco.run(2000)
eco.show(title="Ecosystem after 2000 generations (purple = devoid of life)")
save("ch5_eco_gen2000")
eco.plot_populations(sample_every=4)
save("ch5_eco_populations")
eco.plot_phase_portrait(sample_every=4)
save("ch5_eco_phase")

print("done with core figures.")

# ---- expansion-section figures -----------------------------------------
# (Chapter 3 entropy, Chapter 4 image processing, Chapter 5 extinction and
#  survival sweep, Chapter 6 Wolfram classes and Rule 110).  Kept in a
#  separate module so it can also be run on its own.
print("\ngenerating expansion figures...")
import runpy
runpy.run_path(os.path.join(os.path.dirname(__file__),
                            "make_expansion_figures.py"),
               run_name="__main__")

