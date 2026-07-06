"""Figures for the second-edition expansion sections:

  * ch3_entropy_comparison   spatial entropy vs. time for four rules
  * ch4_image_processing     the Section 4.1 kernels applied to an image
  * ch5_extinction           an imprudent parameter choice -> extinction
  * ch5_survival_sweep       survival phase diagram over (b_pred, d_pred)
  * ch6_wolfram_classes      the four Wolfram classes (rules 254/108/30/110)
  * ch6_rule110_detail       Rule 110 particles from a random start
"""
import sys, os, time
sys.path.insert(0, "/home/claude/ca_dynamics")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cadyn import ca1d, ca2d
from cadyn.ecosystem import Ecosystem

FIG = "/home/claude/ca_dynamics/figures"
os.makedirs(FIG, exist_ok=True)
t0 = time.time()

def save(name, fig=None, dpi=150):
    (fig or plt.gcf()).savefig(f"{FIG}/{name}.png", dpi=dpi,
                               bbox_inches="tight")
    plt.close("all")
    print(f"  {name}.png  ({time.time()-t0:5.1f}s)")


# ---- Chapter 3: spatial entropy comparison ------------------------------
print("ch3 expansion")
rng = np.random.default_rng(5)
n, gens, k = 400, 300, 8
random_init = rng.integers(0, k, n)

runs = {
    "$f\\equiv 0$ (freeze)": ca1d.run_rule(
        random_init, lambda x: np.zeros_like(x), gens),
    "kernel [1 1 1] mod 8 (additive)": ca1d.run_additive(
        random_init, [1, 1, 1], k, gens),
    "sum of squares mod 8": ca1d.run_rule(
        random_init, ca1d.sum_of_squares_step(k), gens),
    "1-D Life (Sec. 3.7)": ca1d.run_rule(
        np.where(rng.random(n) < 0.5, rng.integers(1, k, n), 0),
        ca1d.life_1d_step(rng, n_colors=k), gens),
}
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["0.4", "tab:blue", "tab:orange", "tab:red"]
for (label, hist), c in zip(runs.items(), colors):
    ax.plot(ca1d.spatial_entropy(hist, k), lw=1.8, color=c, label=label)
ax.axhline(np.log2(k), color="0.85", lw=0.8)
ax.text(gens, np.log2(k), " $\\log_2 8 = 3$", va="center", fontsize=8,
        color="0.4")
ax.set_xlabel("generation"); ax.set_ylabel("spatial entropy $H_t$ (bits)")
ax.set_ylim(-0.1, 3.3); ax.legend(loc="center right", fontsize=9)
save("ch3_entropy_comparison")

# ---- Chapter 3: elementary CA figures (Section 3.8) ---------------------
# Rule 30 from a single seed: the pseudo-random workhorse
h = ca1d.run_elementary(ca1d.single_seed(601), 30, 300)
ca1d.plot_spacetime(h, 2)
save("ch3_rule30")

# one representative of each Wolfram class, single-seed
elem = [(250, "Rule 250 (Class 1)"), (90, "Rule 90 (Class 2)"),
        (30, "Rule 30 (Class 3)"), (110, "Rule 110 (Class 4)")]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (rule, label) in zip(axes.flat, elem):
    h = ca1d.run_elementary(ca1d.single_seed(301), rule, 150)
    ax.imshow(h, cmap="binary", interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(label, fontsize=11)
save("ch3_elementary_classes", fig)

# ---- Chapter 4: image processing ----------------------------------------
print("ch4 expansion")
img = ca2d.demo_image(256)
edges = ca2d.filter_image(img, ca2d.EDGE_KERNEL)
smooth = img
for _ in range(6):
    smooth = ca2d.filter_image(smooth, ca2d.BLUR_KERNEL)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
ca2d.show_gray(img, ax=axes[0], title="input image")
ca2d.show_gray(np.abs(edges), ax=axes[1],
               title="edge extraction (one pass, Laplacian kernel)")
ca2d.show_gray(smooth, ax=axes[2],
               title="smoothing (6 iterations, kernel $[1\\,2\\,1]^T[1\\,2\\,1]/16$)")
save("ch4_image_processing", fig)

# ---- Chapter 5: an extinction run ----------------------------------------
print("ch5 expansion")
# imprudent parameters: voracious, short-lived predators -> wild transient
eco_bad = Ecosystem(n=120, p_prey=0.212, p_pred=0.029,
                    b_pred=0.9, d_pred=0.4, g=0.08,
                    rng=np.random.default_rng(4))
ran = eco_bad.run_until(2000)
H = np.array(eco_bad.history, dtype=float)
fig, ax = plt.subplots(figsize=(8.5, 4))
t = np.arange(len(H))
ax.plot(t, H[:, 0], color="green", lw=1.8, label="plants")
ax.plot(t, H[:, 1], color="gold", lw=1.8, label="prey")
ax.plot(t, H[:, 2], color="red", lw=1.8, label="predators")
which = " & ".join(eco_bad.extinct()) or "none"
ax.set_title(f"an imprudent parameter choice: extinction of {which} "
             f"at generation {ran}")
ax.set_xlabel("generation"); ax.set_ylabel("population")
ax.legend()
save("ch5_extinction")
print(f"    extinction of {which} at generation {ran}")

# ---- Chapter 5: survival phase diagram -----------------------------------
b_preds = np.linspace(0.1, 0.9, 9)
d_preds = np.linspace(0.05, 0.45, 9)
GENS, TRIALS = 400, 2
survival = np.zeros((len(d_preds), len(b_preds)))
for i, dp in enumerate(d_preds):
    for j, bp in enumerate(b_preds):
        alive = 0
        for trial in range(TRIALS):
            eco = Ecosystem(n=60, p_prey=0.212, p_pred=0.029,
                            b_pred=bp, d_pred=dp,
                            rng=np.random.default_rng(100*i + 10*j + trial))
            eco.run_until(GENS)
            if not eco.extinct():
                alive += 1
        survival[i, j] = alive / TRIALS
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(survival, origin="lower", cmap="RdYlGn", vmin=0, vmax=1,
               extent=[b_preds[0], b_preds[-1], d_preds[0], d_preds[-1]],
               aspect="auto", interpolation="nearest")
ax.set_xlabel("predation efficiency $b_{\\mathrm{pred}}$")
ax.set_ylabel("predator death rate $d_{\\mathrm{pred}}$")
ax.set_title(f"fraction of runs with all species surviving "
             f"{GENS} generations")
fig.colorbar(im, ax=ax, label="survival fraction")
save("ch5_survival_sweep", fig)

# ---- Chapter 6: the four Wolfram classes ----------------------------------
print("ch6 expansion")
rng = np.random.default_rng(0)
n6, g6 = 300, 150
init6 = rng.integers(0, 2, n6)
classes = [(160, "Class 1 (homogeneous): Rule 160"),
           (108, "Class 2 (periodic): Rule 108"),
           (30,  "Class 3 (chaotic): Rule 30"),
           (110, "Class 4 (complex): Rule 110")]
fig, axes = plt.subplots(2, 2, figsize=(11, 11 * g6 / n6 * 1.18))
for ax, (rule, label) in zip(axes.flat, classes):
    hist = ca1d.run_elementary(init6, rule, g6)
    ax.imshow(hist, cmap="binary", interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=11)
save("ch6_four_classes")

hist = ca1d.run_elementary(rng.integers(0, 2, 500), 110, 400)
ca1d.plot_spacetime(hist, 2)
save("ch6_rule110_detail")

# ---- Chapter 7: evolutionary strategy dynamics --------------------------
print("ch7 expansion")
from cadyn import evolution as ev
p = ev.FITNESS_RPV
L = ev.STRATEGY_LABELS

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
ev.plot_run(p, [1/3, 1/3, 1/3], 200, L, ax=axes[0],
            title="equal start [1/3, 1/3, 1/3]")
ev.plot_run(p, [0.5, 0.3, 0.2], 200, L, ax=axes[1],
            title="unequal start [0.5, 0.3, 0.2]")
fig.suptitle("Mean-field strategy dynamics: veridical (V) drives R and P extinct")
save("ch7_meanfield_runs", fig)

rng2 = np.random.default_rng(1)
starts = [rng2.dirichlet(np.ones(3)) for _ in range(12)] + \
         [[1/3, 1/3, 1/3], [0.5, 0.3, 0.2]]
ev.plot_simplex_trajectory(p, starts, 200)
save("ch7_simplex")

p_pen = np.array([0.6, 0.75, 0.55])       # veridical fitness cut by a time cost
fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
ev.plot_run(p, [1/3, 1/3, 1/3], 200, L, ax=axes[0],
            title="no time penalty: V wins")
ev.plot_run(p_pen, [1/3, 1/3, 1/3], 200,
            ["R (random)", "P (probabilistic)", "V (veridical, penalized)"],
            ax=axes[1], title="V penalized to 0.55: now P wins")
fig.suptitle("A time penalty on veridical perception reverses the outcome")
save("ch7_penalty", fig)

print("done.")
