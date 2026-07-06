# Cellular Automata Dynamics — Second (Python) Edition

A modern Python re-implementation of the software and text of
*Cellular Automata Dynamics: Explorations in Parallel Processing*
(Rafael Espericueta, 1997), which originally shipped with four C++
programs for Windows 95.

<!-- Replace USERNAME with your GitHub username once the repo exists. -->
[![tests](https://github.com/USERNAME/ca-dynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/USERNAME/ca-dynamics/actions/workflows/tests.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/USERNAME/ca-dynamics/main?urlpath=lab/tree/notebooks)
[![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-lightgrey.svg)](LICENSE-TEXT)

## Quick start

```bash
# use the library
pip install git+https://github.com/USERNAME/ca-dynamics

# or clone and work through the book
git clone https://github.com/USERNAME/ca-dynamics
cd ca-dynamics
pip install -e ".[notebooks]"
jupyter lab            # open notebooks/01_what_are_cellular_automata.ipynb
```

**Run in your browser, zero install:** click the Binder badge above, or
open any chapter in Google Colab — replace `USERNAME` in this link:
`https://colab.research.google.com/github/USERNAME/ca-dynamics/blob/main/notebooks/01_what_are_cellular_automata.ipynb`


## What's here

```
ca_dynamics/
├── cadyn/                 the Python package (replaces the C++ programs)
│   ├── fds.py             0-D CA / finite dynamical systems  (was FiniteDynamicalSystems)
│   ├── ca1d.py            1-D CA                             (was CA1dim)
│   ├── ca2d.py            2-D CA + Game of Life              (was CA2dim)
│   ├── ecosystem.py       3-species predator/prey CA         (was CAecosystem)
│   ├── evolution.py       mean-field strategy dynamics (Ch. 7, veridical perception)
│   ├── numtheory.py       Euler totient, Ulam/Collatz helpers
│   └── palettes.py        color maps matching the original figures
├── examples/
│   ├── verify.py                 checks output against the book's stated results
│   ├── make_figures.py           regenerates every chapter figure
│   └── make_appendix_figures.py  Sierpinski construction, Pascal mod 2, totient
├── notebooks/                interactive JupyterLab companion, one per chapter
│   ├── 01_what_are_cellular_automata.ipynb  ...  07_evolutionary_dynamics.ipynb
│   └── build_*.py            scripts that generate the notebooks
├── figures/               generated PNGs (populated by the scripts above)
└── latex/
    ├── book.tex           the book, LaTeX source
    └── chapters/          one file per chapter + appendices + bibliography
```

## Interactive notebooks

A JupyterLab notebook accompanies each chapter in `notebooks/`, wrapping
explanatory markdown around live `cadyn` calls, and ending with
open-ended exercises. They ship already executed (figures visible without
running). See `notebooks/README.md`.

```bash
pip install jupyterlab
jupyter lab      # open notebooks/01_what_are_cellular_automata.ipynb
```

## Requirements

Python 3.9+ with:

```
numpy  scipy  matplotlib  networkx
```

```bash
pip install numpy scipy matplotlib networkx
```

A LaTeX distribution (TeX Live) is needed only to rebuild the PDF.

## Quick tour

```python
from cadyn import fds, ca1d, ca2d
from cadyn.ecosystem import Ecosystem
import matplotlib.pyplot as plt

# --- 0-D: basin of attraction field for x^2 + 1 (mod 37) ---
s = fds.example_2_3()
print(s.trajectory(0))          # 0, 1, 2, 5, 26, 11 (fixed point 11)
s.plot_basin_field(); plt.show()
s.plot_entropy(14); plt.show()

# --- 1-D: the Sierpinski gasket, kernel [1 0 1] mod 2 ---
hist = ca1d.run_additive(ca1d.single_seed(512), [1, 0, 1], k=2, generations=255)
ca1d.plot_spacetime(hist, k=2); plt.show()

# --- 1-D: Wolfram's elementary CA (all 256 rules by number) ---
hist = ca1d.run_elementary(ca1d.single_seed(601), rule=30, generations=300)
ca1d.plot_spacetime(hist, k=2); plt.show()   # Rule 30's pseudo-randomness

# --- 2-D: Conway's Game of Life, age-colored ---
alive0 = ca2d.random_life(200, p=0.3)
alive, age = ca2d.run_life(alive0, 500)
ca2d.show_life(age); plt.show()

# --- 2-D: the Gosper glider gun (grows without bound) ---
gun = ca2d.pattern_grid(ca2d.GOSPER_GLIDER_GUN, 90, 130, top=5, left=5)
alive, age = ca2d.run_life(gun, 120)
ca2d.show_life(age); plt.show()

# --- ecosystem: predator/prey oscillations ---
eco = Ecosystem(n=200, p_prey=0.212, p_pred=0.029)
eco.run(2000)
eco.plot_populations(); plt.show()
eco.plot_phase_portrait(); plt.show()

# --- evolution: does selection favor accurate perception? (Chapter 7) ---
from cadyn import evolution as ev
p = ev.FITNESS_RPV                      # [0.6, 0.75, 0.9] for R, P, V
ev.verify_lemma(p, [0.5, 0.3, 0.2])     # True: three formulations agree
ev.run(p, [0.5, 0.3, 0.2], 200)[-1]     # -> [0, 0, 1]: veridical wins
ev.plot_simplex_trajectory(p, [[0.5,0.3,0.2],[0.1,0.8,0.1]], 200); plt.show()
```

## Reproducing the book

```bash
python examples/verify.py               # sanity checks vs. 1997 results
python examples/make_figures.py         # all figures (chains the expansion set) -> figures/
python examples/make_appendix_figures.py
cd latex && pdflatex book.tex && pdflatex book.tex && pdflatex book.tex   # -> book.pdf
```

(`make_figures.py` calls `make_expansion_figures.py` automatically; run
the latter alone if you only need the new-section figures. A third
`pdflatex` pass settles the table of contents.)

## What changed from the first edition

* **Language:** C++/Win95 → pure Python (NumPy/SciPy/Matplotlib/NetworkX),
  cross-platform and open source.
* **Corrections:** a handful of typos and copy-paste errors in the 1997
  text are fixed and flagged in shaded boxes; see
  *Appendix — Summary of Corrections* in the PDF.
* **Expansions:** short "[2026 update]" notes bring the Collatz status,
  Life's proven universality, Wolfram's *A New Kind of Science*, and
  spatial-ecology results up to date; **Chapter 6** (Wolfram's four
  classes, computational irreducibility, and the philosophical
  discussion the original outlined) is now written out in full.  New
  sections add: the elementary-CA rule numbering and Rule 30/90/110
  (§3.7), one-dimensional spatial entropy comparing four rules (§3.9),
  an image-processing interlude applying the chapter's kernels as edge
  and smoothing filters (§4.3), and an ecosystem extinction run plus a
  survival phase diagram over predator parameters (Chapter 5).  A new
  **Chapter 7** ports the author's later study *Veridical Perception and
  Evolutionary Dynamics*: the mean-field strategy model is implemented
  and its Lemma and (empirical) Theorem are verified, while the spatial
  movement-and-strategy experiment is described in prose and connected to
  Hoffman's Interface Theory of Perception.

## New in the package (beyond the original four programs)

* `ca1d.run_elementary(initial, rule, generations)` — all 256 Wolfram
  elementary rules by number (Rule 90 reproduces the Sierpinski gasket,
  Rule 30 the pseudo-random generator, Rule 110 the universal one).
* `ca1d.spatial_entropy(history, k)` — Shannon entropy of each
  generation across the lattice.
* `ca2d.filter_image` / `EDGE_KERNEL` / `BLUR_KERNEL` / `demo_image` —
  the image-processing side of the convolution machinery.
* `ecosystem.Ecosystem.run_until(...)` and `.extinct()` — early-stop
  runs and extinction detection for parameter sweeps. New
  **Section 3.8** covers Wolfram's 256 elementary CA (Rules 30, 90,
  110, ...), and Chapter 4 now includes a bestiary of Life patterns
  (glider, blinker, block, R-pentomino, Gosper glider gun) shipped as
  reusable data in `cadyn.ca2d`.

## License

This project is released under two licenses, one for the software and one
for the book:

- **Software** — the `cadyn` package, the scripts in `examples/` and
  `notebooks/`, and the tests — is licensed under the **MIT License**
  (see [`LICENSE`](LICENSE)). Use it freely, including in your own
  projects, keeping the copyright notice.
- **Book** — the text and figures in `latex/` and `figures/`, and the
  rendered PDF — is licensed under **Creative Commons Attribution 4.0
  International (CC BY 4.0)** (see [`LICENSE-TEXT`](LICENSE-TEXT)). Share,
  adapt, translate, and reuse with attribution.

Copyright © 1997, 2026 Rafael Espericueta. In the spirit of the original:
please experiment, extend, and above all have fun.

Suggested citation:

> Rafael Espericueta, *Cellular Automata Dynamics: Explorations in
> Parallel Processing* (Second Edition, with software in Python), 2026.
