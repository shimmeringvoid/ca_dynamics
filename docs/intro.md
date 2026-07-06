# Cellular Automata Dynamics

*Explorations in Parallel Processing* — Second Edition, with software in Python

by **Rafael Espericueta**, Professor of Mathematics Emeritus, Bakersfield College

---

This is the interactive companion to the book. Each chapter is a live
notebook: read the explanation, run the code, change it, and watch what
happens. Everything is powered by the [`cadyn`](https://github.com/USERNAME/ca-dynamics)
Python package, a modern re-implementation of the software that
accompanied the 1997 first edition.

## The chapters

1. **What Are Cellular Automata?** — one runnable taste of each world.
2. **Zero-Dimensional CA** — finite dynamical systems: trajectories,
   basins of attraction, and entropy.
3. **One-Dimensional CA** — convolution-kernel rules, the Sierpiński
   gasket, and a bridge to signal processing.
4. **Two-Dimensional CA** — mandalas, image filters, and Conway's Game
   of Life.
5. **Ecosystem Modeling** — a spatial predator/prey/plant world.
6. **Qualitative Classification** — Wolfram's four classes and Rule 110.
7. **Evolutionary Dynamics** — does natural selection favor accurate
   perception?

## How to run these

- **In your browser, zero install:** use the Binder or Colab badges in
  the [repository README](https://github.com/USERNAME/ca-dynamics).
- **Locally:**
  ```bash
  git clone https://github.com/USERNAME/ca-dynamics
  cd ca-dynamics
  pip install -e ".[notebooks]"
  jupyter lab      # open the notebooks/ folder
  ```

## The full book

A typeset PDF of the complete book — including the chapters not covered
by notebooks (appendices, the full mathematical treatment, and the
bibliography) — is available in the repository.

## License

The `cadyn` software is released under the MIT License; the book text and
figures under Creative Commons Attribution 4.0 (CC BY 4.0). You are free
to use, adapt, and share both, with attribution.
