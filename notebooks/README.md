# JupyterLab Notebooks

Interactive, chapter-by-chapter companions to *Cellular Automata
Dynamics &mdash; Python Edition*. Each notebook wraps narrative markdown
around live calls into the `cadyn` package, so you can read, run, tweak,
and experiment.

| Notebook | Chapter | Module explored |
|---|---|---|
| `01_what_are_cellular_automata.ipynb` | 1 &mdash; What Are CA? | one demo from each world |
| `02_zero_dimensional_ca.ipynb` | 2 &mdash; 0-D CA | `cadyn.fds` |
| `03_one_dimensional_ca.ipynb` | 3 &mdash; 1-D CA | `cadyn.ca1d` |
| `04_two_dimensional_ca.ipynb` | 4 &mdash; 2-D CA | `cadyn.ca2d` |
| `05_ecosystem_modeling.ipynb` | 5 &mdash; Ecosystems | `cadyn.ecosystem` |
| `06_wolfram_classification.ipynb` | 6 &mdash; Wolfram classes | `cadyn.ca1d` (elementary) |
| `07_evolutionary_dynamics.ipynb` | 7 &mdash; Evolution | `cadyn.evolution` |

Every notebook ends with a **"Your turn"** section of open-ended
exercises.

## Running them

From the repository root:

```bash
pip install numpy scipy matplotlib networkx   # runtime deps
pip install jupyterlab                          # to run the notebooks
jupyter lab                                     # then open notebooks/
```

The first code cell of each notebook locates the `cadyn` package
automatically by walking up from the working directory, so the notebooks
run whether or not you have installed the package. (If you prefer,
`pip install -e .` from the repo root installs `cadyn` and makes that
path juggling unnecessary.)

The notebooks in this folder are shipped **already executed**, so their
figures are visible on GitHub or nbviewer without running anything.
Choose *Kernel &rarr; Restart & Run All* to reproduce them yourself.

## Rebuilding the notebooks from source

The notebooks are generated from small builder scripts (so the prose and
code stay in one place and are easy to diff):

```bash
cd notebooks
python build_ch1_2.py
python build_ch3_4.py
python build_ch5_6_7.py
# then execute them, embedding outputs:
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb
```

`_nbbuild.py` holds the shared `md()` / `code()` / `notebook()` helpers.
