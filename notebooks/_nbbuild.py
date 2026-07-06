"""Helpers for building the chapter notebooks with nbformat.

Each chapter notebook is assembled as an alternating sequence of markdown
and code cells.  We keep a tiny DSL (md / code) so the chapter builders
read like an outline.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def md(text):
    """A markdown cell.  Leading indentation is stripped so builder code
    can use triple-quoted strings at any indentation level."""
    import textwrap
    return new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(src):
    import textwrap
    return new_code_cell(textwrap.dedent(src).strip("\n"))


def notebook(cells, kernel="python3"):
    nb = new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": kernel},
        "language_info": {"name": "python"},
    }
    return nb


# a standard setup cell shared by every chapter notebook
SETUP = code("""
    # Setup: put the cadyn package on the path and enable inline figures.
    # If you installed the package (`pip install -e .` from the repo root),
    # the sys.path line is unnecessary.
    import sys, pathlib
    repo = pathlib.Path.cwd()
    for _ in range(4):                      # find the repo root from anywhere
        if (repo / "cadyn").is_dir():
            break
        repo = repo.parent
    sys.path.insert(0, str(repo))

    %matplotlib inline
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (8, 5)
""")


def save(nb, path):
    with open(path, "w") as f:
        nbf.write(nb, f)
    return path
