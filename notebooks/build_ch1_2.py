"""Build notebooks for Chapter 1 (intro) and Chapter 2 (0-D CA)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _nbbuild import md, code, notebook, save, SETUP

OUT = pathlib.Path(__file__).parent


# ====================================================================== #
# Chapter 1 -- What Are Cellular Automata?
# ====================================================================== #
def chapter1():
    cells = [
        md(r"""
        # Chapter 1 &mdash; What Are Cellular Automata?

        *Cellular Automata Dynamics &mdash; Python Edition*

        This first notebook is mostly conceptual: it introduces what a
        cellular automaton (CA) **is**, fixes the vocabulary used
        throughout, and gives you a first runnable taste of each of the
        four worlds we will visit (dimensions 0, 1, and 2, plus the
        ecosystem). The later notebooks go deep on each.

        A CA is a simple universe: a lattice of **cells**, each in one of
        finitely many **states**, all updated **synchronously** by a
        **transition rule** that looks only at a cell's **neighborhood**.
        Simple local rules, iterated, produce startlingly complex global
        behavior &mdash; the phenomenon of *emergence* that motivates the
        whole book.
        """),
        SETUP,
        md(r"""
        ## The formal ingredients

        To specify a CA you must give:

        1. **A lattice** of cells (dimension $n$), here made finite with
           *periodic* (wrap-around) boundaries &mdash; a line becomes a
           circle, a plane becomes a torus.
        2. **A neighborhood** for each cell (we use *homogeneous* CA: the
           same relative neighborhood everywhere).
        3. **Transition rules** $T$ giving each cell's next state from its
           neighborhood's current states.

        The dynamics are then fixed by the initial state $C_0$ together
        with
        $$ C_{t+1}(i_1,\dots,i_n) = T\bigl(N_t(i_1,\dots,i_n)\bigr). $$

        Let's see one representative from each chapter.
        """),
        md("### A 0-dimensional CA (Chapter 2): a single cell, $f(x)=x^2+1 \\pmod{17}$"),
        code("""
            from cadyn import fds
            s = fds.example_2_1()          # x^2 + 1 (mod 17)
            print("trajectory from 0:", s.trajectory(0))
            s.plot_basin_field()
            plt.show()
        """),
        md("### A 1-dimensional CA (Chapter 3): the rule $[1\\,0\\,1] \\bmod 2$ makes a Sierpi\u0144ski gasket"),
        code("""
            from cadyn import ca1d
            hist = ca1d.run_additive(ca1d.single_seed(256), [1, 0, 1],
                                     k=2, generations=128)
            ca1d.plot_spacetime(hist, k=2, title="kernel [1 0 1] mod 2")
            plt.show()
        """),
        md("### A 2-dimensional CA (Chapter 4): Conway's Game of Life, age-colored"),
        code("""
            from cadyn import ca2d
            alive0 = ca2d.random_life(120, p=0.3, rng=np.random.default_rng(0))
            alive, age = ca2d.run_life(alive0, 200)
            ca2d.show_life(age, title="Game of Life, generation 200")
            plt.show()
        """),
        md("### An ecosystem (Chapter 5): predator / prey / plants on a torus"),
        code("""
            from cadyn.ecosystem import Ecosystem
            eco = Ecosystem(n=120, rng=np.random.default_rng(1))
            eco.run(400)
            eco.plot_populations()
            plt.show()
        """),
        md(r"""
        ## Where to go next

        Each remaining notebook takes one of these worlds and explores it
        properly:

        | Notebook | Module | Theme |
        |---|---|---|
        | 02 | `cadyn.fds` | finite dynamical systems, basins, entropy |
        | 03 | `cadyn.ca1d` | 1-D CA, convolution kernels, fractals |
        | 04 | `cadyn.ca2d` | 2-D CA, image processing, Life |
        | 05 | `cadyn.ecosystem` | predator/prey dynamics |
        | 06 | `cadyn.ca1d` | Wolfram's four classes, Rule 110 |
        | 07 | `cadyn.evolution` | veridical perception & selection |

        **Exercise.** Re-run the four demos above with different random
        seeds and initial densities. Which worlds are sensitive to the
        starting condition, and which settle to the same behavior
        regardless?
        """),
    ]
    return notebook(cells)


# ====================================================================== #
# Chapter 2 -- 0-dimensional CA (finite dynamical systems)
# ====================================================================== #
def chapter2():
    cells = [
        md(r"""
        # Chapter 2 &mdash; Zero-Dimensional Cellular Automata

        A 0-dimensional CA has a lattice of a **single cell**. That sounds
        trivial &mdash; but a discrete function $f$ iterated on
        $\mathbb{Z}_n$ *is* a 0-D CA, and these **finite dynamical
        systems** already show off cycles, transients, basins of
        attraction, and entropy &mdash; the whole conceptual toolkit we
        reuse in every later chapter.

        The state is a single integer $X_t \in \{0,\dots,n-1\}$, evolving by
        $$ X_{t+1} = f(X_t) \bmod n. $$
        """),
        SETUP,
        code("from cadyn import fds"),
        md(r"""
        ## Trajectories, cycles, and transients

        Because there are only $n$ possible states, every trajectory must
        eventually repeat &mdash; it falls into a **cycle**. The part
        before the cycle is the **transient**. Let's watch this for
        $f(x)=x^2+1 \pmod{17}$ starting at $x=0$.
        """),
        code("""
            s = fds.example_2_1()               # f(x) = x^2 + 1 (mod 17)
            traj = s.trajectory(0)
            print("trajectory:", traj)
            print("the value", traj[-1], "recurs -> the tail is the attractive cycle")
            print("cycles found:", s.cycles())
        """),
        md(r"""
        ## The basin of attraction field

        Draw *every* state as a node with an arrow to its successor and you
        get the **basin of attraction field** &mdash; the discrete analogue
        of a phase portrait. Cycle edges are red, transient edges green.

        (The layout is structure-aware: cycles are drawn as clean polygons
        with their transient trees fanned outward, then relaxed with a
        force-directed pass. See the note at the end of this notebook.)
        """),
        code("""
            s.plot_basin_field()
            plt.show()
        """),
        md(r"""
        Notice the eight outermost nodes with no incoming arrow. These are
        **Garden of Eden** states: reachable as an initial condition but
        never produced by the dynamics.
        """),
        code("""
            print("Garden of Eden states:", sorted(s.garden_of_eden_states()))
        """),
        md(r"""
        ### A symmetry to spot

        Because $x^2 = (-x)^2$ and $-x \equiv n-x \pmod n$, the two
        predecessors of any node sum to $n=17$. Let's verify it.
        """),
        code("""
            g = s.to_graph()
            for v in g:
                preds = list(g.predecessors(v))
                if len(preds) == 2:
                    print(f"node {v:2d}: predecessors {preds} sum to {sum(preds)}")
        """),
        md(r"""
        ## Shannon entropy

        Start from a *uniformly random* initial state and ask: after $t$
        steps, how uncertain are we about the current value? That is
        Shannon's entropy
        $$ H_t = -\sum_k p_t(k)\log_2 p_t(k) \quad\text{(bits)}. $$
        It starts at the maximum $\log_2 n$ and **decreases** as the system
        funnels into its cycles &mdash; irreversibility at work.
        """),
        code("""
            fig, ax = plt.subplots()
            s.plot_entropy(14, ax=ax)
            plt.show()
            H = s.entropy_curve(14)
            print(f"H_0 = {H[0]:.3f} = log2(17) = {np.log2(17):.3f}")
            print(f"H_min = {H.min():.3f}")
        """),
        md(r"""
        ## More examples

        The same function on a larger, composite modulus has a richer basin
        field. $221 = 13\times 17$, and by the Chinese Remainder Theorem its
        dynamics factor &mdash; producing paired cycles (two 12-cycles and
        two 6-cycles).
        """),
        code("""
            s221 = fds.example_2_4()            # x^2 + 1 (mod 221)
            print("cycle lengths:", sorted(len(c) for c in s221.cycles()))
            s221.plot_basin_field(font_size=6)
            plt.show()
        """),
        md(r"""
        ## The Ulam (Collatz) $3N{+}1$ function

        $$ f(n) = \begin{cases} n/2 & n \text{ even}\\ 3n+1 & n\text{ odd}\end{cases} $$

        Iterated on the integers it is conjectured *always* to reach the
        cycle $4\to2\to1$. Computed modulo $n$, extra attractors can appear
        &mdash; e.g. a surprise attractor at 23 when $n=47$.
        """),
        code("""
            for n in (20, 40, 47):
                u = fds.ulam_mod(n)
                print(f"mod {n:2d}: cycles = {[sorted(c) for c in u.cycles()]}")
        """),
        code("""
            fds.ulam_mod(47).plot_basin_field()
            plt.show()
        """),
        md(r"""
        ## Your turn

        1. Define your own function and wrap it in a `FiniteDynamicalSystem`:
           ```python
           g = fds.FiniteDynamicalSystem(lambda x: (x**3 + x + 1), n=29, name="x^3+x+1")
           g.plot_basin_field(); plt.show()
           ```
        2. Sweep the modulus for $x^2+1$ and watch how the number and length
           of cycles change. Are prime moduli simpler?
        3. Compare `refine=0` (pure radial layout) with the default
           `refine=50` (force-relaxed) in `plot_basin_field`.

        > **On the layout.** The nodes repel one another while edges act as
        > fixed-length struts that pivot freely &mdash; a *force-directed*
        > layout. We seed it with a structure-aware radial placement (cycle
        > on a circle, transient trees fanned into angular wedges) and then
        > relax, holding the cycle pinned, so the picture matches the clean
        > hand-drawn figures of the original 1997 book.
        """),
        code("""
            g = fds.FiniteDynamicalSystem(lambda x: x**3 + x + 1, n=29,
                                          name="x^3 + x + 1")
            g.plot_basin_field()
            plt.show()
        """),
    ]
    return notebook(cells)


if __name__ == "__main__":
    save(chapter1(), OUT / "01_what_are_cellular_automata.ipynb")
    save(chapter2(), OUT / "02_zero_dimensional_ca.ipynb")
    print("built chapters 1 and 2")
