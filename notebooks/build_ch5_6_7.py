"""Build notebooks for Chapter 5 (ecosystem), Chapter 6 (Wolfram
classes) and Chapter 7 (evolutionary dynamics)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _nbbuild import md, code, notebook, save, SETUP

OUT = pathlib.Path(__file__).parent


# ====================================================================== #
# Chapter 5 -- Ecosystem modeling
# ====================================================================== #
def chapter5():
    cells = [
        md(r"""
        # Chapter 5 &mdash; Ecosystem Modeling with Cellular Automata

        Here a CA models a small **three-species ecosystem** &mdash;
        plants, prey, and predators &mdash; on a torus. Simple local rules
        for grazing, predation, death, and growth produce the classic
        out-of-phase **predator/prey oscillations** of the Lotka&ndash;Volterra
        equations, but as an *individual-based, spatial* model rather than
        smooth differential equations.
        """),
        SETUP,
        code("from cadyn.ecosystem import Ecosystem"),
        md(r"""
        ## Initialize and evolve

        Cells are `EMPTY`, `PLANT`, `PREY`, or `PREDATOR`. We seed a world
        with a modest prey and predator probability and let it run.
        """),
        code("""
            eco = Ecosystem(n=200, p_prey=0.212, p_pred=0.029,
                            rng=np.random.default_rng(11))
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            eco.show(ax=axes[0], title="initial state")
            eco.run(2000)
            eco.show(ax=axes[1], title="after 2000 generations")
            plt.show()
        """),
        md(r"""
        ## Population dynamics

        Plotting the three populations over time reveals the tell-tale
        signature: predators (red) and prey (yellow) oscillate **out of
        phase**, damping toward a noisy equilibrium. Without predators the
        prey would eat all the plants and then starve.
        """),
        code("""
            eco.plot_populations(sample_every=4)
            plt.show()
        """),
        md(r"""
        ## The phase portrait

        Plot the three populations as one moving point in 3-D and the
        oscillation becomes a **spiral** winding toward an interior
        equilibrium &mdash; exactly the picture Poincar\u00e9 introduced for
        differential equations.
        """),
        code("""
            eco.plot_phase_portrait(sample_every=4)
            plt.show()
        """),
        md(r"""
        ## When does it collapse? A parameter sweep

        Coexistence is not guaranteed. If predators are too feeble they die
        out; too voracious and they strip the world in one overshoot. We can
        map this directly by sweeping two predator parameters and recording
        how often all species survive. (This is a smaller/faster sweep than
        the book figure &mdash; raise the resolution if you have time.)
        """),
        code("""
            b_preds = np.linspace(0.1, 0.9, 7)
            d_preds = np.linspace(0.05, 0.45, 7)
            survival = np.zeros((len(d_preds), len(b_preds)))
            for i, dp in enumerate(d_preds):
                for j, bp in enumerate(b_preds):
                    e = Ecosystem(n=50, p_prey=0.212, p_pred=0.029,
                                  b_pred=bp, d_pred=dp,
                                  rng=np.random.default_rng(10*i + j))
                    e.run_until(300)
                    survival[i, j] = 0.0 if e.extinct() else 1.0

            fig, ax = plt.subplots(figsize=(6.5, 5))
            im = ax.imshow(survival, origin="lower", cmap="RdYlGn",
                           vmin=0, vmax=1, aspect="auto",
                           extent=[b_preds[0], b_preds[-1],
                                   d_preds[0], d_preds[-1]])
            ax.set_xlabel("predation efficiency  b_pred")
            ax.set_ylabel("predator death rate  d_pred")
            ax.set_title("green = all species survived 300 generations")
            fig.colorbar(im, ax=ax); plt.show()
        """),
        md(r"""
        ## Your turn

        1. Push the survival sweep to higher resolution and more trials per
           cell for a smoother phase diagram.
        2. Find an "imprudent" parameter set that drives everything extinct,
           and plot the boom-and-bust transient with `eco.plot_populations()`.
        3. Rotate the phase portrait: `ax = eco.plot_phase_portrait();
           ax.view_init(elev=30, azim=120)`.
        """),
    ]
    return notebook(cells)


# ====================================================================== #
# Chapter 6 -- Wolfram's qualitative classification
# ====================================================================== #
def chapter6():
    cells = [
        md(r"""
        # Chapter 6 &mdash; Qualitative Classification of CA

        Across all the CA we've met, long-term behavior sorts into just
        **four qualitative classes** (Stephen Wolfram): homogeneous,
        periodic, chaotic, and complex. The simplest arena to see them is
        the family of **elementary CA** &mdash; $k=2$, nearest neighbors, so
        exactly $2^8 = 256$ possible rules, each with a number 0&ndash;255.
        """),
        SETUP,
        code("from cadyn import ca1d"),
        md(r"""
        ## Wolfram's rule numbering

        List the 8 neighborhoods `111, 110, ..., 001, 000` and write the
        new center value under each; read those 8 bits as a number. For
        example **Rule 90** turns out to be the additive kernel
        $[1\,0\,1] \bmod 2$ &mdash; our Sierpi\u0144ski gasket &mdash; in
        disguise. Let's confirm that.
        """),
        code("""
            init = ca1d.single_seed(129)
            a = ca1d.run_elementary(init, rule=90, generations=20)
            b = ca1d.run_additive(init, [1, 0, 1], k=2, generations=20)
            print("Rule 90 == kernel [1 0 1] mod 2:", np.array_equal(a, b))
        """),
        md(r"""
        ## The four classes

        One representative rule from each class, evolved from the **same**
        random initial row:

        - **Class 1 (homogeneous):** Rule 160 &mdash; dies to a blank state.
        - **Class 2 (periodic):** Rule 108 &mdash; freezes into stable /
          blinking stripes.
        - **Class 3 (chaotic):** Rule 30 &mdash; boils into noise (it's even
          used as a pseudo-random generator).
        - **Class 4 (complex):** Rule 110 &mdash; localized particles that
          move and interact; *Turing complete*.
        """),
        code("""
            rng = np.random.default_rng(0)
            init = rng.integers(0, 2, 300)
            classes = [(160, "Class 1: Rule 160"), (108, "Class 2: Rule 108"),
                       (30,  "Class 3: Rule 30"),  (110, "Class 4: Rule 110")]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for ax, (rule, label) in zip(axes.flat, classes):
                h = ca1d.run_elementary(init, rule, 150)
                ax.imshow(h, cmap="binary", interpolation="nearest", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_title(label)
            plt.show()
        """),
        md(r"""
        ## Rule 110's particles up close

        Class 4 is the interesting one: neither frozen nor random, but
        structured. From a random start, Rule 110 threads persistent
        moving structures &mdash; the raw material Matthew Cook used to
        prove it computationally **universal**.
        """),
        code("""
            h = ca1d.run_elementary(rng.integers(0, 2, 500), rule=110, generations=400)
            ca1d.plot_spacetime(h, k=2, title="Rule 110 from a random start")
            plt.show()
        """),
        md(r"""
        ## Rule 30 as a random generator

        From a *single* seed, Rule 30's center column is so disordered it
        passes many statistical randomness tests. Let's look at its
        triangular growth.
        """),
        code("""
            h = ca1d.run_elementary(ca1d.single_seed(601), rule=30, generations=300)
            ca1d.plot_spacetime(h, k=2, title="Rule 30 from a single seed")
            plt.show()
            center = h[:, h.shape[1] // 2]
            print("center-column bits, first 40 generations:")
            print("".join(map(str, center[:40])))
        """),
        md(r"""
        ## Your turn

        1. Browse the 256 rules. Some famous ones: 54 and 137 (Class 4
           candidates), 150 (additive, three-neighbor XOR), 184 (traffic
           flow), 60 (another Sierpi\u0144ski relative).
           ```python
           for rule in (54, 150, 184, 60):
               h = ca1d.run_elementary(ca1d.single_seed(301), rule, 150)
               ca1d.plot_spacetime(h, 2, title=f"Rule {rule}"); plt.show()
           ```
        2. Which class does each land in? Is the classification always
           obvious &mdash; or genuinely ambiguous for some rules?
        3. **Computational irreducibility:** for a Class 4 rule there is in
           general no shortcut &mdash; you must run it to know its future.
           Contrast with the additive rules of Chapter 3, whose polynomial
           form *is* a shortcut.
        """),
    ]
    return notebook(cells)


# ====================================================================== #
# Chapter 7 -- Evolutionary dynamics
# ====================================================================== #
def chapter7():
    cells = [
        md(r"""
        # Chapter 7 &mdash; Evolutionary Dynamics: A Case Study

        This chapter puts the ecosystem machinery to work on a real
        scientific question, from Hoffman et al. (2010): **does natural
        selection favor *veridical* (accurate) perception?** The intuitive
        answer is yes &mdash; but accurate perception costs *time*, and in a
        life-or-death moment a slow-but-accurate strategy can lose to a
        fast-but-crude one.

        We develop the compact **mean-field model** in full here (every
        claim is checkable in a few lines); the spatial movement-and-strategy
        experiment is described in the book text.
        """),
        SETUP,
        code("from cadyn import evolution as ev"),
        md(r"""
        ## Three strategies

        One species carries three competing strategies, each with a fixed
        probability of success (its fitness):

        - **R** (random / non-perceiving): $p = 0.6$
        - **P** (probabilistic): $p = 0.75$
        - **V** (veridical / always-perceiving): $p = 0.9$

        A cell keeps its strategy with probability $p_i$, else adopts a
        neighbor's. Treating current proportions as neighbor probabilities
        gives a time-varying matrix update $s(t{+}1) = W(t)\,s(t)$.
        """),
        code("""
            p = ev.FITNESS_RPV
            print("fitness vector (R, P, V):", p)
            print("labels:", ev.STRATEGY_LABELS)
            s = np.array([0.5, 0.3, 0.2])
            print("\\ntransition matrix W(t) at s = [0.5, 0.3, 0.2]:")
            print(np.round(ev.transition_matrix(p, s), 3))
            print("columns sum to 1:", np.allclose(ev.transition_matrix(p, s).sum(0), 1))
        """),
        md(r"""
        ## Veridical perception wins (without a time cost)

        From any interior start, the fittest strategy takes over completely.
        V wins even when it *starts* as the smallest population.
        """),
        code("""
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
            ev.plot_run(p, [1/3, 1/3, 1/3], 200, ev.STRATEGY_LABELS,
                        ax=axes[0], title="equal start")
            ev.plot_run(p, [0.5, 0.3, 0.2], 200, ev.STRATEGY_LABELS,
                        ax=axes[1], title="V starts smallest")
            plt.show()
            print("final from [0.5,0.3,0.2]:",
                  np.round(ev.run(p, [0.5, 0.3, 0.2], 200)[-1], 3))
        """),
        md(r"""
        ## Every path leads to the V vertex

        The three-strategy state lives on a triangle (the 2-simplex). Plot
        trajectories from many starts &mdash; all flow to the veridical
        corner.
        """),
        code("""
            rng = np.random.default_rng(1)
            starts = [rng.dirichlet(np.ones(3)) for _ in range(12)]
            ev.plot_simplex_trajectory(p, starts, 200)
            plt.show()
        """),
        md(r"""
        ## The Lemma: a clean closed form

        The clunky matrix $W(t)$ hides a tidy update. With $\circ$ the
        elementwise product,
        $$ s(t+1) = p \circ s + s\,\langle \vec 1 - p,\ s\rangle. $$
        The module offers three equivalent implementations and a checker
        that confirms they agree &mdash; in **any** dimension.
        """),
        code("""
            print("Lemma holds for the worked example:", ev.verify_lemma(p, s))
            rng = np.random.default_rng(0)
            ok = all(ev.verify_lemma(rng.random(n), rng.dirichlet(np.ones(n)))
                     for n in (2, 3, 4, 5, 8) for _ in range(50))
            print("Lemma holds across 250 random cases, dims 2..8:", ok)
        """),
        md(r"""
        ## The (empirical) Theorem

        If one strategy has strictly greatest fitness, its proportion tends
        to 1 and the rest to 0. A formal proof eluded the original paper,
        but the evidence is overwhelming:
        """),
        code("""
            rng = np.random.default_rng(2)
            wins3 = sum(ev.dominant_strategy_wins(p, rng.dirichlet(np.ones(3)))
                        for _ in range(300))
            print(f"3-D: fittest strategy won {wins3}/300 random starts")
            for n in (4, 6, 10):
                pv = rng.random(n)
                w = sum(ev.dominant_strategy_wins(pv, rng.dirichlet(np.ones(n)))
                        for _ in range(50))
                print(f"{n}-D: fittest won {w}/50")
        """),
        md(r"""
        ## The reversal: charge V for its computation time

        Nothing above explains *why* V should be fittest &mdash; that's an
        input. If accurate perception is costly, model that by lowering V's
        effective fitness. Drop it below P's and the outcome flips: the
        veridical strategy now goes **extinct**.
        """),
        code("""
            p_penalized = np.array([0.6, 0.75, 0.55])   # V's fitness cut by a time cost
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
            ev.plot_run(p, [1/3]*3, 200, ev.STRATEGY_LABELS,
                        ax=axes[0], title="no penalty: V wins")
            ev.plot_run(p_penalized, [1/3]*3, 200,
                        ["R", "P", "V (penalized)"],
                        ax=axes[1], title="V penalized to 0.55: P wins")
            plt.show()
            print("penalized winner index (0=R,1=P,2=V):",
                  int(np.argmax(ev.run(p_penalized, [1/3]*3, 300)[-1])))
        """),
        md(r"""
        ## Your turn

        1. Where exactly is the tipping point? Sweep V's fitness from 0.5 to
           0.9 and find the value at which the winner switches from P to V.
        2. Add a fourth strategy and confirm the Theorem still picks the
           unique fittest.
        3. The spatial experiment (movement + per-individual strategies +
           a move-order time penalty) is described in the book. As a
           project, extend `cadyn.ecosystem` to reproduce it &mdash; the
           mean-field model here is your validated target.

        > **Connection.** This is a concrete, spatial instance of Hoffman's
        > *Interface Theory of Perception*: evolution optimizes fitness, not
        > truth, and the two can diverge once perception has a cost.
        """),
    ]
    return notebook(cells)


if __name__ == "__main__":
    save(chapter5(), OUT / "05_ecosystem_modeling.ipynb")
    save(chapter6(), OUT / "06_wolfram_classification.ipynb")
    save(chapter7(), OUT / "07_evolutionary_dynamics.ipynb")
    print("built chapters 5, 6, and 7")
