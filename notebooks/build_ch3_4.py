"""Build notebooks for Chapter 3 (1-D CA) and Chapter 4 (2-D CA)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _nbbuild import md, code, notebook, save, SETUP

OUT = pathlib.Path(__file__).parent


# ====================================================================== #
# Chapter 3 -- 1-dimensional CA
# ====================================================================== #
def chapter3():
    cells = [
        md(r"""
        # Chapter 3 &mdash; One-Dimensional Cellular Automata

        Now the lattice is a row of cells (a circle, with wrap-around), and
        we can see a CA's *entire history at a glance* by stacking
        successive generations as rows of an image.

        The star of this chapter is the idea that **additive** CA rules are
        exactly **convolution kernels** &mdash; the same operation used to
        filter signals and images. From this one idea flow fractals,
        a polynomial algebra, and a bridge to digital signal processing.
        """),
        SETUP,
        code("from cadyn import ca1d"),
        md(r"""
        ## Transition rules as convolution kernels

        Applying a kernel like $[1\,1\,1]$ means each new cell is the sum of
        its neighborhood (mod $k$). Let's evolve $k=5$, $n=21$ from a single
        seed and view time flowing **left to right** (one generation per
        column), as in the book's Figure 2.
        """),
        code("""
            init = np.zeros(21, dtype=int); init[10] = 1
            hist = ca1d.run_additive(init, [1, 1, 1], k=5, generations=136)
            ca1d.plot_spacetime(hist, k=5, zero_color="black",
                                time_axis="right",
                                title="kernel [1 1 1], k=5, n=21")
            plt.show()
            print("first 10 generations:")
            print(hist[:10])
        """),
        md(r"""
        ## The Sierpi\u0144ski gasket from $[1\,0\,1] \bmod 2$

        The neighborhood is just the two *outer* cells (the center is
        excluded), summed mod 2. From a single seed this draws successive
        approximations to **Sierpi\u0144ski's gasket** &mdash; a fractal.
        """),
        code("""
            hist = ca1d.run_additive(ca1d.single_seed(512), [1, 0, 1],
                                     k=2, generations=255)
            ca1d.plot_spacetime(hist, k=2, title="Sierpinski gasket: [1 0 1] mod 2")
            plt.show()
        """),
        md(r"""
        ### More states, more structure

        The same rule with more states (larger modulus $k$) gives richer
        patterns. Prime moduli stay simple; composite moduli grow complex
        with their number of factors.
        """),
        code("""
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, k in zip(axes, (3, 5, 30)):
                h = ca1d.run_additive(ca1d.single_seed(256), [1, 0, 1], k, 128)
                ca1d.plot_spacetime(h, k, ax=ax, title=f"k = {k}")
            plt.show()
        """),
        md(r"""
        ## Why? The polynomial representation

        Represent a state as a polynomial $A(x)=\sum a_i x^i$ in
        $\mathbb{Z}_k[x]/(x^n-1)$, and the kernel as $T(x)$. Then **one
        generation is one multiplication**: $A^{(t+1)} = T(x)\,A^{(t)}$.
        For $[1\,0\,1]$, $T(x) = x + x^{-1}$, and evolving a single seed
        raises it to a power &mdash; the binomial coefficients
        $\binom{t}{i} \bmod 2$ appear, which is *exactly* why the
        Sierpi\u0144ski pattern shows up.

        Let's confirm the polynomial step agrees with direct convolution.
        """),
        code("""
            n, k = 64, 5
            init = ca1d.single_seed(n, value=3)
            kp = ca1d.kernel_to_poly([1, 2, 3], n)
            a, b = init.copy(), init.copy()
            for _ in range(20):
                a = ca1d.convolve_step(a, [1, 2, 3], k)   # direct
                b = ca1d.poly_step(b, kp, k)              # polynomial
            print("polynomial step == convolution step:", np.array_equal(a, b))
        """),
        md(r"""
        ## A non-additive rule: sum of squares

        Not every rule is a convolution. The **sum-of-squares** rule
        $C' = C_{i-1}^2 + 2C_i^2 + C_{i+1}^2 \pmod k$ is nonlinear, yet the
        Sierpi\u0144ski gasket still surfaces whenever $k$ is a power of 2.
        """),
        code("""
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, k in zip(axes, (7, 8, 128)):     # 8 and 128 are powers of 2
                step = ca1d.sum_of_squares_step(k)
                h = ca1d.run_rule(ca1d.single_seed(200), step, 100)
                ca1d.plot_spacetime(h, k, ax=ax, title=f"sum of squares, k = {k}")
            plt.show()
        """),
        md(r"""
        ## Entropy in one dimension

        We can measure the **spatial entropy** of each generation &mdash;
        how evenly the $k$ values populate the lattice. Different rules sit
        at very different entropy levels: a freezing rule collapses to 0, a
        thorough mixer hovers near $\log_2 k$, and the particle-forming
        1-D Life settles in between.
        """),
        code("""
            rng = np.random.default_rng(5)
            n, gens, k = 400, 300, 8
            x0 = rng.integers(0, k, n)

            runs = {
                "freeze (f=0)":       ca1d.run_rule(x0, lambda x: np.zeros_like(x), gens),
                "additive [1 1 1]":   ca1d.run_additive(x0, [1, 1, 1], k, gens),
                "sum of squares":     ca1d.run_rule(x0, ca1d.sum_of_squares_step(k), gens),
            }
            fig, ax = plt.subplots()
            for label, h in runs.items():
                ax.plot(ca1d.spatial_entropy(h, k), lw=1.8, label=label)
            ax.axhline(np.log2(k), color="0.85", lw=0.8)
            ax.set_xlabel("generation"); ax.set_ylabel("spatial entropy (bits)")
            ax.legend(); plt.show()
        """),
        md(r"""
        ## Your turn

        1. Try asymmetric kernels like `[1, 2, 3]` or longer ones like
           `[1, 0, 0, 0, 1]` at various $k$.
        2. Evolve the **1-D Game of Life** and look for particle-like
           tracks:
           ```python
           rng = np.random.default_rng(7)
           x0 = np.where(rng.random(512) < 0.5, rng.integers(1, 256, 512), 0)
           h = ca1d.run_rule(x0, ca1d.life_1d_step(rng), 300)
           ca1d.plot_spacetime(h, 256, time_axis="right"); plt.show()
           ```
        3. Which of your rules keep entropy high, and which drive it down?
        """),
    ]
    return notebook(cells)


# ====================================================================== #
# Chapter 4 -- 2-dimensional CA
# ====================================================================== #
def chapter4():
    cells = [
        md(r"""
        # Chapter 4 &mdash; Two-Dimensional Cellular Automata

        On a 2-D toroidal lattice a CA state is an image, and its rule is a
        **2-D convolution kernel** &mdash; the exact machinery of image
        processing. From a single seed, additive rules unfold intricate,
        symmetric *mandalas*; and one non-additive rule gives the most
        famous CA of all, Conway's **Game of Life**.
        """),
        SETUP,
        code("from cadyn import ca2d"),
        md(r"""
        ## A mandala from a single seed

        The von Neumann kernel sums a cell with its four orthogonal
        neighbors (mod 256). One nonzero cell, grown for 100 generations,
        produces this:
        """),
        code("""
            final, snaps = ca2d.run_additive(ca2d.single_seed(300),
                                             ca2d.KERNEL_VON_NEUMANN,
                                             k=256, generations=100,
                                             sample_at=(100,))
            ca2d.show_state(snaps[100], k=256, zero_color="black",
                            title="von Neumann kernel, generation 100")
            plt.show()
        """),
        md(r"""
        ## The kernels really are image filters

        The same convolution machinery, applied *once* (without the mod-$k$
        step), is ordinary image processing. Here a Laplacian kernel
        extracts edges and a smoothing kernel blurs &mdash; iterated
        smoothing is discrete diffusion (the heat equation).
        """),
        code("""
            img = ca2d.demo_image(200)
            edges = ca2d.filter_image(img, ca2d.EDGE_KERNEL)
            smooth = img
            for _ in range(6):
                smooth = ca2d.filter_image(smooth, ca2d.BLUR_KERNEL)

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            ca2d.show_gray(img, ax=axes[0], title="input")
            ca2d.show_gray(np.abs(edges), ax=axes[1], title="edges (Laplacian)")
            ca2d.show_gray(smooth, ax=axes[2], title="smoothed (6x)")
            plt.show()
        """),
        md(r"""
        ## Conway's Game of Life

        The rules: a dead cell with exactly 3 live neighbors is **born**; a
        live cell with 2 or 3 neighbors **survives**; otherwise it dies. We
        color live cells by **age** (cerise = newborn ... violet =
        ancient).
        """),
        code("""
            alive0 = ca2d.random_life(200, p=0.3, rng=np.random.default_rng(3))
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            ca2d.show_life(alive0.astype(int), ax=axes[0], title="initial (p=0.3)")
            alive, age = ca2d.run_life(alive0, 500)
            ca2d.show_life(age, ax=axes[1], title="generation 500")
            plt.show()
        """),
        md(r"""
        ### A bestiary: the glider and the Gosper gun

        Life has a rich zoo of patterns. A **glider** is a small shape that
        translates diagonally, returning to itself shifted by $(1,1)$ every
        4 generations. The **Gosper glider gun** endlessly emits gliders
        &mdash; the discovery that first proved Life patterns can grow
        without bound.
        """),
        code("""
            # glider: watch it move
            g = ca2d.pattern_grid(ca2d.GLIDER, 16)
            fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
            a, age = g.astype(bool), g.astype(int)
            for ax, gen in zip(axes, range(5)):
                if gen:
                    a, age = ca2d.run_life(a, 1)
                ca2d.show_life(age, ax=ax, title=f"gen {gen}")
            plt.show()

            # verify the period-4 diagonal translation
            a4, _ = ca2d.run_life(g.astype(bool), 4)
            shifted = np.roll(np.roll(g.astype(bool), 1, 0), 1, 1)
            print("glider returns to itself shifted by (1,1) after 4 gens:",
                  np.array_equal(a4, shifted))
        """),
        code("""
            gun = ca2d.pattern_grid(ca2d.GOSPER_GLIDER_GUN, 90, 130, top=5, left=5)
            _, age = ca2d.run_life(gun, 120)
            ca2d.show_life(age, title="Gosper glider gun after 120 generations")
            plt.show()
        """),
        md(r"""
        ## A number-theoretic rule: Euler's totient

        Non-additive rules need not be Life-like. This one sums Euler's
        totient $\varphi$ over the Moore neighborhood (mod 256), producing
        yet another mandala from a single seed.
        """),
        code("""
            state = ca2d.run_rule(ca2d.single_seed(300),
                                  ca2d.totient_step(256), 200)
            ca2d.show_state(state, k=256, crop=180,
                            title="Euler totient rule, generation 200 (detail)")
            plt.show()
        """),
        md(r"""
        ## Your turn

        1. Swap in the signed kernel `ca2d.KERNEL_4_3` and grow a mandala
           from a single seed.
        2. Seed Life with your own patterns via
           `ca2d.pattern_grid(list_of_(row,col), n)` &mdash; try a
           *blinker* `[(1,0),(1,1),(1,2)]` or an *r-pentomino*.
        3. Start Life from `ca2d.seed_grid` of many seeds and watch the
           interactions. (Animations are especially fun &mdash; export
           frames with `show_life` inside a loop.)
        """),
    ]
    return notebook(cells)


if __name__ == "__main__":
    save(chapter3(), OUT / "03_one_dimensional_ca.ipynb")
    save(chapter4(), OUT / "04_two_dimensional_ca.ipynb")
    print("built chapters 3 and 4")
