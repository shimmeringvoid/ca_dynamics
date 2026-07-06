"""pytest suite for the cadyn package. Run with:  pytest -q"""
import numpy as np
import pytest

from cadyn import fds, ca1d, ca2d, numtheory
from cadyn.ecosystem import Ecosystem


# ---------------------------------------------------------------- 0-D ----
def test_mod17_trajectory_and_cycle():
    s = fds.example_2_1()
    assert s.trajectory(0)[:9] == [0, 1, 2, 5, 9, 14, 10, 16, 2]
    cyc = s.cycles()
    assert len(cyc) == 1
    assert sorted(cyc[0]) == [2, 5, 9, 10, 14, 16]


def test_mod17_garden_of_eden():
    s = fds.example_2_1()
    assert sorted(s.garden_of_eden_states()) == [4, 6, 7, 8, 11, 12, 13, 15]


def test_mod37_basins():
    s = fds.example_2_3()
    b = s.basins()
    assert sorted(tuple(sorted(c["cycle"])) for c in b) == [(8, 28), (11,), (27,)]
    assert sorted(len(c["basin"]) for c in b) == [6, 8, 23]


def test_mod221_cycle_structure():
    s = fds.example_2_4()
    assert sorted(len(c) for c in s.cycles()) == [6, 6, 12, 12]


@pytest.mark.parametrize("n,expected", [
    (37, (np.log2(37), 1.478)),
    (221, (np.log2(221), 4.983)),
])
def test_entropy_endpoints(n, expected):
    s = fds.FiniteDynamicalSystem(numtheory.square_plus_one, n)
    H = s.entropy_curve(16)
    assert abs(H[0] - expected[0]) < 1e-3
    assert abs(H.min() - expected[1]) < 2e-3


def test_ulam_mod20():
    got = {tuple(sorted(c)) for c in fds.ulam_mod(20).cycles()}
    assert got == {(0,), (1, 2, 4)}


def test_ulam_mod47_has_extra_attractor_23():
    got = {tuple(sorted(c)) for c in fds.ulam_mod(47).cycles()}
    assert (23,) in got


# ---------------------------------------------------------------- 1-D ----
def test_k5n21_generations():
    init = ca1d.single_seed(21)
    hist = ca1d.run_additive(init, [1, 1, 1], 5, 9)
    assert list(hist[9]) == \
        [0, 1, 4, 0, 1, 4, 2, 4, 4, 2, 4, 2, 4, 4, 2, 4, 1, 0, 4, 1, 0]


def test_kernel111_cycle_length_124():
    init = ca1d.single_seed(21)
    hist = ca1d.run_additive(init, [1, 1, 1], 5, 400)
    seen = {}
    for t, row in enumerate(map(tuple, hist)):
        if row in seen:
            assert t - seen[row] == 124
            break
        seen[row] = t
    else:
        pytest.fail("no cycle found")


def test_polynomial_equals_convolution():
    n, k = 64, 5
    init = ca1d.single_seed(n, value=3)
    kp = ca1d.kernel_to_poly([1, 2, 3], n)
    a, b = init.copy(), init.copy()
    for _ in range(25):
        a = ca1d.convolve_step(a, [1, 2, 3], k)
        b = ca1d.poly_step(b, kp, k)
    assert np.array_equal(a, b)


def test_sierpinski_row_is_binomial_mod2():
    # kernel [1 0 1] mod 2 from a single seed: row t has binom(t,i) mod 2
    from math import comb
    n = 129
    hist = ca1d.run_additive(ca1d.single_seed(n), [1, 0, 1], 2, 16)
    c = n // 2
    for t in range(1, 17):
        row = hist[t]
        expected = {(c - t + 2 * i) % n for i in range(t + 1)
                    if comb(t, i) % 2}
        got = set(np.flatnonzero(row).tolist())
        assert got == expected


def test_rule90_equals_additive_kernel():
    init = ca1d.single_seed(101)
    h_rule = ca1d.run_elementary(init, 90, 50)
    h_kern = ca1d.run_additive(init, [1, 0, 1], 2, 50)
    assert np.array_equal(h_rule, h_kern)


def test_rule150_equals_sum_kernel():
    init = ca1d.single_seed(101)
    h_rule = ca1d.run_elementary(init, 150, 40)
    h_kern = ca1d.run_additive(init, [1, 1, 1], 2, 40)
    assert np.array_equal(h_rule, h_kern)


def test_rule0_and_rule255():
    x = (np.random.default_rng(0).random(50) < 0.5).astype(int)
    # rule 0 -> everything dies; rule 255 -> everything alive
    assert ca1d.run_elementary(x, 0, 1)[1].sum() == 0
    assert ca1d.run_elementary(x, 255, 1)[1].sum() == 50


def test_elementary_rule_out_of_range():
    with pytest.raises(ValueError):
        ca1d.elementary_step(256)


# ---------------------------------------------------------------- 2-D ----
def test_life_blinker_period_2():
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, 1:4] = True                       # horizontal blinker
    g1 = ca2d.life_step(grid)
    g2 = ca2d.life_step(g1)
    assert not np.array_equal(g1, grid)       # it changed
    assert np.array_equal(g2, grid)           # period 2


def test_life_block_is_still_life():
    grid = np.zeros((4, 4), dtype=bool)
    grid[1:3, 1:3] = True                     # 2x2 block
    assert np.array_equal(ca2d.life_step(grid), grid)


def test_glider_translates_by_one_diagonally():
    g = ca2d.pattern_grid(ca2d.GLIDER, 20)
    g4 = g.copy()
    for _ in range(4):
        g4 = ca2d.life_step(g4)
    # after 4 generations the glider is the original shifted by (1,1)
    assert np.array_equal(np.roll(np.roll(g, 1, 0), 1, 1), g4)


def test_gosper_gun_emits_one_glider_per_period():
    gun = ca2d.pattern_grid(ca2d.GOSPER_GLIDER_GUN, 100, 150,
                            top=5, left=5)
    assert gun.sum() == 36
    a = gun.copy()
    for _ in range(30):
        a = ca2d.life_step(a)
    # the gun regenerates (36 cells) and has emitted one glider (5 cells)
    assert a.sum() == 41


def test_totient_table_matches_scalar():
    tab = numtheory.totient_table(100)
    assert tab[0] == 0 and tab[1] == 1
    for n in range(2, 100):
        assert tab[n] == numtheory.totient(n)


def test_totient_primes():
    for p in (2, 3, 5, 7, 11, 13, 17, 97):
        assert numtheory.totient(p) == p - 1


def test_additive2d_matches_kernel_sum():
    rng = np.random.default_rng(0)
    state = rng.integers(0, 256, (20, 20))
    out, _ = ca2d.run_additive(state, ca2d.KERNEL_VON_NEUMANN, 256, 1)
    manual = (np.roll(state, 1, 0) + np.roll(state, -1, 0) +
              np.roll(state, 1, 1) + np.roll(state, -1, 1) + state) % 256
    assert np.array_equal(out, manual)


# ---------------------------------------------------------------- eco ----
def test_ecosystem_runs_and_conserves_grid():
    eco = Ecosystem(n=40, rng=np.random.default_rng(1))
    eco.run(50)
    assert eco.grid.shape == (40, 40)
    assert set(np.unique(eco.grid)).issubset({0, 1, 2, 3})
    assert len(eco.history) == 51


# ------------------------------------------------------- expansions ------
def test_rule90_equals_kernel_101_mod2():
    # Wolfram Rule 90 is the additive kernel [1 0 1] mod 2
    init = ca1d.single_seed(129)
    a = ca1d.run_elementary(init, 90, 20)
    b = ca1d.run_additive(init, [1, 0, 1], 2, 20)
    assert np.array_equal(a, b)


def test_rule254_fills_from_single_seed():
    # Rule 254: a cell is 1 unless all three neighbors are 0 -> fills
    init = ca1d.single_seed(51)
    hist = ca1d.run_elementary(init, 254, 25)
    assert hist[-1].all()


def test_rule0_empties():
    hist = ca1d.run_elementary(np.ones(40, dtype=int), 0, 1)
    assert not hist[-1].any()


def test_elementary_rule_range():
    with pytest.raises(ValueError):
        ca1d.elementary_step(256)


def test_spatial_entropy_bounds():
    # uniform state -> 0 bits; perfectly mixed k values -> log2(k)
    uniform = np.zeros((1, 100), dtype=int)
    assert ca1d.spatial_entropy(uniform, 4)[0] == pytest.approx(0.0)
    mixed = np.tile(np.arange(4), 25).reshape(1, 100)
    assert ca1d.spatial_entropy(mixed, 4)[0] == pytest.approx(2.0)


def test_edge_kernel_zeros_flat_regions():
    flat = np.full((20, 20), 7.0)
    out = ca2d.filter_image(flat, ca2d.EDGE_KERNEL)
    assert np.allclose(out, 0.0)


def test_smoothing_preserves_mean():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32))
    out = ca2d.filter_image(img, ca2d.BLUR_KERNEL)   # normalized, periodic
    assert out.mean() == pytest.approx(img.mean(), abs=1e-9)


def test_ecosystem_extinction_detection():
    eco = Ecosystem(n=30, rng=np.random.default_rng(0))
    eco.grid[eco.grid == 3] = 0        # remove all predators by hand
    assert "predator" in eco.extinct()


def test_glider_translates_by_diagonal_every_4():
    g = ca2d.pattern_grid(ca2d.GLIDER, 16).astype(bool)
    a, _ = ca2d.run_life(g, 4)
    shifted = np.roll(np.roll(g, 1, 0), 1, 1)
    assert np.array_equal(a, shifted)


def test_gosper_gun_stays_alive():
    gun = ca2d.pattern_grid(ca2d.GOSPER_GLIDER_GUN, 90, 130, top=5, left=5)
    _, age = ca2d.run_life(gun, 120)
    # a working gun keeps emitting; population must not die out
    assert (age > 0).sum() > 30


def test_ecosystem_run_until_stops_early_on_extinction():
    eco = Ecosystem(n=30, rng=np.random.default_rng(0))
    eco.grid[:] = 1                 # all plants: predators already extinct
    ran = eco.run_until(500, stop_on_extinction=True)
    assert ran == 0                 # detected immediately, no steps taken


# ------------------------------------------------ evolution (Ch. 7) ------
from cadyn import evolution as ev


def test_transition_matrix_matches_paper():
    p = ev.FITNESS_RPV
    s = np.array([0.5, 0.3, 0.2])
    W = ev.transition_matrix(p, s)
    paper = np.array([
        [0.6 + 0.4*s[0], 0.25*s[0],       0.1*s[0]],
        [0.4*s[1],       0.75 + 0.25*s[1], 0.1*s[1]],
        [0.4*s[2],       0.25*s[2],        0.9 + 0.1*s[2]],
    ])
    assert np.allclose(W, paper)
    assert np.allclose(W.sum(axis=0), 1.0)      # column-stochastic


def test_lemma_three_formulations_agree():
    p = ev.FITNESS_RPV
    assert ev.verify_lemma(p, [0.5, 0.3, 0.2])
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 5, 8):
        for _ in range(30):
            assert ev.verify_lemma(rng.random(n), rng.dirichlet(np.ones(n)))


def test_step_and_step_matrix_agree():
    p = ev.FITNESS_RPV
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = rng.dirichlet(np.ones(3))
        assert np.allclose(ev.step(p, s), ev.step_matrix(p, s))


def test_paper_numeric_result():
    # from [.5,.3,.2] the paper reports convergence to [0,0,1]
    final = ev.run(ev.FITNESS_RPV, [0.5, 0.3, 0.2], 200)[-1]
    assert np.allclose(final, [0, 0, 1], atol=1e-6)


def test_theorem_fittest_wins_3d():
    p = ev.FITNESS_RPV
    rng = np.random.default_rng(2)
    assert all(ev.dominant_strategy_wins(p, rng.dirichlet(np.ones(3)))
               for _ in range(100))


def test_theorem_higher_dimensions():
    rng = np.random.default_rng(3)
    for n in (4, 6, 10):
        pv = rng.random(n)
        assert all(ev.dominant_strategy_wins(pv, rng.dirichlet(np.ones(n)))
                   for _ in range(30))


def test_penalty_reverses_outcome():
    # cutting V's fitness below P's makes P win instead
    p_pen = np.array([0.6, 0.75, 0.55])
    final = ev.run(p_pen, [1/3, 1/3, 1/3], 300)[-1]
    assert np.argmax(final) == 1                 # P (index 1) dominates


def test_run_conserves_simplex():
    hist = ev.run(ev.FITNESS_RPV, [0.4, 0.4, 0.2], 500)
    assert np.allclose(hist.sum(axis=1), 1.0)
    assert (hist >= -1e-12).all()
