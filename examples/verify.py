"""Verify the Python package against results stated in the 1997 book."""
import sys
sys.path.insert(0, "/home/claude/ca_dynamics")
import numpy as np
from cadyn import fds, ca1d

ok = True
def check(label, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {label} {extra}")

# ---- Section 2.1: x^2+1 mod 17 ------------------------------------------
s = fds.example_2_1()
traj = s.trajectory(0)
check("2.1 trajectory from 0", traj[:9] == [0, 1, 2, 5, 9, 14, 10, 16, 2], traj[:9])
cycs = s.cycles()
check("2.1 single 6-cycle {2,5,9,14,10,16}",
      len(cycs) == 1 and sorted(cycs[0]) == [2, 5, 9, 10, 14, 16], cycs)
goe = s.garden_of_eden_states()
check("2.1 Garden of Eden = {4,6,7,8,11,12,13,15}",
      sorted(goe) == [4, 6, 7, 8, 11, 12, 13, 15], goe)
# predecessors sum to 17
g = s.to_graph()
preds_ok = all(sum(p for p in g.predecessors(v)) % 17 == 0
               for v in g if g.in_degree(v) == 2)
check("2.1 two predecessors of each node sum to 17 (mod 17)", preds_ok)

# ---- Section 2.3: x^2+1 mod 37 ------------------------------------------
s = fds.example_2_3()
b = s.basins()
cyc_summary = sorted((len(c["cycle"]), len(c["basin"])) for c in b)
check("2.3 cycles: fixed pts 27 & 11, 2-cycle {8,28}",
      sorted(tuple(sorted(c["cycle"])) for c in b) == [(8, 28), (11,), (27,)],
      [c["cycle"] for c in b])
check("2.3 basin sizes 8, 6, 23",
      sorted(len(c["basin"]) for c in b) == [6, 8, 23],
      sorted(len(c["basin"]) for c in b))
H = s.entropy_curve(14)
check("2.3 H_0 = log2(37) = 5.209", abs(H[0] - np.log2(37)) < 1e-3, f"{H[0]:.3f}")
check("2.3 H_min = 1.478", abs(H.min() - 1.478) < 2e-3, f"{H.min():.3f}")

# ---- Section 2.4: x^2+1 mod 221 -----------------------------------------
s = fds.example_2_4()
b = s.basins()
cyc_lens = sorted(len(c["cycle"]) for c in b)
check("2.4 four cycles: 6,6,12,12 (36 cyclic states)",
      cyc_lens == [6, 6, 12, 12], cyc_lens)
H = s.entropy_curve(6)
check("2.4 H_0 = 7.787", abs(H[0] - np.log2(221)) < 1e-3, f"{H[0]:.3f}")
check("2.4 H_min = 4.983", abs(H.min() - 4.983) < 2e-3, f"{H.min():.3f}")

# ---- Section 2.5: cubic mod 28 ------------------------------------------
s = fds.example_2_5()
b = s.basins()
check("2.5 two cycles: a 6-cycle and a 2-cycle",
      sorted(len(c["cycle"]) for c in b) == [2, 6],
      [sorted(c["cycle"]) for c in b])
H = s.entropy_curve(5)
check("2.5 H_0 = 4.807", abs(H[0] - np.log2(28)) < 1e-3, f"{H[0]:.3f}")
check("2.5 H_min = 2.761", abs(H.min() - 2.761) < 2e-3, f"{H.min():.3f}")

# ---- Section 2.6: Ulam --------------------------------------------------
for n, attractors in [(20, {(1, 2, 4), (0,)}), (47, {(1, 2, 4), (0,), (23,)})]:
    s = fds.ulam_mod(n)
    got = {tuple(sorted(c)) for c in s.cycles()}
    check(f"2.6 Ulam mod {n} cycles", got == attractors, got)

# ---- Section 3.2: kernel [1 1 1], k=5, n=21 ------------------------------
init = np.zeros(21, dtype=int); init[10] = 1
hist = ca1d.run_additive(init, [1, 1, 1], 5, 9)
check("3.2 generation 3", list(hist[3]) ==
      [0]*7 + [1, 3, 1, 2, 1, 3, 1] + [0]*7, list(hist[3]))
check("3.2 generation 9", list(hist[9]) ==
      [0, 1, 4, 0, 1, 4, 2, 4, 4, 2, 4, 2, 4, 4, 2, 4, 1, 0, 4, 1, 0])
# periodicity: book says pattern repeats; measure the true cycle length
hist_long = ca1d.run_additive(init, [1, 1, 1], 5, 400)
states = {tuple(r): t for t, r in enumerate(map(tuple, hist_long[:1]))}
first_repeat = None
seen = {}
for t, row in enumerate(map(tuple, hist_long)):
    if row in seen:
        first_repeat = (seen[row], t)
        break
    seen[row] = t
check("3.2 attractive cycle detected", first_repeat is not None, first_repeat)
if first_repeat:
    print(f"       transient enters cycle at t={first_repeat[0]}, "
          f"cycle length = {first_repeat[1]-first_repeat[0]}")

# ---- Section 3.3: kernel [1 0 1], k=2 -> Pascal mod 2 --------------------
init = np.zeros(17, dtype=int); init[8] = 1
hist = ca1d.run_additive(init, [1, 0, 1], 2, 4)
check("3.3 generation 3 = 1 0 1 0 1 0 1 pattern",
      list(hist[3][5:12]) == [1, 0, 1, 0, 1, 0, 1], list(hist[3]))

# ---- Section 3.5: polynomial representation agrees with convolution -----
n, k = 64, 5
init = np.zeros(n, dtype=int); init[n//2] = 3
kp = ca1d.kernel_to_poly([1, 2, 3], n)
a = init.copy(); b_ = init.copy()
for _ in range(20):
    a = ca1d.convolve_step(a, [1, 2, 3], k)
    b_ = ca1d.poly_step(b_, kp, k)
check("3.5 polynomial rep == convolution (20 gens, kernel [1 2 3])",
      np.array_equal(a, b_))

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
