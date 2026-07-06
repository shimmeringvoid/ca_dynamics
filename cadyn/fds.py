"""
Zero-dimensional cellular automata: finite dynamical systems.

A 0-dimensional CA has a single cell taking values in Z_n, with a
transition rule x -> f(x) mod n.  This module computes trajectories,
the complete basin-of-attraction field (the discrete analogue of the
phase portrait), and Shannon entropy as a function of time.

Replaces the 1997 program ``FiniteDynamicalSystems``.
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .numtheory import square_plus_one, cubic_2_3_2, ulam


class FiniteDynamicalSystem:
    """The dynamical system  x_{t+1} = f(x_t) mod n  on states {0,...,n-1}."""

    def __init__(self, f, n, name=None):
        self.f = f
        self.n = n
        self.name = name or getattr(f, "__name__", "f")
        # successor of every state, computed once
        self.succ = np.array([f(x) % n for x in range(n)], dtype=np.int64)

    # ------------------------------------------------------------------ #
    # trajectories, cycles, basins
    # ------------------------------------------------------------------ #
    def trajectory(self, x0, steps=None):
        """Trajectory from x0.  If steps is None, run until the first
        repeated value (i.e. one full lap into the cycle is detected)."""
        traj, seen = [x0 % self.n], {x0 % self.n}
        while True:
            nxt = int(self.succ[traj[-1]])
            traj.append(nxt)
            if steps is not None and len(traj) > steps:
                return traj[: steps + 1]
            if steps is None and nxt in seen:
                return traj
            seen.add(nxt)

    def cycles(self):
        """All cycles, each returned as a list of states in cycle order."""
        n, succ = self.n, self.succ
        color = np.zeros(n, dtype=np.int8)         # 0 unvisited, 1 done
        on_cycle = np.zeros(n, dtype=bool)
        cycles = []
        for start in range(n):
            if color[start]:
                continue
            path, pos = [], {}
            x = start
            while color[x] == 0 and x not in pos:
                pos[x] = len(path)
                path.append(x)
                x = int(succ[x])
            if color[x] == 0:                      # found a brand-new cycle
                cyc = path[pos[x]:]
                cycles.append(cyc)
                on_cycle[cyc] = True
            color[path] = 1
        self._on_cycle = on_cycle
        return cycles

    def basins(self):
        """Partition the state space into basins of attraction.

        Returns a list of dicts, one per cycle:
            {'cycle': [...], 'basin': set_of_all_states_flowing_here}
        """
        cycles = self.cycles()
        cycle_id = {}
        for i, cyc in enumerate(cycles):
            for s in cyc:
                cycle_id[s] = i
        basin_sets = [set(cyc) for cyc in cycles]
        # propagate: follow every state to its cycle
        memo = dict(cycle_id)
        for start in range(self.n):
            path = []
            x = start
            while x not in memo:
                path.append(x)
                x = int(self.succ[x])
            cid = memo[x]
            for s in path:
                memo[s] = cid
                basin_sets[cid].add(s)
        return [{"cycle": cyc, "basin": basin_sets[i]}
                for i, cyc in enumerate(cycles)]

    def garden_of_eden_states(self):
        """States with no predecessor (unreachable after one step)."""
        has_pred = np.zeros(self.n, dtype=bool)
        has_pred[self.succ] = True
        return np.flatnonzero(~has_pred).tolist()

    # ------------------------------------------------------------------ #
    # entropy
    # ------------------------------------------------------------------ #
    def entropy_curve(self, t_max):
        """Shannon entropy H_t (bits) of the state distribution after t
        iterations from a uniformly random initial state, t = 0..t_max."""
        H = np.empty(t_max + 1)
        state = np.arange(self.n)                  # image of every initial state
        for t in range(t_max + 1):
            counts = np.bincount(state, minlength=self.n)
            p = counts[counts > 0] / self.n
            H[t] = float(-(p * np.log2(p)).sum())
            state = self.succ[state]
        return H

    # ------------------------------------------------------------------ #
    # plotting
    # ------------------------------------------------------------------ #
    def to_graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(range(self.n))
        g.add_edges_from((x, int(self.succ[x])) for x in range(self.n))
        return g

    def _basin_layout(self, basin, edge_len=1.0, refine=0):
        """Positions for one basin: cycle nodes on a circle, transient
        trees fanned radially outward, each subtree confined to an
        angular wedge proportional to its number of leaves.

        If refine > 0, follow with that many iterations of
        force-directed relaxation (nodes repel, edges act as unit-length
        springs) with the cycle nodes held fixed, letting the transient
        branches pivot apart -- exactly the 'mutually repelling nodes on
        fixed-length struts' picture.
        """
        cycle, members = basin["cycle"], basin["basin"]
        cyc_set = set(cycle)

        # transient predecessor trees: preds of each node, minus the cycle
        preds = {v: [] for v in members}
        for v in members:
            s = int(self.succ[v])
            if v not in cyc_set and s in preds:
                preds[s].append(v)
        for v in preds:
            preds[v].sort()

        # leaf counts weight the angular wedges
        weight = {}
        def leaves(v):
            if v in weight:
                return weight[v]
            kids = preds[v]
            weight[v] = 1 if not kids else sum(leaves(c) for c in kids)
            return weight[v]
        for v in members:
            leaves(v)

        pos = {}
        m = len(cycle)
        if m == 1:                      # fixed point at the wedge center
            r_cyc = 0.0
            pos[cycle[0]] = np.zeros(2)
            wedges = {cycle[0]: (0.0, 2 * np.pi)}
        else:
            # circle large enough that adjacent cycle nodes sit ~edge_len apart
            r_cyc = max(edge_len * m / (2 * np.pi), edge_len * 0.9)
            wedges = {}
            for i, v in enumerate(cycle):
                th = 2 * np.pi * i / m
                pos[v] = r_cyc * np.array([np.cos(th), np.sin(th)])
                half = np.pi / m          # each cycle node owns 2*pi/m outward
                wedges[v] = (th - half, th + half)

        # recursively place each transient tree inside its wedge
        def place(v, lo, hi, depth):
            kids = preds[v]
            if not kids:
                return
            total = sum(weight[c] for c in kids)
            a = lo
            for c in kids:
                b = a + (hi - lo) * weight[c] / total
                th = 0.5 * (a + b)
                r = r_cyc + depth * edge_len
                pos[c] = r * np.array([np.cos(th), np.sin(th)])
                place(c, a, b, depth + 1)
                a = b
        for v in cycle:
            lo, hi = wedges[v]
            place(v, lo, hi, 1)

        if refine:
            sub = nx.Graph()
            sub.add_nodes_from(members)
            sub.add_edges_from((v, int(self.succ[v])) for v in members
                               if int(self.succ[v]) in members and
                               int(self.succ[v]) != v)
            pos = nx.spring_layout(sub, pos=pos, fixed=list(cyc_set),
                                   k=edge_len, iterations=refine, seed=1)
            pos = {v: np.asarray(p) for v, p in pos.items()}
        return pos

    def plot_basin_field(self, ax=None, node_size=None, font_size=None,
                         edge_len=1.0, refine=50, pad=1.6):
        """Draw the basin-of-attraction field in the style of the
        original figures: each basin is laid out radially (cycle in the
        center as a polygon, transients branching outward), and the
        basins are packed side by side.

        Cycle edges red, transient edges green, node labels blue.
        Set refine > 0 (e.g. 50) for a force-directed relaxation pass
        with the cycle pinned.
        """
        basins = self.basins()
        layouts = [self._basin_layout(b, edge_len, refine) for b in basins]

        # give small basins a modest scale boost so a 6-node basin isn't
        # dwarfed beside a 23-node one: below `ref` states, positions are
        # magnified by sqrt(ref/size), capped at `max_boost`.
        ref, max_boost = 14, 1.6
        for lay, b in zip(layouts, basins):
            size = len(b["basin"])
            if size < ref:
                f = min(max_boost, np.sqrt(ref / size))
                for v in lay:
                    lay[v] = lay[v] * f

        # pack the basins on a row-major grid, spaced by their true radii
        radii = [max((np.linalg.norm(p) for p in lay.values()), default=1.0)
                 + pad for lay in layouts]
        order = np.argsort(radii)[::-1]           # biggest first
        ncols = int(np.ceil(np.sqrt(len(layouts))))
        pos, x, y, row_h, col = {}, 0.0, 0.0, 0.0, 0
        for idx in order:
            r = radii[idx]
            if col == ncols:
                col, x, y, row_h = 0, 0.0, y - row_h * 2, 0.0
            center = np.array([x + r, y - r])
            for v, p in layouts[idx].items():
                pos[v] = p + center
            x += 2 * r
            row_h = max(row_h, r)
            col += 1

        # scale marker/font sizes to the number of nodes
        if node_size is None:
            node_size = max(40, int(9000 / max(self.n, 1)))
        if font_size is None:
            font_size = 8 if self.n <= 60 else 6

        if ax is None:
            side = max(7, 1.9 * np.sqrt(sum(r * r for r in radii)))
            _, ax = plt.subplots(figsize=(min(side, 14), min(side, 14) * 0.8))

        g = self.to_graph()
        on_cycle = self._on_cycle
        cyc_edges = [(u, v) for u, v in g.edges()
                     if on_cycle[u] and on_cycle[v] and u != v]
        self_loops = [(u, v) for u, v in g.edges() if u == v]
        trans_edges = [e for e in g.edges()
                       if e not in set(cyc_edges) and e not in set(self_loops)]

        nx.draw_networkx_edges(g, pos, edgelist=trans_edges, ax=ax,
                               edge_color="green", arrows=True,
                               arrowsize=7, width=1.0,
                               node_size=node_size)
        nx.draw_networkx_edges(g, pos, edgelist=cyc_edges, ax=ax,
                               edge_color="red", arrows=True,
                               arrowsize=8, width=1.8,
                               connectionstyle="arc3,rad=0.12",
                               node_size=node_size)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size,
                               node_color="white", edgecolors="none")
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=font_size,
                                font_color="blue")
        # mark fixed points (self-loops) with a red ring
        fixed_pts = [u for u, _ in self_loops]
        if fixed_pts:
            nx.draw_networkx_nodes(g, pos, nodelist=fixed_pts, ax=ax,
                                   node_size=node_size * 2.2,
                                   node_color="none", edgecolors="red",
                                   linewidths=1.5)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(f"Basin of attraction field:  {self.name}  (mod {self.n})")
        return ax

    def plot_entropy(self, t_max, ax=None):
        H = self.entropy_curve(t_max)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(t_max + 1), H, color="red", lw=2)
        ax.axhline(H[0], color="0.8", lw=0.5)
        ax.axhline(H.min(), color="0.8", lw=0.5)
        ax.set_yticks(sorted({round(H[0], 3), round(H.min(), 3)}))
        ax.set_xlabel("iterations, $t$")
        ax.set_ylabel("entropy $H_t$ (bits)")
        ax.set_title(f"Entropy vs. time:  {self.name}  (mod {self.n})")
        return ax


# ---------------------------------------------------------------------- #
# ready-made systems from Chapter 2
# ---------------------------------------------------------------------- #
def example_2_1():
    return FiniteDynamicalSystem(square_plus_one, 17, "x^2 + 1")

def example_2_3():
    return FiniteDynamicalSystem(square_plus_one, 37, "x^2 + 1")

def example_2_4():
    return FiniteDynamicalSystem(square_plus_one, 221, "x^2 + 1")

def example_2_5():
    return FiniteDynamicalSystem(cubic_2_3_2, 28, "x^3 + 2x^2 + 3x + 2")

def ulam_mod(n):
    return FiniteDynamicalSystem(ulam, n, "Ulam (3N+1)")
