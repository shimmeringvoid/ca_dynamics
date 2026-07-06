"""
cadyn -- Cellular Automata Dynamics
===================================

A Python re-implementation of the software accompanying
*Cellular Automata Dynamics: Explorations in Parallel Processing*
(Rafael Espericueta, 1997), originally written in C++ for Windows 95.

Modules
-------
fds        0-dimensional CA (finite dynamical systems): trajectories,
           basin-of-attraction fields, Shannon entropy.
           (replaces the program ``FiniteDynamicalSystems``)
ca1d       1-dimensional CA: additive rules as convolution kernels,
           non-additive rules, space-time diagrams.
           (replaces ``CA1dim``)
ca2d       2-dimensional CA: additive kernels, the Game of Life,
           the Euler-totient rule.
           (replaces ``CA2dim``)
ecosystem  Three-species (plant / prey / predator) stochastic CA
           ecosystem with population tracking and phase portraits.
           (replaces ``CAecosystem``)
evolution  Mean-field model of evolutionary strategy competition
           (Chapter 7): the veridical-perception dynamics, with the
           Lemma and (empirical) Theorem made checkable.
numtheory  Small number-theoretic helpers (Euler totient, Ulam/Collatz).
palettes   Color maps matching the spirit of the original figures.
"""

from . import fds, ca1d, ca2d, ecosystem, evolution, numtheory, palettes

__version__ = "1.0.0"
__all__ = ["fds", "ca1d", "ca2d", "ecosystem", "evolution",
           "numtheory", "palettes"]
