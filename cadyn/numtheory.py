"""Small number-theoretic helpers: Euler's totient, the Ulam (Collatz)
function, and a few convenience routines used elsewhere in the package."""

from functools import lru_cache

import numpy as np


def totient_table(limit):
    """Return array phi[0..limit-1] of Euler totient values.

    phi(0) is defined as 0 and phi(1) as 1, matching the convention used
    for the CA transition rule in Chapter 4 (cell value 0 contributes 0).

    Uses a linear sieve, O(limit).
    """
    phi = np.arange(limit, dtype=np.int64)
    phi[1] = 1
    for p in range(2, limit):
        if phi[p] == p:               # p is prime
            phi[p::p] -= phi[p::p] // p
    if limit > 0:
        phi[0] = 0
    return phi


@lru_cache(maxsize=None)
def totient(n):
    """Euler's totient of a single integer n >= 0."""
    if n < 2:
        return n
    result, m, p = n, n, 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if m > 1:
        result -= result // m
    return result


def ulam(n):
    """One step of the Ulam / Collatz '3N + 1' function."""
    return n // 2 if n % 2 == 0 else 3 * n + 1


def square_plus_one(x):
    """The workhorse example of Chapter 2:  f(x) = x^2 + 1."""
    return x * x + 1


def cubic_2_3_2(x):
    """f(x) = x^3 + 2x^2 + 3x + 2   (Section 2.5)."""
    return x**3 + 2 * x**2 + 3 * x + 2
