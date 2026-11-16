"""Vectorized arithmetic modulo a prime t < 2^40 using uint64 numpy arrays.

A plain uint64 product of two 39-bit values would overflow, so mulmod splits
one operand into a 20-bit low half and a high half (Part 2, Protocol B setup).
"""

import numpy as np

_U = np.uint64
_LOW_BITS = 20
_LOW_MASK = _U((1 << _LOW_BITS) - 1)


def _check_modulus(t: int) -> None:
    if not 2 < t < (1 << 40):
        raise ValueError("modulus must be below 2^40")


def mulmod(a, b, t: int) -> np.ndarray:
    """(a * b) mod t for arrays of values already reduced mod t."""
    _check_modulus(t)
    a = np.asarray(a, dtype=_U)
    b = np.asarray(b, dtype=_U)
    tt = _U(t)
    lo = (a * (b & _LOW_MASK)) % tt                 # < 2^40 * 2^20 = 2^60
    hi = (a * (b >> _U(_LOW_BITS))) % tt            # < 2^40 * 2^20
    hi = (hi << _U(_LOW_BITS)) % tt                 # < 2^60
    return (lo + hi) % tt


def submod(a, b, t: int) -> np.ndarray:
    tt = _U(t)
    return (np.asarray(a, dtype=_U) + tt - np.asarray(b, dtype=_U)) % tt


def powmod(a, e: int, t: int) -> np.ndarray:
    """Element-wise a^e mod t by square-and-multiply."""
    a = np.asarray(a, dtype=_U) % _U(t)
    result = np.ones_like(a)
    base = a.copy()
    while e:
        if e & 1:
            result = mulmod(result, base, t)
        base = mulmod(base, base, t)
        e >>= 1
    return result


def poly_from_roots(roots: np.ndarray, t: int) -> np.ndarray:
    """Coefficients of prod_i (X - r_i) mod t, one polynomial per row.

    roots: shape (rows, d). Returns shape (rows, d + 1), lowest degree first.
    """
    roots = np.asarray(roots, dtype=_U) % _U(t)
    rows, d = roots.shape
    coeffs = np.zeros((rows, d + 1), dtype=_U)
    coeffs[:, 0] = 1
    for i in range(d):
        r = roots[:, i:i + 1]
        shifted = np.zeros_like(coeffs)
        shifted[:, 1:] = coeffs[:, :-1]           # X * P(X)
        coeffs = submod(shifted, mulmod(coeffs, r, t), t)
    return coeffs


def poly_eval(coeffs, x: int, t: int) -> int:
    """Horner evaluation with Python integers (reference used by the tests)."""
    acc = 0
    for c in reversed([int(c) for c in coeffs]):
        acc = (acc * x + c) % t
    return acc
