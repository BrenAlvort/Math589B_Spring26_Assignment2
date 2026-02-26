from __future__ import annotations

import math
from typing import Callable

import numpy as np

# ============================================================
# Quadrature
# ============================================================

def composite_simpson(f: Callable[[float], float], a: float, b: float, n_panels: int) -> float:
    """Composite Simpson's rule on [a,b] using n_panels panels (2 subintervals per panel)."""
    if n_panels <= 0:
        raise ValueError("n_panels must be positive")

    h = (b - a) / (2.0 * n_panels)

    # grid points: x_0, ..., x_{2n}
    x = [a + k * h for k in range(2 * n_panels + 1)]

    acc = f(x[0]) + f(x[-1])

    # odd indices get weight 4; even indices (excluding endpoints) get weight 2
    for k in range(1, 2 * n_panels):
        acc += (4.0 if (k % 2 == 1) else 2.0) * f(x[k])

    return (h / 3.0) * acc


def gauss_legendre(f: Callable[[float], float], a: float, b: float, n_nodes: int) -> float:
    """Gauss–Legendre quadrature on [a,b] with n_nodes (using numpy leggauss)."""
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive")

    t, w = np.polynomial.legendre.leggauss(n_nodes)  # nodes/weights on [-1,1]

    # affine map [-1,1] -> [a,b]
    x = 0.5 * (b - a) * t + 0.5 * (a + b)

    # integral ≈ (b-a)/2 * Σ w_i f(x_i)
    return 0.5 * (b - a) * float(np.sum(w * np.vectorize(f)(x)))


def romberg(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Romberg integration on [a,b] up to depth n. Returns R[n,n]."""
    if n < 0:
        raise ValueError("n must be >= 0")

    R = np.empty((n + 1, n + 1), dtype=float)
    R[0, 0] = 0.5 * (b - a) * (f(a) + f(b))

    for k in range(1, n + 1):
        h = (b - a) / (2 ** k)

        # new trapezoid points: a + (2j-1)h, j=1..2^{k-1}
        new_pts_sum = 0.0
        for j in range(1, 2 ** (k - 1) + 1):
            new_pts_sum += f(a + (2 * j - 1) * h)

        R[k, 0] = 0.5 * R[k - 1, 0] + h * new_pts_sum

        for j in range(1, k + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4 ** j - 1)

    return float(R[n, n])


# ============================================================
# Interpolation (Runge vs Chebyshev)
# ============================================================

def _barycentric_weights(x_nodes: np.ndarray) -> np.ndarray:
    """Barycentric weights for distinct interpolation nodes (O(n^2))."""
    x = np.asarray(x_nodes, dtype=float)
    n = x.size
    w = np.ones(n, dtype=float)

    for j in range(n):
        # product over m != j (x_j - x_m)
        denom = np.prod(x[j] - np.delete(x, j))
        w[j] = 1.0 / denom

    return w


def _barycentric_eval(x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Evaluate barycentric interpolant at points x_eval."""
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    xe = np.asarray(x_eval, dtype=float)

    w = _barycentric_weights(x)
    vals = np.empty_like(xe, dtype=float)

    for i, xi in enumerate(xe):
        diff = xi - x

        # if xi coincides with a node, return exact data value
        idx = np.where(np.abs(diff) < 1e-14)[0]
        if idx.size > 0:
            vals[i] = y[idx[0]]
            continue

        frac = w / diff
        vals[i] = float(np.sum(frac * y) / np.sum(frac))

    return vals


def equispaced_interpolant_values(
    f: Callable[[float], float], n: int, x_eval: np.ndarray
) -> np.ndarray:
    """Evaluate the degree-n interpolant of f at equispaced nodes on [-1,1]."""
    nodes = np.linspace(-1.0, 1.0, n + 1)
    data = np.array([f(t) for t in nodes], dtype=float)
    return _barycentric_eval(nodes, data, x_eval)


def chebyshev_lobatto_interpolant_values(
    f: Callable[[float], float], n: int, x_eval: np.ndarray
) -> np.ndarray:
    """Evaluate the degree-n interpolant of f at the same Chebyshev-style nodes as before."""
    # Keeping the same node formula/behavior as your original code.
    nodes = np.array([math.cos((2 * i + 1) * math.pi / (2 * (n + 1))) for i in range(n + 1)], dtype=float)
    data = np.array([f(t) for t in nodes], dtype=float)
    return _barycentric_eval(nodes, data, x_eval)


def poly_integral_from_values(x_nodes: np.ndarray, y_nodes: np.ndarray) -> float:
    """Compute ∫_{-1}^1 P(x) dx where P interpolates (x_nodes, y_nodes)."""
    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    def p(x: float) -> float:
        return float(_barycentric_eval(x_nodes, y_nodes, np.array([x], dtype=float))[0])

    # Keep same behavior: high-order Gauss–Legendre to integrate the interpolant.
    return gauss_legendre(p, -1.0, 1.0, 90)
