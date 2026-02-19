import os
import sys

# Ensure repo root is on path so `import student` works
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import numpy as np
import mpmath as mp

import student


def test_simpson_exact_on_cubic():
    # Simpson should integrate any cubic exactly (up to roundoff).
    f = lambda x: 2.0*x**3 - 3.0*x**2 + 5.0*x - 7.0
    a, b = -1.2, 0.7
    exact = (2*(b**4-a**4)/4) - (3*(b**3-a**3)/3) + (5*(b**2-a**2)/2) - 7*(b-a)
    approx = student.composite_simpson(f, a, b, n_panels=5)
    assert abs(approx - exact) < 5e-12


def test_gauss_legendre_exact_polynomial_degree_5():
    # n_nodes=3 Gauss-Legendre is exact for degree <= 5.
    f = lambda x: x**5 - 2*x**3 + 4*x + 1
    a, b = -0.5, 1.3
    # exact integral
    exact = (b**6-a**6)/6 - 2*(b**4-a**4)/4 + 4*(b**2-a**2)/2 + (b-a)
    approx = student.gauss_legendre(f, a, b, n_nodes=3)
    assert abs(approx - exact) < 1e-12


def test_romberg_reasonable_on_smooth_function():
    f = lambda x: np.exp(-x**2)
    a, b = 0.0, 1.0
    approx = student.romberg(f, a, b, n=3)
    exact = float(mp.quad(lambda t: mp.e**(-t**2), [a, b]))
    assert abs(approx - exact) < 5e-10
