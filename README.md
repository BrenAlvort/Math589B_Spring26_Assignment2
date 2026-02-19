# Programming Assignment 2 (Autograded): Numerical Quadrature + Interpolation

You will submit **one file** to Gradescope:

- `student.py`

It must define the required functions with the exact signatures described below.

## Allowed libraries

You may use:
- `numpy`
- `mpmath`

(You may use standard library modules too.)

## What to implement (required API)

Implement these functions in `student.py`:

```python
def composite_simpson(f, a, b, n_panels):
    """Composite Simpson's rule with n_panels panels (2 subintervals per panel)."""

def gauss_legendre(f, a, b, n_nodes):
    """Gauss-Legendre quadrature with n_nodes on [a,b]."""

def romberg(f, a, b, n):
    """Romberg integration; return R[n,n]."""

def equispaced_interpolant_values(f, n, x_eval):
    """Evaluate the degree-n interpolant at equispaced nodes on [-1,1] at x_eval."""

def chebyshev_lobatto_interpolant_values(f, n, x_eval):
    """Evaluate the degree-n interpolant at Chebyshev-Lobatto nodes on [-1,1] at x_eval."""

def poly_integral_from_values(x_nodes, y_nodes):
    """Return integral over [-1,1] of interpolating polynomial through (x_nodes, y_nodes)."""
```

## Local sanity checks

If you have Python 3.10+:

```bash
pip install -r requirements.txt
pytest -q
```

The included tests are only a **small public subset** of what Gradescope will run.

## Notes

- `f` will be called with scalar floats.
- Your implementations should not print during grading.
- Use numerically stable interpolation (barycentric strongly recommended).
