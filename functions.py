"""Useful functions."""

import numpy as _np


def polyfit(x, y, monomials):
    """Implement Custom polyfit."""
    coef = _np.polynomial.polynomial.polyfit(x, y, deg=monomials)

    # finds maximum diff and its base value
    y_fitted = _np.polynomial.polynomial.polyval(x, coef)
    y_diff = abs(y_fitted - y)
    idx = _np.argmax(y_diff)

    coeffs = coef[monomials]
    return (coeffs, (y_diff[idx], y[idx]))
