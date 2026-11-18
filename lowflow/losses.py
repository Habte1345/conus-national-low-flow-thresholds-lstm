"""Continuous ranked probability score of a Gaussian predictive distribution.

CRPS(N(mu, s^2), y) = s [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],  z = (y - mu) / s
(Gneiting & Raftery, 2007, eq. 21). Strictly proper, so its minimizer is the true distribution.
"""
import math

import torch

_SQ2, _SQPI, _SQ2PI = math.sqrt(2.0), math.sqrt(math.pi), math.sqrt(2.0 * math.pi)


def crps_gaussian(mu, sigma, y):
    z = (y - mu) / sigma
    return sigma * (z * torch.erf(z / _SQ2) + 2.0 * torch.exp(-0.5 * z * z) / _SQ2PI - 1.0 / _SQPI)


def crps_loss(mu, sigma, y):
    """Mean CRPS over finite targets (missing thresholds, e.g. an unfitted nQy, are masked)."""
    m = torch.isfinite(y)
    if m.sum() == 0:
        return (mu * 0.0).sum()
    return crps_gaussian(mu[m], sigma[m], y[m]).mean()
