"""
CRPS loss for Gaussian predictive distributions.

Theory
------
The CRPS is a strictly proper scoring rule (Gneiting & Raftery 2007, Sec. 4.2), so the
optimum-score estimator  theta_hat = argmin_theta mean_i CRPS(F_theta, x_i)  is consistent
(Gneiting & Raftery 2007, Sec. 9.1, eq. 59). For F = N(mu, sigma^2) the CRPS has the closed
form (Gneiting & Raftery 2007, eq. 21; Gneiting et al. 2005, eq. 5)

    CRPS(N(mu, s^2), y) = s * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],   z = (y - mu)/s,

which is differentiable in (mu, s); gradients flow by ordinary backpropagation.
Training a neural network with this loss follows Rasp & Lerch (2018).
"""
import math

import torch
import torch.nn as nn

_SQRT2 = math.sqrt(2.0)
_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise closed-form CRPS of N(mu, sigma^2) against observation y."""
    z = (y - mu) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / _SQRT2))
    phi = torch.exp(-0.5 * z * z) / _SQRT_2PI
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / _SQRT_PI)


class CRPSLoss(nn.Module):
    """Mean CRPS over all (sample, target) pairs with a finite observation.

    NaN targets (e.g. a missing LP3 fit for one site) are masked out, so a site with 4 of
    5 valid thresholds still contributes gradients from the 4.
    """

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            return (mu * 0.0).sum()
        return crps_gaussian(mu[mask], sigma[mask], y[mask]).mean()


class MaskedMSELoss(nn.Module):
    """Point-estimate baseline on mu (same NaN masking). Used with --loss mse."""

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            return (mu * 0.0).sum()
        return ((mu[mask] - y[mask]) ** 2).mean()
