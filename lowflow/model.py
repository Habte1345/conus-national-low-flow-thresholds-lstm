"""LSTM with a probabilistic threshold head.

h_t = LSTM([x_t, s])                     x_t daily forcing (+ day-of-year terms), s static attributes
[a, b] = W h_t + c                       for each of the last K days
mu_1 = a_1,  mu_k = mu_(k-1) + softplus(a_k)      ('vt': P5 <= P10 <= ... <= P30 by construction)
mu_k = a_k                                        ('fq')
sigma_k = softplus(b_k) + sigma_min
The predictive distribution of log(T + eps) is N(mu, sigma^2) in standardized units.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_MIN = 1e-3


class ThresholdLSTM(nn.Module):
    def __init__(self, n_in, n_out, hidden=128, dropout=0.2, forget_bias=3.0, monotone=True):
        super().__init__()
        self.n_out, self.monotone = n_out, monotone
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        with torch.no_grad():                                              # gate order i, f, g, o
            self.lstm.bias_hh_l0[hidden:2 * hidden].fill_(forget_bias)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 2 * n_out)

    def forward(self, x, k_last=1):
        h, _ = self.lstm(x)
        out = self.head(self.drop(h[:, -k_last:, :]))                      # (B, K, 2n)
        a, b = out[..., :self.n_out], out[..., self.n_out:]
        if self.monotone:
            a = torch.cat([a[..., :1], a[..., :1] + torch.cumsum(F.softplus(a[..., 1:]), dim=-1)], dim=-1)
        return a, F.softplus(b) + SIGMA_MIN
