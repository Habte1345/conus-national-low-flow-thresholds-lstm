
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import torch


class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss.

    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the 
    discharge from the basin, to which the sample belongs.

    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, eps: float = 0.1):
        super(NSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample

        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true)**2
        weights = 1 / (q_stds + self.eps)**2
        scaled_loss = weights * squared_error

        return torch.mean(scaled_loss)



class HydroKGRealTimeLoss(nn.Module):
    """
    RMSE + HydroKG rule-based soft penalties.

    Inputs:
        qsim: simulated streamflow, shape [B, T], preferably mm/day
        qobs: observed streamflow, shape [B, T], preferably mm/day
        P: precipitation, shape [B, T], mm/day
        PET: potential evapotranspiration, shape [B, T], mm/day

    Returns:
        total_loss, loss_parts
    """

    def __init__(
        self,
        # -----------------------------
        # Rule thresholds
        # -----------------------------
        ratio_low=0.20,          # R1 lower ratio threshold
        ratio_high=5.00,         # R1 upper ratio threshold
        rel_err_thr=1.00,        # R3 relative error threshold
        peak_shift_thr=0.50,     # R4 hours
        budyko_dev_thr=0.15,     # R6 EI deviation threshold
        omega=2.6,               # Fu Budyko omega

        # -----------------------------
        # Loss weights
        # -----------------------------
        w_rmse=1.00,
        w_r0=0.50,
        w_r1=0.20,
        w_r2=0.30,
        w_r3=0.20,
        w_r4=0.10,
        w_r5=0.30,
        w_r6=0.30,

        eps=1e-6
    ):
        super().__init__()

        self.ratio_low = ratio_low
        self.ratio_high = ratio_high
        self.rel_err_thr = rel_err_thr
        self.peak_shift_thr = peak_shift_thr
        self.budyko_dev_thr = budyko_dev_thr
        self.omega = omega

        self.w_rmse = w_rmse
        self.w_r0 = w_r0
        self.w_r1 = w_r1
        self.w_r2 = w_r2
        self.w_r3 = w_r3
        self.w_r4 = w_r4
        self.w_r5 = w_r5
        self.w_r6 = w_r6

        self.eps = eps

    def fu_budyko(self, ai):
        """
        Fu Budyko curve:
        EI = 1 + AI - (1 + AI^omega)^(1/omega)
        """
        omega = self.omega
        ai = torch.clamp(ai, min=self.eps)
        return 1.0 + ai - torch.pow(1.0 + torch.pow(ai, omega), 1.0 / omega)

    def soft_peak_time(self, q, temperature=0.10):
        """
        Differentiable approximation of peak timing.
        Returns soft peak index for each batch item.
        """
        B, T = q.shape
        t = torch.arange(T, device=q.device, dtype=q.dtype).view(1, T)

        q_norm = q / (q.mean(dim=1, keepdim=True) + self.eps)
        weights = torch.softmax(q_norm / temperature, dim=1)

        return torch.sum(weights * t, dim=1)

    def forward(self, qsim, qobs, P=None, PET=None):
        eps = self.eps

        qsim = qsim.float()
        qobs = qobs.float()

        mask = torch.isfinite(qsim) & torch.isfinite(qobs)
        qsim = torch.where(mask, qsim, torch.zeros_like(qsim))
        qobs = torch.where(mask, qobs, torch.zeros_like(qobs))

        # Basin/batch adaptive flow scale
        q_mean = torch.mean(torch.clamp(qobs, min=0.0), dim=1, keepdim=True) + eps
        q_zero = 0.01 * q_mean
        q_active = 0.05 * q_mean

        # =====================================================
        # Traditional loss: RMSE
        # =====================================================
        rmse = torch.sqrt(torch.mean((qsim - qobs) ** 2) + eps)

        # =====================================================
        # R0: Negative flow
        # Penalize Qsim < 0
        # =====================================================
        r0_penalty = torch.mean((F.relu(-qsim) / q_mean) ** 2)

        # =====================================================
        # R1: Extreme prediction-to-observation ratio
        # Penalize ratio < 0.2 or ratio > 5.0
        # evaluated only when observed flow is active
        # =====================================================
        active_mask = (qobs > q_active).float()

        ratio = (qsim + eps) / (qobs + eps)
        log_ratio = torch.log(torch.clamp(ratio, min=eps))

        lower_violation = F.relu(torch.log(torch.tensor(self.ratio_low, device=qsim.device)) - log_ratio)
        upper_violation = F.relu(log_ratio - torch.log(torch.tensor(self.ratio_high, device=qsim.device)))

        r1_penalty = torch.sum(active_mask * (lower_violation ** 2 + upper_violation ** 2)) / (
            torch.sum(active_mask) + eps
        )

        # =====================================================
        # R2: Zero-flow collapse
        # Penalize Qsim near zero while Qobs is active
        # =====================================================
        r2_raw = F.relu(q_zero - qsim) / (q_zero + eps)
        r2_penalty = torch.sum(active_mask * r2_raw ** 2) / (torch.sum(active_mask) + eps)

        # =====================================================
        # R3: High relative error
        # Penalize relative error > 1.0
        # =====================================================
        rel_err = torch.abs(qsim - qobs) / (qobs + q_zero + eps)
        r3_raw = F.relu(rel_err - self.rel_err_thr)
        r3_penalty = torch.sum(active_mask * r3_raw ** 2) / (torch.sum(active_mask) + eps)

        # =====================================================
        # R4: Peak timing error
        # Differentiable soft peak timing penalty
        # Penalize peak shift > 2 days
        # =====================================================
        if qsim.shape[1] >= 5:
            peak_sim = self.soft_peak_time(qsim)
            peak_obs = self.soft_peak_time(qobs)

            peak_shift = torch.abs(peak_sim - peak_obs)
            r4_penalty = torch.mean((F.relu(peak_shift - self.peak_shift_thr) / self.peak_shift_thr) ** 2)
        else:
            r4_penalty = torch.tensor(0.0, device=qsim.device)

        # =====================================================
        # R5 and R6 need P and PET
        # =====================================================
        if P is not None and PET is not None:
            P = P.float()
            PET = PET.float()

            P = torch.where(torch.isfinite(P), P, torch.zeros_like(P))
            PET = torch.where(torch.isfinite(PET), PET, torch.zeros_like(PET))

            P_sum = torch.sum(torch.clamp(P, min=0.0), dim=1) + eps
            PET_sum = torch.sum(torch.clamp(PET, min=0.0), dim=1) + eps
            Qsim_sum = torch.sum(torch.clamp(qsim, min=0.0), dim=1)

            # =================================================
            # R5: Mass balance
            # Penalize Qsim annual/sequence sum > P sum
            # =================================================
            r5_raw = F.relu(Qsim_sum - P_sum) / P_sum
            r5_penalty = torch.mean(r5_raw ** 2)

            # =================================================
            # R6: Budyko deviation
            # EI_sim = 1 - Qsim/P
            # AI = PET/P
            # Penalize |EI_sim - EI_budyko| > threshold
            # =================================================
            ai = PET_sum / P_sum
            ei_sim = 1.0 - (Qsim_sum / P_sum)
            ei_sim = torch.clamp(ei_sim, min=0.0, max=1.5)

            ei_budyko = self.fu_budyko(ai)
            budyko_dev = torch.abs(ei_sim - ei_budyko)

            r6_raw = F.relu(budyko_dev - self.budyko_dev_thr)
            r6_penalty = torch.mean(r6_raw ** 2)

        else:
            r5_penalty = torch.tensor(0.0, device=qsim.device)
            r6_penalty = torch.tensor(0.0, device=qsim.device)

        # =====================================================
        # TOTAL LOSS
        # =====================================================
        total_loss = (
            self.w_rmse * rmse +
            self.w_r0 * r0_penalty +
            self.w_r1 * r1_penalty +
            self.w_r2 * r2_penalty +
            self.w_r3 * r3_penalty +
            self.w_r4 * r4_penalty +
            self.w_r5 * r5_penalty +
            self.w_r6 * r6_penalty
        )

        loss_parts = {
            "total_loss": total_loss.detach(),
            "rmse": rmse.detach(),
            "R0_negative_flow": r0_penalty.detach(),
            "R1_extreme_ratio": r1_penalty.detach(),
            "R2_zero_collapse": r2_penalty.detach(),
            "R3_high_relative_error": r3_penalty.detach(),
            "R4_peak_timing": r4_penalty.detach(),
            "R5_mass_balance": r5_penalty.detach(),
            "R6_budyko_deviation": r6_penalty.detach(),
        }

        return total_loss, loss_parts # New loss function returns both total loss and individual loss components for better interpretability