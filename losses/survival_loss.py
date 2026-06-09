#losses/survival_loss.py

import torch
import torch.nn as nn


class DiscreteTimeSurvivalLoss(nn.Module):
    def __init__(self, bin_width: float = 6.0, eps: float = 1e-7,
                 alpha: float = 0.0, sigma: float = 0.1):
        super().__init__()
        self.bin_width = float(bin_width)
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.sigma = float(sigma)

    def _ranking_loss(self, hazards, idx, events):
        """
        DeepHit-style concordance ranking loss.

        Penalizes pairs (i,j) where subject i had an event before j,
        but the model assigns a lower CIF to i at time t_i.

        hazards: [B, T] hazard rates in (0,1)
        idx:     [B]    discrete bin indices (from times)
        events:  [B]    1=event, 0=censored
        """
        h = hazards.clamp(self.eps, 1.0 - self.eps)

        # PMF: p(T=t) = h_t * prod_{j<t}(1 - h_j)
        log_1mh = torch.log1p(-h)                       # [B, T]
        log_h   = torch.log(h)                           # [B, T]
        # shifted cumsum: sum_{j<t} log(1-h_j)
        cum_log_1mh = torch.cumsum(log_1mh, dim=1)       # [B, T]
        shifted = torch.cat([
            torch.zeros(h.size(0), 1, device=h.device),
            cum_log_1mh[:, :-1]
        ], dim=1)                                         # [B, T]
        log_pmf = log_h + shifted                         # [B, T]
        pmf = torch.exp(log_pmf)                          # [B, T]

        # CIF at each subject's event time bin
        cif = pmf.cumsum(dim=1)                           # [B, T]
        cif_at_t = cif[torch.arange(len(idx)), idx]      # [B]

        # Comparable pairs: event_i=1 and idx_i < idx_j
        e_i = (events == 1).unsqueeze(1)                  # [B, 1]
        t_i = idx.unsqueeze(1)                             # [B, 1]
        t_j = idx.unsqueeze(0)                             # [1, B]
        rank_mat = e_i & (t_i < t_j)                      # [B, B]

        n_pairs = rank_mat.sum()
        if n_pairs == 0:
            return hazards.new_tensor(0.0)

        # eta(i,j) = exp(-(CIF_i(t_i) - CIF_j(t_i)) / sigma)
        # cif[:, idx].T[i, j] = cif[j, idx[i]] = CIF_j evaluated at t_i (not t_j)
        cif_at_ti = cif[:, idx].T  # [B, B]: entry [i,j] = CIF_j(t_i)
        diff = cif_at_t.unsqueeze(1) - cif_at_ti  # [B, B]: CIF_i(t_i) - CIF_j(t_i)
        loss = (rank_mat.float() * torch.exp(-diff / self.sigma)).sum()
        return loss / n_pairs

    def forward(self, hazards, times, events):
        """
        hazards: [B, T] in (0,1)  (if your model outputs logits, apply sigmoid before calling loss)
        times:   [B]  (months)
        events:  [B]  (0=censored, 1=event)
        """
        B, T = hazards.shape
        device = hazards.device

        # Bins are (0,w], (w,2w], ... so idx = ceil(t/w) - 1
        idx = torch.ceil(times / self.bin_width).long() - 1
        idx = torch.clamp(idx, min=0, max=T - 1)

        # Numerically stable logs
        h = hazards.clamp(self.eps, 1.0 - self.eps)  # [B,T]
        log_h = torch.log(h)                         # [B,T]
        log_1mh = torch.log1p(-h)                    # [B,T]

        # cumulative log survival: S(t) = prod_{j<=t} (1-h_j)
        cumsum_log_1mh = torch.cumsum(log_1mh, dim=1)  # [B,T]

        # log survival through (idx-1)
        prev_idx = (idx - 1).clamp(min=0)  # [B]
        log_surv_before = cumsum_log_1mh.gather(1, prev_idx.unsqueeze(1)).squeeze(1)  # [B]
        log_surv_before = torch.where(idx > 0, log_surv_before, torch.zeros(B, device=device))

        # log survival through idx (used for censoring)
        log_surv_at = cumsum_log_1mh.gather(1, idx.unsqueeze(1)).squeeze(1)  # [B]

        # log hazard at idx
        log_h_at = log_h.gather(1, idx.unsqueeze(1)).squeeze(1)  # [B]

        events = events.long()
        event_mask = (events == 1)
        cens_mask = (events == 0)

        loglik = torch.zeros(B, device=device)
        loglik[event_mask] = log_surv_before[event_mask] + log_h_at[event_mask]  # event at idx
        loglik[cens_mask] = log_surv_at[cens_mask]                                # censored at idx

        nll = -loglik.mean()

        if self.alpha == 0.0:
            return nll

        rank = self._ranking_loss(hazards, idx, events)
        return nll + self.alpha * rank