import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNPLoss(nn.Module):
    def __init__(
        self,
        rank_weight: float = 0.3,
        nll_weight: float = 1.0,
        lambda_rank_weight: float = 0.1,
    ):
        super().__init__()
        self.rank_weight = rank_weight
        self.nll_weight = nll_weight
        self.lambda_rank_weight = lambda_rank_weight

    def step_schedule(self, episode: int, total_episodes: int):
        pass

    def forward(
        self,
        mu: torch.Tensor,      # [N]
        sigma: torch.Tensor,   # [N]
        labels: torch.Tensor,  # [N]
    ) -> dict:
        # Gaussian NLL
        nll = 0.5 * math.log(2 * math.pi) + torch.log(sigma) + (labels - mu) ** 2 / (2 * sigma ** 2)
        nll_loss = nll.mean()

        # ListNet ranking loss
        target_probs = F.softmax(labels, dim=0)
        log_pred_probs = F.log_softmax(mu, dim=0)
        listnet_loss = -(target_probs * log_pred_probs).sum()

        # Lambda-MART pairwise loss
        pairs = (labels.unsqueeze(0) - labels.unsqueeze(1)) > 0.5
        if pairs.any():
            pred_diff = mu.unsqueeze(0) - mu.unsqueeze(1)
            lambda_loss = F.softplus(-pred_diff[pairs]).mean()
        else:
            lambda_loss = torch.tensor(0.0, device=mu.device)

        total = (
            self.nll_weight * nll_loss
            + self.rank_weight * (listnet_loss + self.lambda_rank_weight * lambda_loss)
        )

        return {
            "total_loss": total,
            "nll": nll_loss.detach(),
            "listnet": listnet_loss.detach(),
            "lambda": lambda_loss.detach(),
        }
