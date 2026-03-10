import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Learnable log-variances for homoscedastic uncertainty weighting
        # Initializing at 0 means the initial weight is exp(0) = 1.0
        self.log_var_nll = nn.Parameter(torch.zeros(1))
        self.log_var_listnet = nn.Parameter(torch.zeros(1))
        self.log_var_lambda = nn.Parameter(torch.zeros(1))

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

        # ListNet ranking loss with temperature to prevent winner-takes-all collapse
        # Raw pIC50s (5-9) produce very peaky softmax; T=2 gives meaningful gradients
        # across the full ranked list rather than just the top-1
        T = 2.0
        target_probs = F.softmax(labels / T, dim=0)
        log_pred_probs = F.log_softmax(mu / T, dim=0)
        listnet_loss = -(target_probs * log_pred_probs).sum()

        # Lambda-MART pairwise loss
        pairs = (labels.unsqueeze(0) - labels.unsqueeze(1)) > 0.5
        if pairs.any():
            pred_diff = mu.unsqueeze(0) - mu.unsqueeze(1)
            lambda_loss = F.softplus(-pred_diff[pairs]).mean()
        else:
            lambda_loss = torch.tensor(0.0, device=mu.device)

        # Homoscedastic Uncertainty Weighting
        # Formula: L_weighted = exp(-log_var) * L + 0.5 * log_var
        w_nll = torch.exp(-self.log_var_nll)
        w_listnet = torch.exp(-self.log_var_listnet)
        w_lambda = torch.exp(-self.log_var_lambda)

        total = (
            (w_nll * nll_loss + 0.5 * self.log_var_nll)
            + (w_listnet * listnet_loss + 0.5 * self.log_var_listnet)
            + (w_lambda * lambda_loss + 0.5 * self.log_var_lambda)
        )

        return {
            "total_loss": total,
            "nll": nll_loss.detach(),
            "listnet": listnet_loss.detach(),
            "lambda": lambda_loss.detach(),
            "w_nll": w_nll.detach(),          # Track these to watch them evolve
            "w_listnet": w_listnet.detach(),
            "w_lambda": w_lambda.detach(),
        }