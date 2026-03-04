import torch
import torch.nn as nn
import torch.nn.functional as F


class EBLLoss(nn.Module):
    def __init__(
        self,
        ebl_alpha=0.5,
        temperature=0.05,
        eps=0.15,
        rank_weight=1.0,
        lambda_rank_weight=0.5,
        min_temperature=0.02,
        min_eps=0.02,
    ):
        super().__init__()
        self.ebl_alpha = ebl_alpha
        self.base_temperature = temperature
        self.base_eps = eps
        self.temperature = temperature
        self.eps = eps
        self.rank_weight = rank_weight
        self.lambda_rank_weight = lambda_rank_weight
        self.min_temperature = min_temperature
        self.min_eps = min_eps

    def step_schedule(self, step_idx, total_steps):
        progress = min(max(step_idx / max(total_steps, 1), 0.0), 1.0)
        self.temperature = self.base_temperature * (1.0 - progress) + self.min_temperature * progress
        self.eps = self.base_eps * (1.0 - progress) + self.min_eps * progress

    def _compute_listnet_loss(self, preds, labels):
        if len(preds) < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        target_probs = F.softmax(labels / 0.5, dim=0)
        pred_probs = F.log_softmax(preds, dim=0)
        return -torch.sum(target_probs * pred_probs)

    def _compute_lambda_ci_loss(self, preds, labels):
        if len(preds) < 2:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        diff_pred = preds.unsqueeze(1) - preds.unsqueeze(0)
        diff_true = labels.unsqueeze(1) - labels.unsqueeze(0)
        pair_mask = diff_true > 0
        if not pair_mask.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        margins = diff_pred[pair_mask]
        # logistic pairwise ranking surrogate for CI maximization
        return F.softplus(-margins).mean()

    def forward(self, final_preds, true_labels, gate_probs, expert_tensor, protein_level_gate=False):
        mse_per_expert = (expert_tensor - true_labels.unsqueeze(1)) ** 2  # [N, K]
        mean_mse = mse_per_expert.mean(dim=0, keepdim=True)  # [1, K]

        with torch.no_grad():
            raw_target_probs = F.softmax(-mean_mse / self.temperature, dim=-1)
            K = expert_tensor.size(1)
            target_gate_probs = raw_target_probs * (1.0 - self.eps) + (self.eps / K)

        listnet_loss = self._compute_listnet_loss(final_preds, true_labels)
        lambda_loss = self._compute_lambda_ci_loss(final_preds, true_labels)
        ranking_loss = listnet_loss + self.lambda_rank_weight * lambda_loss

        if protein_level_gate:
            # Gate loss is computed once per protein episode: [1, K] vs [1, K].
            protein_gate_probs = gate_probs[:1]
            gate_loss = -torch.sum(target_gate_probs * torch.log(protein_gate_probs + 1e-12), dim=-1).mean()
        else:
            # Backward-compatible behavior for per-example gate outputs [N, K].
            gate_loss = -torch.sum(target_gate_probs * torch.log(gate_probs + 1e-12), dim=-1).mean()

        expert_loss = torch.sum(mean_mse * target_gate_probs.detach(), dim=-1).mean()
        total_loss = expert_loss + (self.ebl_alpha * gate_loss) + (self.rank_weight * ranking_loss)

        return {
            "total_loss": total_loss,
            "expert_loss": expert_loss.detach(),
            "gate_loss": gate_loss.detach(),
            "rank_loss": ranking_loss.detach(),
            "listnet_loss": listnet_loss.detach(),
            "lambda_ci_loss": lambda_loss.detach(),
        }
