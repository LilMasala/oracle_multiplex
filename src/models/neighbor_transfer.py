"""Top-k neighbor activity transfer with a learned delta correction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tnp import BindingEncoder


class NeighborTransferModel(nn.Module):
    """
    Predict affinity by transferring exact-drug activity from similar proteins
    and learning only the correction from target-vs-neighbor differences.
    """

    def __init__(
        self,
        protein_dim: int,
        drug_dim: int,
        go_fp_dim: int = 0,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.go_fp_dim = go_fp_dim

        feature_dim = protein_dim * 3 + drug_dim * 3 + 4
        if go_fp_dim > 0:
            feature_dim += go_fp_dim * 3

        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.binding_encoder = BindingEncoder(protein_dim, drug_dim, hidden=hidden_dim)
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.last_forward_stats = {}

    def _build_features(
        self,
        qry_protein: torch.Tensor,
        qry_drug: torch.Tensor,
        neighbor_protein: torch.Tensor,
        neighbor_drug: torch.Tensor,
        neighbor_affinity: torch.Tensor,
        neighbor_ppr: torch.Tensor,
        neighbor_trust: torch.Tensor,
        qry_go_fp: torch.Tensor | None = None,
        neighbor_go_fp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N_qry, K, _ = neighbor_protein.shape
        qry_protein_exp = qry_protein.unsqueeze(1).expand(-1, K, -1)
        qry_drug_exp = qry_drug.unsqueeze(1).expand(-1, K, -1)
        protein_delta = qry_protein_exp - neighbor_protein
        drug_delta = qry_drug_exp - neighbor_drug
        drug_sim = F.cosine_similarity(qry_drug_exp, neighbor_drug, dim=-1).unsqueeze(-1)

        parts = [
            qry_protein_exp,
            neighbor_protein,
            protein_delta,
            qry_drug_exp,
            neighbor_drug,
            drug_delta,
            neighbor_affinity.unsqueeze(-1),
            neighbor_ppr.unsqueeze(-1),
            neighbor_trust.unsqueeze(-1),
            drug_sim,
        ]
        if qry_go_fp is not None and neighbor_go_fp is not None and self.go_fp_dim > 0:
            qry_go_exp = qry_go_fp.unsqueeze(1).expand(-1, K, -1)
            go_delta = qry_go_exp - neighbor_go_fp
            parts.extend([qry_go_exp, neighbor_go_fp, go_delta])

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        neighbor_protein: torch.Tensor,   # [N_qry, K, protein_dim]
        neighbor_drug: torch.Tensor,      # [N_qry, K, drug_dim] (kept for API symmetry)
        neighbor_affinity: torch.Tensor,  # [N_qry, K]
        neighbor_ppr: torch.Tensor,       # [N_qry, K]
        neighbor_trust: torch.Tensor,     # [N_qry, K]
        neighbor_mask: torch.Tensor,      # [N_qry, K]
        qry_protein: torch.Tensor,        # [N_qry, protein_dim]
        qry_drug: torch.Tensor,           # [N_qry, drug_dim]
        qry_go_fp: torch.Tensor | None = None,       # [N_qry, go_dim]
        neighbor_go_fp: torch.Tensor | None = None,  # [N_qry, K, go_dim]
        global_mean_affinity: float = 6.5,
    ):
        features = self._build_features(
            qry_protein,
            qry_drug,
            neighbor_protein,
            neighbor_drug,
            neighbor_affinity,
            neighbor_ppr,
            neighbor_trust,
            qry_go_fp=qry_go_fp,
            neighbor_go_fp=neighbor_go_fp,
        )

        weight_logits = self.weight_net(features).squeeze(-1)
        weight_logits = weight_logits.masked_fill(~neighbor_mask, -1e9)
        weights = torch.softmax(weight_logits, dim=1)
        weights = weights * neighbor_mask.float()
        norm = weights.sum(dim=1, keepdim=True)
        weights = torch.where(norm > 0, weights / norm.clamp(min=1e-8), torch.zeros_like(weights))

        transfer = (weights * neighbor_affinity).sum(dim=1)
        delta = self.delta_net(features).squeeze(-1)
        correction = (weights * delta).sum(dim=1)
        direct_prior = self.binding_encoder(qry_protein, qry_drug) + global_mean_affinity

        has_neighbors = neighbor_mask.any(dim=1)
        mu = torch.where(has_neighbors, transfer + correction, direct_prior)

        base_sigma = F.softplus(self.log_sigma).expand_as(mu)
        sigma = base_sigma + (~has_neighbors).float() * 0.15 + 1e-4

        self.last_forward_stats = {
            "mu_std": float(mu.detach().std(unbiased=False).item()) if mu.numel() > 1 else 0.0,
            "binding_prior_std": float(direct_prior.detach().std(unbiased=False).item()) if direct_prior.numel() > 1 else 0.0,
            "log_ppr_alpha": 0.0,
            "centroid_alpha": 0.0,
            "density": float(neighbor_mask.float().sum(dim=1).mean().item()),
        }
        return mu, sigma
