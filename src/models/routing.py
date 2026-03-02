import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertScorer(nn.Module):
    def __init__(self, protein_dim, drug_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.p_proj = nn.Linear(protein_dim, hidden_dim)
        self.d_proj = nn.Linear(drug_dim, hidden_dim)
        self.interaction = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_p, z_d):
        p = self.p_proj(z_p)
        d = self.d_proj(z_d)
        if p.size(0) == 1 and d.size(0) > 1:
            p = p.expand(d.size(0), -1)
        joint_rep = self.interaction(p, d)
        return self.mlp(joint_rep).squeeze(-1)


class MultiplexMoEGate(nn.Module):
    """The Meta-Gate. Looks at the protein and trust diagnostics to pick experts."""

    def __init__(self, protein_dim, num_experts, hidden_dim=128, top_k=2):
        super().__init__()
        self.top_k = top_k
        # Input: protein + form_trust(3) + role_trust(3) + cross-floor agreement(1)
        self.router = nn.Sequential(
            nn.Linear(protein_dim + 7, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, z_refined, trust_form, trust_role, cross_floor_jaccard):
        gate_input = torch.cat(
            [
                z_refined,
                trust_form.view(1, -1),
                trust_role.view(1, -1),
                cross_floor_jaccard.view(1, 1),
            ],
            dim=-1,
        )
        logits = self.router(gate_input)

        k = min(self.top_k, logits.size(-1))
        if k < logits.size(-1):
            top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            sparse_logits.scatter_(dim=-1, index=top_idx, src=top_vals)
            logits = sparse_logits

        return F.softmax(logits, dim=-1)


class MultiplexRoutingHead(nn.Module):
    def __init__(self, protein_dim, drug_dim, num_experts=4, top_k=2):
        super().__init__()
        self.gate = MultiplexMoEGate(protein_dim, num_experts, top_k=top_k)
        self.experts = nn.ModuleList([ExpertScorer(protein_dim, drug_dim) for _ in range(num_experts)])

    def _calculate_trust_vector(self, footprints, floor_attn):
        device = footprints.device
        if footprints.size(0) == 0:
            return torch.zeros(3, device=device)

        neighbor_count = torch.log1p(torch.tensor(float(footprints.size(0)), device=device))

        if floor_attn.numel() > 1:
            p = floor_attn.clamp_min(1e-12)
            entropy = -(p * p.log()).sum() / torch.log(torch.tensor(float(p.numel()), device=device))
        else:
            entropy = torch.tensor(0.0, device=device)

        med = footprints.median(dim=0).values
        mad = (footprints - med).abs().median(dim=0).values.mean()
        robust_dispersion = 1.0 / (1.0 + mad)

        return torch.stack([neighbor_count, 1.0 - entropy, robust_dispersion])

    def forward(self, z_refined, form_footprints, role_footprints, query_drug_features, floor_stats, cross_floor_jaccard):
        if z_refined.dim() == 1:
            z_refined = z_refined.unsqueeze(0)

        trust_form = self._calculate_trust_vector(form_footprints, floor_stats.get("form_attn", torch.empty(0, device=z_refined.device)))
        trust_role = self._calculate_trust_vector(role_footprints, floor_stats.get("role_attn", torch.empty(0, device=z_refined.device)))

        gate_probs = self.gate(z_refined, trust_form, trust_role, cross_floor_jaccard)

        expert_predictions = [expert(z_refined, query_drug_features) for expert in self.experts]
        expert_tensor = torch.stack(expert_predictions, dim=1)
        final_scores = torch.sum(expert_tensor * gate_probs, dim=-1)
        return final_scores, gate_probs, expert_tensor
