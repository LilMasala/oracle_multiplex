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
    """
    Trust-aware gate.
    Input = [protein_raw_features, v_prior, trust_vector].
    """

    def __init__(self, protein_dim, drug_dim, trust_dim, num_experts, hidden_dim=128, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Sequential(
            nn.Linear(protein_dim + drug_dim + trust_dim, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, protein_raw, v_prior, trust_vector):
        if protein_raw.dim() == 1:
            protein_raw = protein_raw.unsqueeze(0)
        if v_prior.dim() == 1:
            v_prior = v_prior.unsqueeze(0)
        if trust_vector.dim() == 1:
            trust_vector = trust_vector.unsqueeze(0)

        gate_input = torch.cat([protein_raw, v_prior, trust_vector], dim=-1)
        logits = self.router(gate_input)

        k = min(self.top_k, logits.size(-1))
        if k < logits.size(-1):
            top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
            sparse_logits = torch.full_like(logits, float("-inf"))
            sparse_logits.scatter_(dim=-1, index=top_idx, src=top_vals)
            logits = sparse_logits

        return F.softmax(logits, dim=-1)


class MultiplexRoutingHead(nn.Module):
    def __init__(self, protein_dim, drug_dim, num_experts=4, top_k=2, trust_dim=3):
        super().__init__()
        self.gate = MultiplexMoEGate(
            protein_dim=protein_dim,
            drug_dim=drug_dim,
            trust_dim=trust_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.experts = nn.ModuleList([ExpertScorer(protein_dim, drug_dim) for _ in range(num_experts)])

    def forward(self, z_refined, protein_raw_features, v_prior, query_drug_features, trust_vector):
        if z_refined.dim() == 1:
            z_refined = z_refined.unsqueeze(0)

        gate_probs = self.gate(protein_raw_features, v_prior, trust_vector)
        expert_predictions = [expert(z_refined, query_drug_features) for expert in self.experts]
        expert_tensor = torch.stack(expert_predictions, dim=1)
        final_scores = torch.sum(expert_tensor * gate_probs, dim=-1)
        return final_scores, gate_probs, expert_tensor
