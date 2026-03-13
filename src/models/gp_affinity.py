"""
GP-inspired affinity model with drug-conditioned deep kernel learning.

Architecture:
  DrugConditionalEncoder  f(protein, drug) → embedding  (the deep kernel)
  CrossAttentionLayer     query protein attends to context proteins, drug-conditionally
  GPAffinityModel         GP transfer + correction head; prior supplied by caller

Inference cascade:
  Level 1: D's exact binding profile {(protX, affinity)}
  Level 2: D' analog profiles {(protX, affinity × drug_sim)}, D' ~ D
  Level 3: no context → caller-supplied prior (frozen BindingEncoder output + global mean)

The cross-attention approximates the GP posterior mean:
  μ(A) = k_D(A, X) [K_D(X,X) + σ²I]⁻¹ y
where k_D is the drug-conditional deep kernel learned by DrugConditionalEncoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugConditionalEncoder(nn.Module):
    """
    Projects protein into the subspace drug D responds to — the deep kernel.

    Two proteins that differ only in regions irrelevant to D's binding pocket
    map close together; proteins that differ along D-relevant dimensions map
    far apart. This is what prevents oversmoothing in the GP transfer.
    """

    def __init__(self, protein_dim: int, drug_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(protein_dim + drug_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, protein: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        """[..., protein_dim], [..., drug_dim] → [..., out_dim]"""
        return self.norm(self.net(torch.cat([protein, drug], dim=-1)))


class CrossAttentionLayer(nn.Module):
    """
    Pre-norm cross-attention: query [B, 1, dim] attends to context [B, K, dim].

    Used to build the attended representation for the correction head —
    multi-head so different heads can attend to different aspects of how
    D's binding profile varies across protein space.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        query: torch.Tensor,    # [B, 1, dim]
        context: torch.Tensor,  # [B, K, dim]
        mask: torch.Tensor,     # [B, K] bool, True = valid slot
    ) -> torch.Tensor:          # [B, 1, dim]
        B, _, D = query.shape
        H, d = self.n_heads, self.head_dim

        q = self.q_proj(self.norm_q(query)).view(B, 1, H, d).transpose(1, 2)
        k = self.k_proj(self.norm_kv(context)).view(B, -1, H, d).transpose(1, 2)
        v = self.v_proj(self.norm_kv(context)).view(B, -1, H, d).transpose(1, 2)

        # Additive bias: -inf blocks invalid context positions
        attn_bias = torch.zeros(B, 1, 1, mask.size(1), device=query.device, dtype=query.dtype)
        attn_bias = attn_bias.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, H, 1, d]

        out = out.transpose(1, 2).reshape(B, 1, D)
        query = query + self.out_proj(out)
        query = query + self.ff(self.norm_ff(query))
        return query


class GPAffinityModel(nn.Module):
    """
    Binding affinity predictor via drug-conditioned GP approximation.

    Forward pass:
      1. Encode query and context proteins drug-conditionally (deep kernel).
      2. GP transfer: single-head attention over scalar affinities →
         approximates GP posterior mean μ(A) = k_D(A,X)[K_D(X,X)+σ²I]⁻¹y.
      3. Correction: multi-layer cross-attention builds an attended
         representation; correction_head reads (query_emb, attended) →
         nonlinear residual for where A diverges from context in ways D cares
         about.
      4. Fallback: when no context, uses caller-supplied prior
         (frozen_binding_encoder(qry_protein, qry_drug) + global_mean_affinity).
    """

    def __init__(
        self,
        protein_dim: int,
        drug_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_dim = out_dim

        # Deep kernel
        self.encoder = DrugConditionalEncoder(protein_dim, drug_dim, hidden_dim, out_dim)

        # Multi-layer cross-attention for correction representation
        self.layers = nn.ModuleList([
            CrossAttentionLayer(out_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(out_dim)

        # Output: (query_emb ‖ attended_emb) → (mu_residual, log_sigma)
        self.output_head = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        # Zero-init so the model starts as a pure GP transfer, no correction
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

        # Context quality gate: how much to trust GP vs. prior.
        # Reads the attended representation → scalar alpha in (0,1).
        # Bias init at -2 → sigmoid(-2) ≈ 0.12, so model starts near pure prior
        # and learns to open the gate as context proves reliable.
        self.ctx_gate = nn.Linear(out_dim, 1)
        nn.init.zeros_(self.ctx_gate.weight)
        nn.init.constant_(self.ctx_gate.bias, -2.0)

        self.last_forward_stats = {}

    def forward(
        self,
        qry_protein: torch.Tensor,     # [n_qry, protein_dim]
        qry_drug: torch.Tensor,        # [n_qry, drug_dim]
        ctx_proteins: torch.Tensor,    # [n_qry, K, protein_dim]
        ctx_affinities: torch.Tensor,  # [n_qry, K]  (drug_sim-scaled at level 2)
        ctx_mask: torch.Tensor,        # [n_qry, K] bool, True = valid
        prior: torch.Tensor,           # [n_qry] precomputed by caller as frozen_binding_encoder(qry_protein, qry_drug) + global_mean_affinity
    ):
        n_qry, K = ctx_proteins.shape[:2]
        device = qry_protein.device
        has_ctx = ctx_mask.any(dim=1)  # [n_qry]

        if prior is None:
            prior = torch.zeros(n_qry, device=device)

        # Drug-conditional query encoding (deep kernel)
        Q = self.encoder(qry_protein, qry_drug)  # [n_qry, out_dim]

        if K > 0:
            # Drug-conditional context encoding
            # Each context protein is encoded through the lens of its query drug
            qry_drug_exp = qry_drug.unsqueeze(1).expand(-1, K, -1).reshape(n_qry * K, -1)
            K_emb = self.encoder(
                ctx_proteins.reshape(n_qry * K, -1),
                qry_drug_exp,
            ).view(n_qry, K, self.out_dim)  # [n_qry, K, out_dim]

            # GP transfer: single-head attention weights over scalar affinities
            # → approximates k_D(A,X)[K_D(X,X)+σ²I]⁻¹y
            scale = math.sqrt(self.out_dim)
            transfer_logits = (Q.unsqueeze(1) @ K_emb.transpose(-1, -2)).squeeze(1) / scale
            # Safely handle fully-masked rows: replace with zeros before softmax
            safe_logits = torch.where(
                has_ctx.unsqueeze(1),
                transfer_logits.masked_fill(~ctx_mask, float("-inf")),
                torch.zeros_like(transfer_logits),
            )
            transfer_weights = torch.softmax(safe_logits, dim=-1) * ctx_mask.float()
            transfer = (transfer_weights * ctx_affinities).sum(1)  # [n_qry]

            # Multi-layer cross-attention for correction representation
            x = Q.unsqueeze(1)  # [n_qry, 1, out_dim]
            for layer in self.layers:
                x = layer(x, K_emb, ctx_mask)
            attended = self.final_norm(x).squeeze(1)  # [n_qry, out_dim]
        else:
            transfer = torch.zeros(n_qry, device=device)
            attended = torch.zeros(n_qry, self.out_dim, device=device)

        # Correction head: reads query embedding + attended context representation
        pred = self.output_head(torch.cat([Q, attended], dim=-1))  # [n_qry, 2]
        mu_residual = pred[:, 0]
        log_sigma = pred[:, 1]

        # mu = prior + GP residual transfer + correction when context available; prior otherwise.
        # Context stores residuals (label - prior), so we must add prior back to get absolute scale.
        # alpha gates how much to trust the GP vs. the prior — learned from attended representation.
        # When context is noisy (e.g. level-3 neighborhood fallback for unrelated drugs),
        # the gate learns to stay near 0, keeping predictions anchored to the prior.
        alpha = torch.sigmoid(self.ctx_gate(attended)).squeeze(-1)  # [n_qry]
        mu_ctx = prior + alpha * (transfer + mu_residual)
        mu = torch.where(has_ctx, mu_ctx, prior)

        sigma = F.softplus(log_sigma) + 1e-4

        self.last_forward_stats = {
            "mu_std": float(mu.detach().std(unbiased=False).item()) if mu.numel() > 1 else 0.0,
            "log_ppr_alpha": 0.0,
            "centroid_alpha": 0.0,
            "density": float(ctx_mask.float().sum(dim=1).mean().item()),
        }

        return mu, sigma


if __name__ == "__main__":
    prot_dim, drug_dim = 2816, 512
    model = GPAffinityModel(prot_dim, drug_dim, hidden_dim=256, out_dim=128, n_heads=4, n_layers=2)

    n_qry, K = 50, 16
    qry_p = torch.randn(n_qry, prot_dim)
    qry_d = torch.randn(n_qry, drug_dim)
    ctx_p = torch.randn(n_qry, K, prot_dim)
    ctx_a = torch.randn(n_qry, K)
    ctx_m = torch.ones(n_qry, K, dtype=torch.bool)
    prior = torch.full((n_qry,), 6.5)

    mu, sigma = model(qry_p, qry_d, ctx_p, ctx_a, ctx_m, prior=prior)
    assert mu.shape == (n_qry,)
    assert sigma.shape == (n_qry,)
    assert (sigma > 0).all()
    print(f"Warm: mu={mu.mean():.3f} sigma={sigma.mean():.3f} — PASSED")

    # Cold-start: all mask False
    cold_mask = torch.zeros(n_qry, K, dtype=torch.bool)
    mu0, sigma0 = model(qry_p, qry_d, ctx_p, ctx_a, cold_mask, prior=prior)
    assert mu0.shape == (n_qry,)
    assert (sigma0 > 0).all()
    print(f"Cold: mu={mu0.mean():.3f} sigma={sigma0.mean():.3f} — PASSED")

    # K=0 edge case
    mu1, sigma1 = model(
        qry_p, qry_d,
        torch.zeros(n_qry, 0, prot_dim),
        torch.zeros(n_qry, 0),
        torch.zeros(n_qry, 0, dtype=torch.bool),
        prior=prior,
    )
    assert mu1.shape == (n_qry,)
    print(f"K=0: mu={mu1.mean():.3f} — PASSED")
    print("All GPAffinityModel tests passed.")
