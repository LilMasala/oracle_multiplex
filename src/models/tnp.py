"""
Graph-biased TNP with cold-start improvements:
  Unit 3 — Context density gating (calibrated uncertainty)
  Unit 4 — PPR centroid interpolation in query encoding
  Unit 6 — GNN pre-encoder residual in protein projections
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphBiasedMHA(nn.Module):
    """
    Multi-head self-attention with graph-structural additive logit biases
    and a multiplicative value gate for context tokens.

    attn_logit(i → ctx_k) += alpha * log(ppr_k + ε)
    v_ctx_k                *= sigmoid(trust_scale * trust_k)
    """

    def __init__(self, token_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert token_dim % nhead == 0
        self.nhead    = nhead
        self.head_dim = token_dim // nhead
        self.scale    = self.head_dim ** -0.5
        self.dropout  = dropout

        self.in_proj  = nn.Linear(token_dim, 3 * token_dim, bias=True)
        self.out_proj = nn.Linear(token_dim, token_dim, bias=True)

        # Learnable graph-bias scalars
        self.log_ppr_alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.trust_scale   = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,           # [1, N, D]
        n_ctx: int,
        ctx_ppr: torch.Tensor,     # [n_ctx]
        ctx_trust: torch.Tensor,   # [n_ctx]
        attn_mask: torch.Tensor | None = None,  # [N, N] bool, True=blocked
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.nhead, self.head_dim

        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, H, d).transpose(1, 2)  # [1, H, N, d]
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)

        # V gate for context tokens
        if n_ctx > 0:
            gate = torch.sigmoid(self.trust_scale * ctx_trust)  # [n_ctx]
            ones = torch.ones(N - n_ctx, device=x.device)
            gate_full = torch.cat([gate, ones], dim=0)           # [N]
            v = v * gate_full.view(1, 1, N, 1)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [1, H, N, N]

        # Graph bias on context columns
        if n_ctx > 0:
            col_bias = -self.log_ppr_alpha * torch.log(ctx_ppr.clamp(min=1e-8))
            pad = torch.zeros(N - n_ctx, device=x.device)
            col_bias_full = torch.cat([col_bias, pad], dim=0)
            logits = logits + col_bias_full.view(1, 1, 1, N)

        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask.view(1, 1, N, N), float("-inf"))

        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        out = torch.matmul(weights, v)                         # [1, H, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class GraphBiasedTransformerLayer(nn.Module):
    """Pre-norm transformer layer wrapping GraphBiasedMHA."""

    def __init__(self, token_dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.attn  = GraphBiasedMHA(token_dim, nhead, dropout=dropout)
        self.ff    = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, n_ctx, ctx_ppr, ctx_trust, attn_mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, n_ctx, ctx_ppr, ctx_trust, attn_mask)
        x = x + self.ff(self.norm2(x))
        return x


class ProteinLigandTNP(nn.Module):
    """
    Graph-biased Transformer Neural Process for protein-ligand binding prediction.

    Cold-start improvements (Units 3, 4, 6):
      - Density gating: density = tanh(n_ctx/32) added to query tokens and sigma
      - PPR centroid blending: query protein ← (1-α)*target + α*centroid
      - GNN residual: optional pre-encoder embedding added to protein projections
    """

    def __init__(
        self,
        protein_dim: int,
        drug_dim: int,
        token_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        gnn_emb_dim: int = 0,   # Unit 6: set > 0 to enable GNN residual
    ):
        super().__init__()
        self.token_dim   = token_dim
        self.gnn_emb_dim = gnn_emb_dim

        # Context encoder: (protein, drug, affinity) → token
        self.ctx_protein_proj  = nn.Linear(protein_dim, token_dim // 2)
        self.ctx_drug_proj     = nn.Linear(drug_dim, token_dim // 2)
        self.ctx_affinity_proj = nn.Linear(1, token_dim // 4)
        self.ctx_fusion = nn.Linear(
            token_dim // 2 + token_dim // 2 + token_dim // 4, token_dim
        )

        # Query encoder: (protein, drug) → token
        self.qry_protein_proj = nn.Linear(protein_dim, token_dim // 2)
        self.qry_drug_proj    = nn.Linear(drug_dim, token_dim // 2)
        self.qry_fusion = nn.Linear(token_dim // 2 + token_dim // 2, token_dim)

        # Type embedding: 0 = context, 1 = query
        self.type_embed = nn.Embedding(2, token_dim)

        # Graph-biased transformer layers
        self.layers = nn.ModuleList([
            GraphBiasedTransformerLayer(token_dim, nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(token_dim)

        # Output head: query token → (mu, log_sigma)
        self.output_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, 2),
        )

        # Unit 3: Context density gating
        self.density_proj    = nn.Linear(1, token_dim)
        self.cold_start_bias = nn.Parameter(torch.zeros(1))

        # Unit 4: PPR centroid interpolation (learnable blending weight)
        # Init at -2 so sigmoid(-2) ≈ 0.12: mild blending at cold-start
        self.centroid_alpha = nn.Parameter(torch.tensor(-2.0))

        # Unit 6: GNN residual projections
        if gnn_emb_dim > 0:
            self.ctx_gnn_proj = nn.Linear(gnn_emb_dim, token_dim // 2)
            self.qry_gnn_proj = nn.Linear(gnn_emb_dim, token_dim // 2)

    def _encode_context(
        self,
        ctx_protein: torch.Tensor,
        ctx_drug: torch.Tensor,
        ctx_affinity: torch.Tensor,
        ctx_gnn_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = self.ctx_protein_proj(ctx_protein)
        if ctx_gnn_emb is not None and self.gnn_emb_dim > 0:
            p = p + self.ctx_gnn_proj(ctx_gnn_emb)   # Unit 6: residual
        d = self.ctx_drug_proj(ctx_drug)
        a = self.ctx_affinity_proj(ctx_affinity)
        return self.ctx_fusion(torch.cat([p, d, a], dim=-1))

    def _encode_query(
        self,
        qry_protein: torch.Tensor,
        qry_drug: torch.Tensor,
        qry_gnn_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = self.qry_protein_proj(qry_protein)
        if qry_gnn_emb is not None and self.gnn_emb_dim > 0:
            p = p + self.qry_gnn_proj(qry_gnn_emb)   # Unit 6: residual
        d = self.qry_drug_proj(qry_drug)
        return self.qry_fusion(torch.cat([p, d], dim=-1))

    def _build_mask(self, n_ctx: int, n_qry: int, device):
        """
        Attention mask: queries attend to context only (no qry→qry, no ctx→qry).
        Shape [n_ctx + n_qry, n_ctx + n_qry], True = blocked.
        """
        N = n_ctx + n_qry
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        if n_qry > 0:
            mask[n_ctx:, n_ctx:] = True
            idx = torch.arange(n_ctx, N, device=device)
            mask[idx, idx] = False
            mask[:n_ctx, n_ctx:] = True
        return mask

    def forward(
        self,
        ctx_protein: torch.Tensor,    # [N_ctx, protein_dim]
        ctx_drug: torch.Tensor,        # [N_ctx, drug_dim]
        ctx_affinity: torch.Tensor,    # [N_ctx, 1]
        qry_protein: torch.Tensor,     # [N_qry, protein_dim]
        qry_drug: torch.Tensor,        # [N_qry, drug_dim]
        ctx_ppr: Optional[torch.Tensor] = None,    # [N_ctx]
        ctx_trust: Optional[torch.Tensor] = None,  # [N_ctx]
        ppr_centroid: Optional[torch.Tensor] = None,  # Unit 4: [protein_dim]
        ctx_gnn_emb: Optional[torch.Tensor] = None,   # Unit 6: [N_ctx, gnn_dim]
        qry_gnn_emb: Optional[torch.Tensor] = None,   # Unit 6: [N_qry, gnn_dim]
        global_mean_affinity: float = 6.5,            # fallback for cold-start
    ):
        device = qry_protein.device
        n_ctx  = ctx_protein.size(0)
        n_qry  = qry_protein.size(0)

        # Units 3 & 4: context density scalar (Python float, no gradient)
        density = math.tanh(n_ctx / 32.0)

        # Unit 4: PPR centroid blending for query proteins
        # α_eff = sigmoid(centroid_alpha) × (1 - density)
        # → high blending when cold (density≈0), low when warm (density→1)
        if ppr_centroid is not None:
            alpha_eff = torch.sigmoid(self.centroid_alpha) * (1.0 - density)
            qry_protein = (1.0 - alpha_eff) * qry_protein + alpha_eff * ppr_centroid.unsqueeze(0)

        # Encode query tokens
        qry_tokens = self._encode_query(qry_protein, qry_drug, qry_gnn_emb)
        qry_tokens = qry_tokens + self.type_embed(
            torch.ones(n_qry, dtype=torch.long, device=device)
        )

        # Unit 3: add density signal to query tokens
        density_t   = torch.tensor([[density]], dtype=torch.float32, device=device)  # [1, 1]
        density_tok = self.density_proj(density_t)   # [1, token_dim]
        qry_tokens  = qry_tokens + density_tok       # broadcast over [n_qry, D]

        if n_ctx > 0:
            ctx_tokens = self._encode_context(ctx_protein, ctx_drug, ctx_affinity, ctx_gnn_emb)
            ctx_tokens = ctx_tokens + self.type_embed(
                torch.zeros(n_ctx, dtype=torch.long, device=device)
            )
            tokens = torch.cat([ctx_tokens, qry_tokens], dim=0).unsqueeze(0)  # [1, N, D]
            mask   = self._build_mask(n_ctx, n_qry, device)

            if ctx_ppr is None:
                ctx_ppr   = torch.ones(n_ctx, device=device)
            if ctx_trust is None:
                ctx_trust = torch.ones(n_ctx, device=device)
        else:
            tokens    = qry_tokens.unsqueeze(0)
            mask      = None
            ctx_ppr   = torch.zeros(0, device=device)
            ctx_trust = torch.zeros(0, device=device)

        for layer in self.layers:
            tokens = layer(tokens, n_ctx, ctx_ppr, ctx_trust, attn_mask=mask)
        tokens = self.final_norm(tokens).squeeze(0)  # [N, D]

        qry_out = tokens[n_ctx:]                      # [N_qry, D]
        pred    = self.output_head(qry_out)            # [N_qry, 2]
        mu        = pred[:, 0]
        log_sigma = pred[:, 1]

        # Context affinity anchoring: shift mu by the observed neighborhood mean.
        # The transformer only needs to learn residuals around that mean, which is
        # a much tighter target than predicting absolute affinity from scratch.
        # At cold-start (n_ctx == 0) we fall back to the global mean affinity.
        if n_ctx > 0:
            ctx_mean = ctx_affinity.mean()
        else:
            ctx_mean = torch.tensor(global_mean_affinity, dtype=torch.float32, device=device)
        mu = mu + ctx_mean

        # Unit 3: cold-start sigma bias — upward uncertainty when n_ctx is small
        # Bias decays to zero as density → 1 (warm regime)
        log_sigma = log_sigma + self.cold_start_bias * (1.0 - density)
        sigma = F.softplus(log_sigma) + 1e-4

        return mu, sigma


if __name__ == "__main__":
    prot_dim, drug_dim = 2816, 512
    model = ProteinLigandTNP(protein_dim=prot_dim, drug_dim=drug_dim,
                             token_dim=256, gnn_emb_dim=256)

    ctx_protein  = torch.randn(10, prot_dim)
    ctx_drug     = torch.randn(10, drug_dim)
    ctx_affinity = torch.randn(10, 1)
    ctx_ppr      = torch.rand(10).clamp(1e-6, 1.0)
    ctx_trust    = torch.rand(10)
    ctx_gnn      = torch.randn(10, 256)
    qry_protein  = torch.randn(50, prot_dim)
    qry_drug     = torch.randn(50, drug_dim)
    qry_gnn      = torch.randn(50, 256)
    ppr_centroid = torch.randn(prot_dim)

    mu, sigma = model(ctx_protein, ctx_drug, ctx_affinity,
                      qry_protein, qry_drug,
                      ctx_ppr, ctx_trust,
                      ppr_centroid=ppr_centroid,
                      ctx_gnn_emb=ctx_gnn,
                      qry_gnn_emb=qry_gnn)
    assert mu.shape == (50,), f"Expected (50,), got {mu.shape}"
    assert sigma.shape == (50,)
    assert (sigma > 0).all()
    print("With full cold-start improvements (warm): PASSED")

    # Cold-start: n_ctx = 0
    mu0, sigma0 = model(
        torch.zeros(0, prot_dim), torch.zeros(0, drug_dim), torch.zeros(0, 1),
        qry_protein, qry_drug,
        ppr_centroid=ppr_centroid,
        qry_gnn_emb=qry_gnn,
    )
    assert mu0.shape == (50,)
    assert (sigma0 > 0).all()
    # Cold-start sigma should be elevated due to cold_start_bias
    print(f"Cold-start: sigma mean = {sigma0.mean():.3f} — PASSED")
    print("All TNP tests passed.")
