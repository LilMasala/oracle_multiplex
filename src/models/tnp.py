"""
Graph-biased TNP with cold-start improvements:
  Unit 3 — Context density gating (calibrated uncertainty)
  Unit 4 — PPR centroid interpolation in query encoding
  Unit 6 — GNN pre-encoder residual in protein projections
  Unit 7 — Direct binding encoder (separate learning path for protein-drug features)
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
        self.dropout  = dropout

        self.in_proj  = nn.Linear(token_dim, 3 * token_dim, bias=True)
        self.out_proj = nn.Linear(token_dim, token_dim, bias=True)

        # Learnable graph-bias scalars
        self.log_ppr_alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.trust_scale   = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,           # [B, N, D]  B=1 shared mode, B=N_qry per-query mode
        n_ctx: int,
        ctx_ppr: torch.Tensor,     # [n_ctx] shared  OR  [B, n_ctx] per-query
        ctx_trust: torch.Tensor,   # [n_ctx] shared  OR  [B, n_ctx] per-query
        attn_mask: torch.Tensor | None = None,  # [N, N] bool, True=blocked
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.nhead, self.head_dim
        per_query = ctx_ppr.dim() == 2   # True when ctx_ppr is [B, K]

        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, H, d).transpose(1, 2)  # [B, H, N, d]
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)

        # V gate for context tokens
        if n_ctx > 0:
            gate = torch.sigmoid(self.trust_scale * ctx_trust)  # [K] or [B, K]
            if per_query:
                ones = torch.ones(B, N - n_ctx, device=x.device)
                gate_full = torch.cat([gate, ones], dim=1)       # [B, N]
                v = v * gate_full.view(B, 1, N, 1)
            else:
                ones = torch.ones(N - n_ctx, device=x.device)
                gate_full = torch.cat([gate, ones], dim=0)       # [N]
                v = v * gate_full.view(1, 1, N, 1)

        # Build combined additive bias: PPR column logit bonus + block mask.
        # [B, 1, N, N] broadcasts over heads; pure FlashAttention when bias=None.
        bias = None
        if n_ctx > 0 or attn_mask is not None:
            bias = torch.zeros(B, 1, N, N, device=x.device, dtype=x.dtype)
            if n_ctx > 0:
                col_bias = -self.log_ppr_alpha * torch.log(ctx_ppr.clamp(min=1e-8))
                if per_query:
                    pad = torch.zeros(B, N - n_ctx, device=x.device)
                    col_bias_full = torch.cat([col_bias, pad], dim=1)    # [B, N]
                    bias = bias + col_bias_full.view(B, 1, 1, N)
                else:
                    pad = torch.zeros(N - n_ctx, device=x.device)
                    col_bias_full = torch.cat([col_bias, pad], dim=0)    # [N]
                    bias = bias + col_bias_full.view(1, 1, 1, N)
            if attn_mask is not None:
                bias = bias.masked_fill(attn_mask.view(1, 1, N, N), float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, H, N, d]

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


class BindingEncoder(nn.Module):
    """
    Unit 7: Direct protein-drug binding encoder.

    Learns what raw protein and drug features predict affinity, independent of
    the transformer's context-aggregation path.  Trained on every revealed
    (protein, drug, affinity) triple via the same loss as the TNP — but its
    gradient only flows into this MLP, not back through the transformer (we
    detach its output before injecting it as a query-token residual).

    At cold-start its output contributes fully to mu; as context grows the
    contribution is density-gated toward zero so the transformer's
    neighbor-transfer signal takes over.
    """

    def __init__(self, protein_dim: int, drug_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(protein_dim + drug_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Zero-init last layer so it starts as a no-op
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, protein: torch.Tensor, drug: torch.Tensor) -> torch.Tensor:
        """Returns [N] affinity residual."""
        return self.net(torch.cat([protein, drug], dim=-1)).squeeze(-1)


class BindingOnlyAffinityModel(nn.Module):
    """Direct protein-drug baseline with a global affinity anchor and scalar sigma."""

    def __init__(self, protein_dim: int, drug_dim: int, hidden: int = 512):
        super().__init__()
        self.binding_encoder = BindingEncoder(protein_dim, drug_dim, hidden=hidden)
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.last_forward_stats = {}

    def forward(
        self,
        qry_protein: torch.Tensor,
        qry_drug: torch.Tensor,
        global_mean_affinity: float = 6.5,
    ):
        binding_prior = self.binding_encoder(qry_protein, qry_drug)
        mu = binding_prior + global_mean_affinity
        sigma = F.softplus(self.log_sigma).expand_as(mu) + 1e-4
        self.last_forward_stats = {
            "mu_std": float(mu.detach().std(unbiased=False).item()) if mu.numel() > 1 else 0.0,
            "binding_prior_std": float(binding_prior.detach().std(unbiased=False).item()) if binding_prior.numel() > 1 else 0.0,
        }
        return mu, sigma


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
        go_fp_dim:  int = 0,   # Unit 9: set > 0 to enable GO fingerprint residual
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

        # Unit 9: GO functional fingerprint projections (anc2vec mean-pool)
        self.go_fp_dim = go_fp_dim
        if go_fp_dim > 0:
            self.ctx_go_fp_proj = nn.Linear(go_fp_dim, token_dim // 2)
            self.qry_go_fp_proj = nn.Linear(go_fp_dim, token_dim // 2)

        # Unit 7: Direct binding encoder + projection into token space
        self.binding_encoder = BindingEncoder(protein_dim, drug_dim)
        self.prior_proj = nn.Linear(1, token_dim)
        self.last_forward_stats = {}

    def _encode_context(
        self,
        ctx_protein: torch.Tensor,
        ctx_drug: torch.Tensor,
        ctx_affinity: torch.Tensor,
        ctx_gnn_emb: Optional[torch.Tensor] = None,
        ctx_go_fp:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = self.ctx_protein_proj(ctx_protein)
        if ctx_gnn_emb is not None and self.gnn_emb_dim > 0:
            p = p + self.ctx_gnn_proj(ctx_gnn_emb)   # Unit 6: residual
        if ctx_go_fp is not None and self.go_fp_dim > 0:
            p = p + self.ctx_go_fp_proj(ctx_go_fp)   # Unit 9: GO fingerprint residual
        d = self.ctx_drug_proj(ctx_drug)
        a = self.ctx_affinity_proj(ctx_affinity)
        return self.ctx_fusion(torch.cat([p, d, a], dim=-1))

    def _encode_query(
        self,
        qry_protein: torch.Tensor,
        qry_drug: torch.Tensor,
        qry_gnn_emb: Optional[torch.Tensor] = None,
        qry_go_fp:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p = self.qry_protein_proj(qry_protein)
        if qry_gnn_emb is not None and self.gnn_emb_dim > 0:
            p = p + self.qry_gnn_proj(qry_gnn_emb)   # Unit 6: residual
        if qry_go_fp is not None and self.go_fp_dim > 0:
            p = p + self.qry_go_fp_proj(qry_go_fp)   # Unit 9: GO fingerprint residual
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
        ctx_go_fp:  Optional[torch.Tensor] = None,    # Unit 9: [N_ctx, go_fp_dim]
        qry_go_fp:  Optional[torch.Tensor] = None,    # Unit 9: [N_qry, go_fp_dim]
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

        # Unit 7: direct binding prior — trained end-to-end via mu, but detached
        # before injection into query tokens so the transformer's context-reading
        # gradient is not polluted by the binding encoder's learning signal.
        binding_prior = self.binding_encoder(qry_protein, qry_drug)  # [N_qry]

        # Encode query tokens
        qry_tokens = self._encode_query(qry_protein, qry_drug, qry_gnn_emb, qry_go_fp)
        qry_tokens = qry_tokens + self.type_embed(
            torch.ones(n_qry, dtype=torch.long, device=device)
        )
        # Let the transformer see what the binding encoder thinks (detached)
        prior_tok  = self.prior_proj(binding_prior.detach().unsqueeze(-1))  # [N_qry, D]
        qry_tokens = qry_tokens + prior_tok

        # Unit 3: add density signal to query tokens
        density_t   = torch.tensor([[density]], dtype=torch.float32, device=device)  # [1, 1]
        density_tok = self.density_proj(density_t)   # [1, token_dim]
        qry_tokens  = qry_tokens + density_tok       # broadcast over [n_qry, D]

        if n_ctx > 0:
            ctx_tokens = self._encode_context(ctx_protein, ctx_drug, ctx_affinity, ctx_gnn_emb, ctx_go_fp)
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

        # Unit 7: add binding encoder contribution unconditionally.
        # Context and direct binding features are complementary, not substitutes:
        # context answers "what did similar proteins bind?", binding_prior answers
        # "do THIS protein's pocket features match THIS drug's scaffold?".
        # The last layer is zero-initialised so it starts as a no-op and only
        # grows when it genuinely reduces the loss — no manual gating needed.
        mu = mu + binding_prior

        # Unit 3: cold-start sigma bias — upward uncertainty when n_ctx is small
        # Bias decays to zero as density → 1 (warm regime)
        log_sigma = log_sigma + self.cold_start_bias * (1.0 - density)
        sigma = F.softplus(log_sigma) + 1e-4
        self.last_forward_stats = {
            "mu_std": float(mu.detach().std(unbiased=False).item()) if mu.numel() > 1 else 0.0,
            "binding_prior_std": float(binding_prior.detach().std(unbiased=False).item()) if binding_prior.numel() > 1 else 0.0,
            "log_ppr_alpha": float(self.layers[0].attn.log_ppr_alpha.detach().item()) if self.layers else 0.0,
            "centroid_alpha": float(self.centroid_alpha.detach().item()),
            "density": float(density),
        }

        return mu, sigma

    def forward_per_query(
        self,
        pq_protein:  torch.Tensor,          # [N_qry, K, protein_dim]
        pq_drug:     torch.Tensor,          # [N_qry, K, drug_dim]
        pq_affinity: torch.Tensor,          # [N_qry, K, 1]
        pq_ppr:      torch.Tensor,          # [N_qry, K]
        pq_trust:    torch.Tensor,          # [N_qry, K]
        qry_protein: torch.Tensor,          # [N_qry, protein_dim]
        qry_drug:    torch.Tensor,          # [N_qry, drug_dim]
        pq_aff_mean: torch.Tensor,          # [N_qry]  per-query context affinity mean
        pq_gnn_emb:  Optional[torch.Tensor] = None,  # [N_qry, K, gnn_dim]
        qry_gnn_emb: Optional[torch.Tensor] = None,  # [N_qry, gnn_dim]
        pq_go_fp:    Optional[torch.Tensor] = None,  # [N_qry, K, go_fp_dim]
        qry_go_fp:   Optional[torch.Tensor] = None,  # [N_qry, go_fp_dim]
        ppr_centroid: Optional[torch.Tensor] = None,
    ):
        """
        Per-query dynamic context forward pass.

        Each query drug attends to its own K context tokens selected by
        drug_sim × PPR, rather than a shared pool.  Transformer runs as a
        batch of [N_qry, K+1, D] sequences — fully parallel.
        """
        N_qry, K = pq_protein.shape[:2]
        device = qry_protein.device

        density = math.tanh(K / 32.0)

        # Unit 4: PPR centroid blending
        if ppr_centroid is not None:
            alpha_eff = torch.sigmoid(self.centroid_alpha) * (1.0 - density)
            qry_protein = (1.0 - alpha_eff) * qry_protein + alpha_eff * ppr_centroid.unsqueeze(0)

        # Unit 7: binding prior (direct features, detached for transformer path)
        binding_prior = self.binding_encoder(qry_protein, qry_drug)   # [N_qry]

        if K == 0:
            qry_tokens = self._encode_query(qry_protein, qry_drug, qry_gnn_emb, qry_go_fp)
            qry_tokens = qry_tokens + self.type_embed(
                torch.ones(N_qry, dtype=torch.long, device=device)
            )
            qry_tokens = qry_tokens + self.prior_proj(binding_prior.detach().unsqueeze(-1))
            density_t = torch.tensor([[density]], dtype=torch.float32, device=device)
            qry_tokens = qry_tokens + self.density_proj(density_t)

            tokens = qry_tokens.unsqueeze(1)  # [N_qry, 1, D]
            for layer in self.layers:
                tokens = layer(
                    tokens,
                    0,
                    torch.zeros(N_qry, 0, device=device),
                    torch.zeros(N_qry, 0, device=device),
                    attn_mask=None,
                )
            tokens = self.final_norm(tokens)
            qry_out = tokens[:, 0, :]

            pred = self.output_head(qry_out)
            mu = pred[:, 0] + pq_aff_mean + binding_prior
            log_sigma = pred[:, 1] + self.cold_start_bias * (1.0 - density)
            sigma = F.softplus(log_sigma) + 1e-4
            self.last_forward_stats = {
                "mu_std": float(mu.detach().std(unbiased=False).item()) if mu.numel() > 1 else 0.0,
                "binding_prior_std": float(binding_prior.detach().std(unbiased=False).item()) if binding_prior.numel() > 1 else 0.0,
                "log_ppr_alpha": float(self.layers[0].attn.log_ppr_alpha.detach().item()) if self.layers else 0.0,
                "centroid_alpha": float(self.centroid_alpha.detach().item()),
                "density": float(density),
            }
            return mu, sigma

        # --- Encode context tokens (flatten → encode → reshape) ---
        pq_p_flat    = pq_protein.reshape(N_qry * K, -1)
        pq_d_flat    = pq_drug.reshape(N_qry * K, -1)
        pq_a_flat    = pq_affinity.reshape(N_qry * K, 1)
        pq_gnn_flat  = pq_gnn_emb.reshape(N_qry * K, -1) if pq_gnn_emb is not None else None
        pq_go_flat   = pq_go_fp.reshape(N_qry * K, -1)   if pq_go_fp  is not None else None

        ctx_tokens = self._encode_context(pq_p_flat, pq_d_flat, pq_a_flat, pq_gnn_flat, pq_go_flat)
        ctx_tokens = ctx_tokens.view(N_qry, K, self.token_dim)        # [N_qry, K, D]
        ctx_tokens = ctx_tokens + self.type_embed(
            torch.zeros(K, dtype=torch.long, device=device)
        ).unsqueeze(0)                                                 # broadcast over N_qry

        # --- Encode query tokens ---
        qry_tokens = self._encode_query(qry_protein, qry_drug, qry_gnn_emb, qry_go_fp)   # [N_qry, D]
        qry_tokens = qry_tokens + self.type_embed(
            torch.ones(N_qry, dtype=torch.long, device=device)
        )
        # Inject binding prior hint (detached) and density signal
        qry_tokens = qry_tokens + self.prior_proj(binding_prior.detach().unsqueeze(-1))
        density_t  = torch.tensor([[density]], dtype=torch.float32, device=device)
        qry_tokens = qry_tokens + self.density_proj(density_t)        # broadcast

        # --- Combine: [N_qry, K+1, D] ---
        tokens = torch.cat([ctx_tokens, qry_tokens.unsqueeze(1)], dim=1)

        # Mask: context tokens cannot attend to the query (last position)
        mask = torch.zeros(K + 1, K + 1, dtype=torch.bool, device=device)
        mask[:K, K] = True

        # --- Transformer (per-query PPR/trust are [N_qry, K] → 2D mode) ---
        for layer in self.layers:
            tokens = layer(tokens, K, pq_ppr, pq_trust, attn_mask=mask)
        tokens  = self.final_norm(tokens)       # [N_qry, K+1, D]
        qry_out = tokens[:, K, :]               # [N_qry, D]

        pred      = self.output_head(qry_out)   # [N_qry, 2]
        mu        = pred[:, 0]
        log_sigma = pred[:, 1]

        # Per-query affinity anchoring: each drug anchors to its own context mean
        mu = mu + pq_aff_mean + binding_prior

        log_sigma = log_sigma + self.cold_start_bias * (1.0 - density)
        sigma     = F.softplus(log_sigma) + 1e-4
        self.last_forward_stats = {
            "mu_std": float(mu.detach().std(unbiased=False).item()) if mu.numel() > 1 else 0.0,
            "binding_prior_std": float(binding_prior.detach().std(unbiased=False).item()) if binding_prior.numel() > 1 else 0.0,
            "log_ppr_alpha": float(self.layers[0].attn.log_ppr_alpha.detach().item()) if self.layers else 0.0,
            "centroid_alpha": float(self.centroid_alpha.detach().item()),
            "density": float(density),
        }

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
