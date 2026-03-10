import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBiasedMHA(nn.Module):
    """
    Multi-head self-attention with graph-structural additive logit biases
    and a multiplicative value gate for context tokens.

    attn_logit(i → ctx_k) += alpha * log(ppr_k + ε) + delta_proj(delta_k)
    v_ctx_k                *= sigmoid(trust_scale * trust_k)
    """

    def __init__(self, token_dim: int, nhead: int, dropout: float = 0.1, protein_dim: int = 0):
        super().__init__()
        assert token_dim % nhead == 0
        self.nhead    = nhead
        self.head_dim = token_dim // nhead
        self.scale    = self.head_dim ** -0.5
        self.dropout  = dropout

        self.in_proj  = nn.Linear(token_dim, 3 * token_dim, bias=True)
        self.out_proj = nn.Linear(token_dim, token_dim, bias=True)

        # Learnable graph-bias scalars
        # Initialize slightly above zero so the network "feels" the graph immediately
        self.log_ppr_alpha = nn.Parameter(torch.ones(1) * 0.1)   # α in α·log(ppr)
        self.trust_scale   = nn.Parameter(torch.ones(1) * 0.1)   # s in sigmoid(s·decay)

        # Learned projection from protein_dim delta → scalar bias per context token
        # Upgraded to an MLP for non-linear deduction of structural feature spaces
        if protein_dim > 0:
            self.delta_proj = nn.Sequential(
                nn.Linear(protein_dim, token_dim // 2),
                nn.GELU(),
                nn.Linear(token_dim // 2, 1)
            )
            # Initialize the final layer to small values to prevent initial logit explosion
            nn.init.normal_(self.delta_proj[-1].weight, std=0.01)
            nn.init.zeros_(self.delta_proj[-1].bias)
        else:
            self.delta_proj = None

    def forward(
        self,
        x: torch.Tensor,           # [1, N, D]
        n_ctx: int,
        ctx_ppr: torch.Tensor,     # [n_ctx]
        ctx_delta: torch.Tensor,   # [n_ctx, protein_dim]
        ctx_trust: torch.Tensor,   # [n_ctx]
        attn_mask: torch.Tensor | None = None,  # [N, N] bool, True=blocked
    ) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.nhead, self.head_dim

        qkv = self.in_proj(x)                                  # [1, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                        # each [1, N, D]

        q = q.view(B, N, H, d).transpose(1, 2)                # [1, H, N, d]
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)

        # --- V gate for context tokens: sigmoid(trust_scale * decay_weight) ---
        if n_ctx > 0:
            gate = torch.sigmoid(self.trust_scale * ctx_trust)  # [n_ctx]
            # Pad with ones for query positions
            ones = torch.ones(N - n_ctx, device=x.device)
            gate_full = torch.cat([gate, ones], dim=0)          # [N]
            v = v * gate_full.view(1, 1, N, 1)

        # --- Attention logits ---
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [1, H, N, N]

        # --- Graph bias: columns 0..n_ctx-1 (all rows attend to context tokens) ---
        if n_ctx > 0:
            alpha = self.log_ppr_alpha
            ppr_bias = alpha * torch.log(ctx_ppr.clamp(min=1e-8))  # [n_ctx]
            if self.delta_proj is not None:
                delta_bias = self.delta_proj(ctx_delta).squeeze(-1)  # [n_ctx]
            else:
                delta_bias = torch.zeros(n_ctx, device=x.device)
            col_bias = ppr_bias + delta_bias                         # [n_ctx]
            # Broadcast: [1, 1, 1, n_ctx] → added to all [batch, head, row] slices
            pad = torch.zeros(N - n_ctx, device=x.device)
            col_bias_full = torch.cat([col_bias, pad], dim=0)        # [N]
            logits = logits + col_bias_full.view(1, 1, 1, N)

        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask.view(1, 1, N, N), float("-inf"))

        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        out = torch.matmul(weights, v)                         # [1, H, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [1, N, D]
        return self.out_proj(out)


class GraphBiasedTransformerLayer(nn.Module):
    """Pre-norm transformer layer wrapping GraphBiasedMHA."""

    def __init__(self, token_dim: int, nhead: int, dropout: float = 0.1, protein_dim: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.attn  = GraphBiasedMHA(token_dim, nhead, dropout=dropout, protein_dim=protein_dim)
        self.ff    = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, n_ctx, ctx_ppr, ctx_delta, ctx_trust, attn_mask=None):
        # Pre-norm self-attention
        h = self.norm1(x)
        x = x + self.attn(h, n_ctx, ctx_ppr, ctx_delta, ctx_trust, attn_mask)
        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class ProteinLigandTNP(nn.Module):
    """
    Graph-biased Transformer Neural Process for protein-ligand binding prediction.

    Context set: known (protein, drug, affinity) triples from neighbor proteins.
    Query set:   (target_protein, drug) pairs to rank.
    Output:      (mu, sigma) predicted affinity distribution per query.

    Attention incorporates three graph signals for context tokens:
      - PPR score   → additive logit bias: α · log(ppr_k)
      - Delta (structural diff) → additive logit bias: w_delta · delta_k
      - Trust/decay → multiplicative V gate: sigmoid(s · decay_k)
    """

    def __init__(
        self,
        protein_dim: int,
        drug_dim: int,
        token_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim

        # Context encoder: (protein, drug, affinity) → token
        self.ctx_protein_proj  = nn.Linear(protein_dim, token_dim // 2)
        self.ctx_drug_proj     = nn.Linear(drug_dim, token_dim // 2)
        self.ctx_affinity_proj = nn.Linear(1, token_dim // 4)
        self.ctx_fusion = nn.Linear(token_dim // 2 + token_dim // 2 + token_dim // 4, token_dim)

        # Query encoder: (protein, drug) → token  (no affinity at query time)
        self.qry_protein_proj = nn.Linear(protein_dim, token_dim // 2)
        self.qry_drug_proj    = nn.Linear(drug_dim, token_dim // 2)
        self.qry_fusion = nn.Linear(token_dim // 2 + token_dim // 2, token_dim)

        # Type embedding: 0 = context token, 1 = query token
        self.type_embed = nn.Embedding(2, token_dim)

        # Graph-biased transformer layers (share protein_dim for delta projection)
        self.layers = nn.ModuleList([
            GraphBiasedTransformerLayer(token_dim, nhead, dropout=dropout, protein_dim=protein_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(token_dim)

        # Output head: query token → (mu, log_sigma)
        self.output_head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.ReLU(),
            nn.Linear(token_dim // 2, 2),
        )

    def _encode_context(self, ctx_protein, ctx_drug, ctx_affinity):
        p = self.ctx_protein_proj(ctx_protein)
        d = self.ctx_drug_proj(ctx_drug)
        a = self.ctx_affinity_proj(ctx_affinity)
        return self.ctx_fusion(torch.cat([p, d, a], dim=-1))

    def _encode_query(self, qry_protein, qry_drug):
        p = self.qry_protein_proj(qry_protein)
        d = self.qry_drug_proj(qry_drug)
        return self.qry_fusion(torch.cat([p, d], dim=-1))

    def _build_mask(self, n_ctx, n_qry, device):
        """
        Attention mask: query tokens cannot attend to each other.
        Shape [n_ctx + n_qry, n_ctx + n_qry], True = blocked.
        """
        N = n_ctx + n_qry
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        if n_qry > 0:
            mask[n_ctx:, n_ctx:] = True
            idx = torch.arange(n_ctx, N, device=device)
            mask[idx, idx] = False  # each query attends to itself
        return mask

    def forward(
        self,
        ctx_protein: torch.Tensor,    # [N_ctx, protein_dim]
        ctx_drug: torch.Tensor,        # [N_ctx, drug_dim]
        ctx_affinity: torch.Tensor,    # [N_ctx, 1]
        qry_protein: torch.Tensor,     # [N_qry, protein_dim]
        qry_drug: torch.Tensor,        # [N_qry, drug_dim]
        ctx_ppr: torch.Tensor | None = None,    # [N_ctx]
        ctx_delta: torch.Tensor | None = None,  # [N_ctx, protein_dim]
        ctx_trust: torch.Tensor | None = None,  # [N_ctx]
    ):
        device  = qry_protein.device
        n_ctx   = ctx_protein.size(0)
        n_qry   = qry_protein.size(0)

        qry_tokens = self._encode_query(qry_protein, qry_drug)
        qry_tokens = qry_tokens + self.type_embed(
            torch.ones(n_qry, dtype=torch.long, device=device)
        )

        if n_ctx > 0:
            ctx_tokens = self._encode_context(ctx_protein, ctx_drug, ctx_affinity)
            ctx_tokens = ctx_tokens + self.type_embed(
                torch.zeros(n_ctx, dtype=torch.long, device=device)
            )
            tokens = torch.cat([ctx_tokens, qry_tokens], dim=0).unsqueeze(0)  # [1, N, D]
            mask   = self._build_mask(n_ctx, n_qry, device)

            # Default graph signals if not provided
            if ctx_ppr is None:
                ctx_ppr = torch.ones(n_ctx, device=device)
            if ctx_delta is None:
                ctx_delta = torch.zeros(n_ctx, ctx_protein.size(1), device=device)
            if ctx_trust is None:
                ctx_trust = torch.ones(n_ctx, device=device)
        else:
            tokens    = qry_tokens.unsqueeze(0)
            mask      = None
            ctx_ppr   = torch.zeros(0, device=device)
            ctx_delta = torch.zeros(0, ctx_protein.size(1) if ctx_protein.size(1) > 0
                                    else qry_protein.size(1), device=device)
            ctx_trust = torch.zeros(0, device=device)

        for layer in self.layers:
            tokens = layer(tokens, n_ctx, ctx_ppr, ctx_delta, ctx_trust, attn_mask=mask)
        tokens = self.final_norm(tokens).squeeze(0)    # [N, D]

        qry_out = tokens[n_ctx:]                       # [N_qry, D]
        pred    = self.output_head(qry_out)             # [N_qry, 2]
        mu        = pred[:, 0]
        log_sigma = pred[:, 1]
        sigma     = F.softplus(log_sigma) + 1e-4

        return mu, sigma


if __name__ == "__main__":
    prot_dim, drug_dim = 2816, 512
    model = ProteinLigandTNP(protein_dim=prot_dim, drug_dim=drug_dim, token_dim=256)

    ctx_protein  = torch.randn(10, prot_dim)
    ctx_drug     = torch.randn(10, drug_dim)
    ctx_affinity = torch.randn(10, 1)
    ctx_ppr      = torch.rand(10).clamp(1e-6, 1.0)
    ctx_delta    = torch.randn(10, prot_dim)
    ctx_trust    = torch.rand(10)
    qry_protein  = torch.randn(50, prot_dim)
    qry_drug     = torch.randn(50, drug_dim)

    mu, sigma = model(ctx_protein, ctx_drug, ctx_affinity,
                      qry_protein, qry_drug,
                      ctx_ppr, ctx_delta, ctx_trust)
    assert mu.shape == (50,), f"Expected (50,), got {mu.shape}"
    assert sigma.shape == (50,)
    assert (sigma > 0).all()
    print("With graph-biased context: PASSED")

    mu0, sigma0 = model(
        torch.zeros(0, prot_dim), torch.zeros(0, drug_dim), torch.zeros(0, 1),
        qry_protein, qry_drug,
    )
    assert mu0.shape == (50,)
    assert (sigma0 > 0).all()
    print("Cold-start: PASSED")
    print("All tests passed.")
