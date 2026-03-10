import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinLigandTNP(nn.Module):
    """
    Transformer Neural Process for protein-ligand binding affinity prediction.

    Context set: known (protein, drug, affinity) triples from neighbor proteins.
    Query set:   (target_protein, drug) pairs to rank.
    Output:      (mu, sigma) predicted affinity distribution per query.
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=token_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        Attention mask: query tokens cannot attend to other query tokens.
        Shape [n_ctx + n_qry, n_ctx + n_qry], True = blocked.
        """
        N = n_ctx + n_qry
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        if n_qry > 0:
            mask[n_ctx:, n_ctx:] = True
            # Each query attends to itself
            idx = torch.arange(n_ctx, N, device=device)
            mask[idx, idx] = False
        return mask

    def forward(
        self,
        ctx_protein: torch.Tensor,    # [N_ctx, protein_dim]
        ctx_drug: torch.Tensor,        # [N_ctx, drug_dim]
        ctx_affinity: torch.Tensor,    # [N_ctx, 1]
        qry_protein: torch.Tensor,     # [N_qry, protein_dim]
        qry_drug: torch.Tensor,        # [N_qry, drug_dim]
    ):
        device = qry_protein.device
        n_ctx = ctx_protein.size(0)
        n_qry = qry_protein.size(0)

        qry_tokens = self._encode_query(qry_protein, qry_drug)         # [N_qry, D]
        qry_tokens = qry_tokens + self.type_embed(torch.ones(n_qry, dtype=torch.long, device=device))

        if n_ctx > 0:
            ctx_tokens = self._encode_context(ctx_protein, ctx_drug, ctx_affinity)  # [N_ctx, D]
            ctx_tokens = ctx_tokens + self.type_embed(torch.zeros(n_ctx, dtype=torch.long, device=device))
            tokens = torch.cat([ctx_tokens, qry_tokens], dim=0).unsqueeze(0)        # [1, N, D]
            mask = self._build_mask(n_ctx, n_qry, device)
        else:
            tokens = qry_tokens.unsqueeze(0)   # [1, N_qry, D]
            mask = None

        out = self.transformer(tokens, mask=mask).squeeze(0)  # [N, D]
        qry_out = out[n_ctx:]                                  # [N_qry, D]

        pred = self.output_head(qry_out)                       # [N_qry, 2]
        mu        = pred[:, 0]
        log_sigma = pred[:, 1]
        sigma     = F.softplus(log_sigma) + 1e-4

        return mu, sigma


if __name__ == "__main__":
    model = ProteinLigandTNP(protein_dim=2816, drug_dim=512, token_dim=256)

    ctx_protein  = torch.randn(10, 2816)
    ctx_drug     = torch.randn(10, 512)
    ctx_affinity = torch.randn(10, 1)
    qry_protein  = torch.randn(50, 2816)
    qry_drug     = torch.randn(50, 512)

    mu, sigma = model(ctx_protein, ctx_drug, ctx_affinity, qry_protein, qry_drug)
    assert mu.shape == (50,), f"Expected (50,), got {mu.shape}"
    assert sigma.shape == (50,)
    assert (sigma > 0).all()
    print("With context: PASSED")

    mu0, sigma0 = model(
        torch.zeros(0, 2816), torch.zeros(0, 512), torch.zeros(0, 1),
        qry_protein, qry_drug,
    )
    assert mu0.shape == (50,)
    assert (sigma0 > 0).all()
    print("Cold-start: PASSED")
    print("All tests passed.")
