"""Unit 6: 2-layer GAT protein GNN pre-encoder for multiplex graph (form + role layers)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class ProteinGNN(nn.Module):
    """
    2-layer GAT on form and role layers separately, fused via MLP.
    Output: [N_proteins, out_dim] — residual embeddings encoding neighborhood structure.

    Even with random initialization, GAT aggregates neighborhood features,
    giving each protein an embedding that encodes its multiplex graph context.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = dropout
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads

        # Form layer GAT (structural similarity)
        self.form_gat1 = GATConv(in_dim, head_dim, heads=heads, dropout=dropout, add_self_loops=True)
        self.form_gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, add_self_loops=True)

        # Role layer GAT (functional GO-shared)
        self.role_gat1 = GATConv(in_dim, head_dim, heads=heads, dropout=dropout, add_self_loops=True)
        self.role_gat2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, add_self_loops=True)

        # Fusion MLP: [hidden_dim * 2] → [out_dim]
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        form_edge_index: torch.Tensor,
        role_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [N_proteins, in_dim] protein features
            form_edge_index: [2, E_form] structural similarity edges
            role_edge_index: [2, E_role] functional GO-shared edges
        Returns:
            emb: [N_proteins, out_dim]
        """
        # Form branch — guard against empty edge_index
        if form_edge_index.size(1) > 0:
            h_form = F.elu(self.form_gat1(x, form_edge_index))
            h_form = F.dropout(h_form, p=self.dropout, training=self.training)
            h_form = self.form_gat2(h_form, form_edge_index)
        else:
            h_form = torch.zeros(x.size(0), self.form_gat2.out_channels, device=x.device)

        # Role branch — guard against empty edge_index
        if role_edge_index.size(1) > 0:
            h_role = F.elu(self.role_gat1(x, role_edge_index))
            h_role = F.dropout(h_role, p=self.dropout, training=self.training)
            h_role = self.role_gat2(h_role, role_edge_index)
        else:
            h_role = torch.zeros(x.size(0), self.role_gat2.out_channels, device=x.device)

        h = torch.cat([h_form, h_role], dim=-1)  # [N, hidden_dim * 2]
        return self.norm(self.fusion(h))


def compute_all_embeddings(
    model: ProteinGNN,
    protein_x: torch.Tensor,
    form_edge_index: torch.Tensor,
    role_edge_index: torch.Tensor,
) -> torch.Tensor:
    """Precompute GNN embeddings for all proteins (detached, for use as features)."""
    model.eval()
    with torch.no_grad():
        embs = model(protein_x, form_edge_index, role_edge_index)
    return embs.detach()


if __name__ == "__main__":
    torch.manual_seed(42)
    N, D = 100, 2816
    x = torch.randn(N, D)
    src = torch.randint(0, N, (200,))
    dst = torch.randint(0, N, (200,))
    form_ei = torch.stack([src, dst])
    role_ei = torch.stack([dst, src])  # reverse for variety

    gnn = ProteinGNN(in_dim=D, hidden_dim=128, out_dim=256, heads=4)
    out = gnn(x, form_ei, role_ei)
    assert out.shape == (N, 256), f"Expected ({N}, 256), got {out.shape}"
    print(f"ProteinGNN output: {out.shape} — PASSED")

    # Test with empty edges
    out_empty = gnn(x, torch.zeros(2, 0, dtype=torch.long), form_ei)
    assert out_empty.shape == (N, 256)
    print("Empty form edge: PASSED")
