"""Simple HeteroGraphSAGE with MLP link scorer.

Uses GO, protein-similarity, and drug-similarity edges.
Binding edges are excluded from message passing to avoid label leakage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import PReLU
from torch_geometric.nn import SAGEConv, HeteroConv


class LinkScorer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            PReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            PReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            PReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, src_emb: torch.Tensor, dst_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)


class HeteroGraphSAGE(torch.nn.Module):
    """HeteroSAGE over GO + protein-sim + drug-sim edges with MLP link scorer."""

    def __init__(
        self,
        hidden_channels: int = 256,
        num_layers: int = 3,
        protein_feat_dim: int = 2816,
        go_feat_dim: int = 200,
        drug_feat_dim: int = 512,
        sage_aggr: str = "mean",
        hetero_aggr: str = "sum",
        dropout: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.normalize = normalize

        pf, gf, df = protein_feat_dim, go_feat_dim, drug_feat_dim
        h = hidden_channels

        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList([PReLU() for _ in range(num_layers)])

        # First layer: raw feature dims → hidden
        self.convs.append(HeteroConv({
            ("protein", "relates", "go"):      SAGEConv((pf, gf), h, aggr=sage_aggr),
            ("go", "rev_relates", "protein"):  SAGEConv((gf, pf), h, aggr=sage_aggr),
            ("protein", "similar", "protein"): SAGEConv((pf, pf), h, aggr=sage_aggr),
            ("drug", "similar", "drug"):       SAGEConv((df, df), h, aggr=sage_aggr),
        }, aggr=hetero_aggr))

        # Subsequent layers: hidden → hidden
        for _ in range(num_layers - 1):
            self.convs.append(HeteroConv({
                ("protein", "relates", "go"):      SAGEConv((h, h), h, aggr=sage_aggr, normalize=normalize),
                ("go", "rev_relates", "protein"):  SAGEConv((h, h), h, aggr=sage_aggr, normalize=normalize),
                ("protein", "similar", "protein"): SAGEConv((h, h), h, aggr=sage_aggr, normalize=normalize),
                ("drug", "similar", "drug"):       SAGEConv((h, h), h, aggr=sage_aggr, normalize=normalize),
            }, aggr=hetero_aggr))

        self.scorer = LinkScorer(hidden_channels)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.activations[i](v) for k, v in x_dict.items()}
            if self.dropout:
                x_dict = {k: F.dropout(v, p=0.3, training=self.training) for k, v in x_dict.items()}
        return x_dict

    def predict_links(
        self,
        z_dict: dict,
        edge_index: torch.Tensor,
        edge_type: tuple = ("protein", "binds_activity", "drug"),
    ) -> torch.Tensor:
        src, dst = edge_index
        return self.scorer(z_dict[edge_type[0]][src], z_dict[edge_type[2]][dst])
