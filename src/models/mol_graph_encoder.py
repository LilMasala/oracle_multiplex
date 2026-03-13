import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.utils import scatter


class GINEEncoder(nn.Module):
    """Shared GINE-based graph encoder used by both ProteinGraphEncoder and DrugGraphEncoder."""

    def __init__(self, node_in: int, edge_in: int, hidden: int = 256, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.node_proj = nn.Linear(node_in, hidden)
        self.edge_proj = nn.Linear(edge_in, hidden)
        self.convs = nn.ModuleList([
            GINEConv(nn=nn.Sequential(
                nn.Linear(hidden, 2 * hidden), nn.ReLU(),
                nn.Linear(2 * hidden, hidden)
            ))
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_attr, batch) -> Tensor:
        x = F.relu(self.node_proj(x))
        edge_attr = F.relu(self.edge_proj(edge_attr))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class ProteinGraphEncoder(GINEEncoder):
    def __init__(self, node_in=11, edge_in=13, hidden=256, num_layers=4, dropout=0.1):
        super().__init__(node_in, edge_in, hidden, num_layers, dropout)


class DrugGraphEncoder(GINEEncoder):
    def __init__(self, node_in=20, edge_in=5, hidden=256, num_layers=4, dropout=0.1):
        super().__init__(node_in, edge_in, hidden, num_layers, dropout)


class GOEnricher(nn.Module):
    def __init__(self, go_dim=200, hidden=256):
        super().__init__()
        self.go_proj = nn.Linear(go_dim, hidden)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, prot_emb, go_x, pg_edge_index, num_proteins) -> Tensor:
        go_h = F.relu(self.go_proj(go_x))
        go_agg = scatter(
            go_h[pg_edge_index[1]], pg_edge_index[0],
            dim=0, dim_size=num_proteins, reduce='mean'
        )
        ones = torch.ones(pg_edge_index.size(1), device=prot_emb.device)
        count = scatter(ones, pg_edge_index[0], dim=0, dim_size=num_proteins, reduce='sum')
        go_present = (count > 0).float().unsqueeze(-1)
        fused = self.fuse(torch.cat([prot_emb, go_agg], dim=-1))
        return self.norm(prot_emb + go_present * fused)


class BilinearScoringHead(nn.Module):
    def __init__(self, hidden=256, rank=128):
        super().__init__()
        self.p_proj = nn.Linear(hidden, rank, bias=False)
        self.d_proj = nn.Linear(hidden, rank, bias=False)

    def forward(self, p_emb, d_emb) -> Tensor:
        return (self.p_proj(p_emb) * self.d_proj(d_emb)).sum(-1)


class MolGraphPrior(nn.Module):
    def __init__(
        self,
        prot_node_in=11,
        prot_edge_in=13,
        drug_node_in=20,
        drug_edge_in=5,
        go_dim=200,
        n_go_terms: int = 1,
        hidden=256,
        num_layers=4,
        bilinear_rank=128,
        dropout=0.1,
    ):
        super().__init__()
        self.prot_enc = ProteinGraphEncoder(prot_node_in, prot_edge_in, hidden, num_layers, dropout)
        self.drug_enc = DrugGraphEncoder(drug_node_in, drug_edge_in, hidden, num_layers, dropout)
        self.go_emb = nn.Embedding(n_go_terms, go_dim)
        self.go_enr = GOEnricher(go_dim=go_dim, hidden=hidden)
        self.scorer = BilinearScoringHead(hidden=hidden, rank=bilinear_rank)

    def forward(
        self,
        prot_batch: Batch,
        drug_batch: Batch,
        pg_edge_index: Tensor,
        num_proteins: int,
    ) -> Tuple[Tensor, Tensor]:
        p_raw = self.prot_enc(
            prot_batch.x, prot_batch.edge_index, prot_batch.edge_attr, prot_batch.batch
        )
        d = self.drug_enc(
            drug_batch.x, drug_batch.edge_index, drug_batch.edge_attr, drug_batch.batch
        )
        p_full = self.go_enr(p_raw, self.go_emb.weight, pg_edge_index, num_proteins)
        return p_full, d

    def predict_links(
        self,
        z_dict: dict,
        edge_index: Tensor,
        edge_type: tuple = ("protein", "binds_activity", "drug"),
    ) -> Tensor:
        src, dst = edge_index
        return self.scorer(z_dict[edge_type[0]][src], z_dict[edge_type[2]][dst])


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    N_prot, N_drug, N_go = 4, 8, 20
    prot_graphs = [
        Data(x=torch.randn(10, 11),
             edge_index=torch.zeros(2, 0, dtype=torch.long),
             edge_attr=torch.zeros(0, 13))
        for _ in range(N_prot)
    ]
    drug_graphs = [
        Data(x=torch.randn(6, 20),
             edge_index=torch.zeros(2, 0, dtype=torch.long),
             edge_attr=torch.zeros(0, 5))
        for _ in range(N_drug)
    ]
    model = MolGraphPrior(n_go_terms=N_go)
    pb = Batch.from_data_list(prot_graphs)
    db = Batch.from_data_list(drug_graphs)
    pg_ei = torch.zeros(2, 0, dtype=torch.long)
    p_full, d = model(pb, db, pg_ei, N_prot)
    assert p_full.shape == (N_prot, 256), f"Expected ({N_prot}, 256), got {p_full.shape}"
    assert d.shape == (N_drug, 256), f"Expected ({N_drug}, 256), got {d.shape}"
    z = {"protein": p_full, "drug": d}
    ei = torch.tensor([[0, 1], [0, 1]])
    scores = model.predict_links(z, ei)
    assert scores.shape == (2,), f"Expected (2,), got {scores.shape}"
    print("Unit A smoke test PASSED")
