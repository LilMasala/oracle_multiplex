import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch


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
        x = torch.nan_to_num(x, nan=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
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
        hidden=256,
        num_layers=4,
        bilinear_rank=128,
        dropout=0.1,
        # Legacy kwargs ignored — GO enrichment removed
        go_dim=200,
        n_go_terms=1,
    ):
        super().__init__()
        self.prot_enc = ProteinGraphEncoder(prot_node_in, prot_edge_in, hidden, num_layers, dropout)
        self.drug_enc = DrugGraphEncoder(drug_node_in, drug_edge_in, hidden, num_layers, dropout)
        self.scorer = BilinearScoringHead(hidden=hidden, rank=bilinear_rank)

    def forward(
        self,
        prot_batch: Batch,
        drug_batch: Batch,
    ) -> Tuple[Tensor, Tensor]:
        p = self.prot_enc(
            prot_batch.x, prot_batch.edge_index, prot_batch.edge_attr, prot_batch.batch
        )
        d = self.drug_enc(
            drug_batch.x, drug_batch.edge_index, drug_batch.edge_attr, drug_batch.batch
        )
        return p, d

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

    N_prot, N_drug = 4, 8
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
    model = MolGraphPrior()
    pb = Batch.from_data_list(prot_graphs)
    db = Batch.from_data_list(drug_graphs)
    p, d = model(pb, db)
    assert p.shape == (N_prot, 256), f"Expected ({N_prot}, 256), got {p.shape}"
    assert d.shape == (N_drug, 256), f"Expected ({N_drug}, 256), got {d.shape}"
    z = {"protein": p, "drug": d}
    ei = torch.tensor([[0, 1], [0, 1]])
    scores = model.predict_links(z, ei)
    assert scores.shape == (2,), f"Expected (2,), got {scores.shape}"
    print("Unit A smoke test PASSED")
