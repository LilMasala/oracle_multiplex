import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation
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
        self.pool = AttentionalAggregation(
            gate_nn=nn.Linear(hidden, 1),
            nn=nn.Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr, batch, return_nodes: bool = False):
        x = torch.nan_to_num(x, nan=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0)
        x = F.relu(self.node_proj(x))
        edge_attr = F.relu(self.edge_proj(edge_attr))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = self.pool(x, batch)
        if return_nodes:
            return x, pooled
        return pooled


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


class CrossAttentionScoringHead(nn.Module):
    """
    Drug embedding attends over learned protein 'aspects' via multi-head attention.

    The protein embedding is expanded into num_aspects key/value vectors so the
    drug can selectively weight different facets of the protein pocket.
    Both embeddings are still precomputable; only the scorer needs to run at
    inference time.
    """
    def __init__(self, hidden=256, num_heads=4, num_aspects=8):
        super().__init__()
        self.p_expand = nn.Linear(hidden, num_aspects * hidden)
        self.d_query  = nn.Linear(hidden, hidden)
        self.attn     = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.out      = nn.Linear(hidden, 1)

    def forward(self, p_emb, d_emb) -> Tensor:
        B = p_emb.size(0)
        K = V = self.p_expand(p_emb).view(B, -1, p_emb.size(-1))
        Q = self.d_query(d_emb).unsqueeze(1)
        ctx, _ = self.attn(Q, K, V)
        return self.out(ctx.squeeze(1)).squeeze(-1)


class MLPScoringHead(nn.Module):
    """Simple MLP scorer on concatenated [p_emb, d_emb]. Used with node_cross_attn."""
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, p_emb, d_emb) -> Tensor:
        return self.net(torch.cat([p_emb, d_emb], dim=-1)).squeeze(-1)


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
        scorer="bilinear",
        # Legacy kwargs ignored — GO enrichment removed
        go_dim=200,
        n_go_terms=1,
    ):
        super().__init__()
        self.scorer_type = scorer
        self.prot_enc = ProteinGraphEncoder(prot_node_in, prot_edge_in, hidden, num_layers, dropout)
        self.drug_enc = DrugGraphEncoder(drug_node_in, drug_edge_in, hidden, num_layers, dropout)

        if scorer == "node_cross_attn":
            self.cross_attn   = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True, dropout=dropout)
            self.drug_pool_ctx = AttentionalAggregation(
                gate_nn=nn.Linear(hidden, 1),
                nn=nn.Linear(hidden, hidden),
            )
            self.scorer = MLPScoringHead(hidden=hidden)
        elif scorer == "cross_attn":
            self.scorer = CrossAttentionScoringHead(hidden=hidden)
        else:
            self.scorer = BilinearScoringHead(hidden=hidden, rank=bilinear_rank)

    def _node_cross_attend(self, drug_nodes, drug_batch_vec, prot_nodes, prot_batch_vec) -> Tensor:
        """Drug nodes attend over protein nodes. Returns context-aware drug node embeddings."""
        B      = int(drug_batch_vec.max()) + 1
        hidden = drug_nodes.size(-1)
        device = drug_nodes.device

        prot_sizes = torch.bincount(prot_batch_vec, minlength=B)
        drug_sizes = torch.bincount(drug_batch_vec, minlength=B)
        max_np = int(prot_sizes.max())
        max_nd = int(drug_sizes.max())

        prot_padded  = drug_nodes.new_zeros(B, max_np, hidden)
        drug_padded  = drug_nodes.new_zeros(B, max_nd, hidden)
        # True = padding position (ignored by MHA)
        prot_key_mask = torch.ones(B, max_np, dtype=torch.bool, device=device)

        p_off = d_off = 0
        for i in range(B):
            np_i = int(prot_sizes[i])
            nd_i = int(drug_sizes[i])
            prot_padded[i, :np_i]  = prot_nodes[p_off : p_off + np_i]
            drug_padded[i, :nd_i]  = drug_nodes[d_off : d_off + nd_i]
            prot_key_mask[i, :np_i] = False
            p_off += np_i
            d_off += nd_i

        # Q = drug nodes, K = V = protein nodes
        ctx, _ = self.cross_attn(drug_padded, prot_padded, prot_padded,
                                 key_padding_mask=prot_key_mask)

        return torch.cat([ctx[i, :int(drug_sizes[i])] for i in range(B)], dim=0)

    def encode(
        self,
        prot_batch: Batch,
        drug_batch: Batch,
    ) -> Tuple[Tensor, Tensor]:
        """Return (p_emb, d_emb) without scoring. Used by eval code."""
        if self.scorer_type == "node_cross_attn":
            p_nodes, p_emb = self.prot_enc(
                prot_batch.x, prot_batch.edge_index, prot_batch.edge_attr, prot_batch.batch,
                return_nodes=True,
            )
            d_nodes, _ = self.drug_enc(
                drug_batch.x, drug_batch.edge_index, drug_batch.edge_attr, drug_batch.batch,
                return_nodes=True,
            )
            d_ctx = self._node_cross_attend(d_nodes, drug_batch.batch, p_nodes, prot_batch.batch)
            d_emb = self.drug_pool_ctx(d_ctx, drug_batch.batch)
        else:
            p_emb = self.prot_enc(
                prot_batch.x, prot_batch.edge_index, prot_batch.edge_attr, prot_batch.batch
            )
            d_emb = self.drug_enc(
                drug_batch.x, drug_batch.edge_index, drug_batch.edge_attr, drug_batch.batch
            )
        return p_emb, d_emb

    def forward(
        self,
        prot_batch: Batch,
        drug_batch: Batch,
    ) -> Tensor:
        """Full forward pass returning scores. Used during training so scorer runs inside DDP."""
        p_emb, d_emb = self.encode(prot_batch, drug_batch)
        return self.scorer(p_emb, d_emb)

    def predict_links(
        self,
        z_dict: dict,
        edge_index: Tensor,
        edge_type: tuple = ("protein", "binds_activity", "drug"),
    ) -> Tensor:
        src, dst = edge_index
        return self.scorer(z_dict[edge_type[0]][src], z_dict[edge_type[2]][dst])


class ESMGuidedMolPrior(nn.Module):
    """
    Drug molecular graph encoder guided by ESM-2 protein embeddings.

    Protein side: a single ESM-2 vector (esm_dim=2816) is expanded into
    n_aspects learned token vectors (each hidden-dim) so the drug can attend
    to different protein facets.  No protein GINE — ESM-2 is used directly.

    Drug side: standard GINE on the molecular graph (node_in=20, edge_in=5).

    Cross-attention: drug GINE nodes (Q) attend over protein aspect tokens (K, V).
    Context-aware drug nodes are then pooled and scored against the protein summary.
    """

    scorer_type = "esm_cross_attn"

    def __init__(
        self,
        esm_dim: int = 2816,
        drug_node_in: int = 20,
        drug_edge_in: int = 5,
        hidden: int = 256,
        num_layers: int = 4,
        n_aspects: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_aspects = n_aspects

        # Protein: ESM-2 → n_aspects tokens of size hidden
        self.prot_expand = nn.Linear(esm_dim, n_aspects * hidden)

        # Drug: GINE encoder on molecular graph
        self.drug_enc = DrugGraphEncoder(drug_node_in, drug_edge_in, hidden, num_layers, dropout)

        # Cross-attention: drug nodes attend over protein aspect tokens
        self.cross_attn = nn.MultiheadAttention(
            hidden, num_heads, batch_first=True, dropout=dropout
        )

        # Pool context-aware drug nodes
        self.drug_pool = AttentionalAggregation(
            gate_nn=nn.Linear(hidden, 1),
            nn=nn.Linear(hidden, hidden),
        )

        # Scorer: concatenate protein summary + drug context
        self.scorer = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, esm_emb: Tensor, drug_batch: Batch) -> Tuple[Tensor, Tensor]:
        """
        Returns (prot_summary [B, hidden], drug_ctx [B, hidden]).
        esm_emb: (B, esm_dim) — one ESM-2 vector per protein in the batch.
        """
        B = esm_emb.size(0)
        device = esm_emb.device

        # Protein aspect tokens: (B, n_aspects, hidden)
        prot_tokens = self.prot_expand(esm_emb).view(B, self.n_aspects, self.hidden)
        prot_summary = prot_tokens.mean(dim=1)  # (B, hidden)

        # Drug GINE node embeddings
        drug_nodes, _ = self.drug_enc(
            drug_batch.x, drug_batch.edge_index, drug_batch.edge_attr,
            drug_batch.batch, return_nodes=True,
        )  # (total_drug_nodes, hidden)

        # Pad drug nodes per batch item for batched attention
        drug_sizes = torch.bincount(drug_batch.batch, minlength=B).tolist()
        drug_padded = pad_sequence(
            list(torch.split(drug_nodes, drug_sizes)), batch_first=True
        )  # (B, max_nd, hidden)
        max_nd = drug_padded.size(1)
        # True = padding position to ignore in attention output unpack
        drug_pad_mask = torch.arange(max_nd, device=device).unsqueeze(0) >= \
                        torch.tensor(drug_sizes, device=device).unsqueeze(1)

        # Cross-attention: drug nodes (Q) attend over protein aspect tokens (K, V)
        # Protein has no padding so key_padding_mask=None on protein side
        ctx, _ = self.cross_attn(drug_padded, prot_tokens, prot_tokens)
        # (B, max_nd, hidden)

        # Unpack valid nodes (drop padding) and pool
        ctx_flat = ctx[~drug_pad_mask]  # (total_drug_nodes, hidden), same order as drug_batch.batch
        drug_ctx = self.drug_pool(ctx_flat, drug_batch.batch)  # (B, hidden)

        return prot_summary, drug_ctx

    def forward(self, esm_emb: Tensor, drug_batch: Batch) -> Tensor:
        prot_summary, drug_ctx = self.encode(esm_emb, drug_batch)
        return self.scorer(torch.cat([prot_summary, drug_ctx], dim=-1)).squeeze(-1)


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
    pb = Batch.from_data_list(prot_graphs)
    db = Batch.from_data_list(drug_graphs)
    ei = torch.tensor([[0, 1], [0, 1]])

    for scorer in ["bilinear", "cross_attn", "node_cross_attn"]:
        model = MolGraphPrior(scorer=scorer)
        p, d = model(pb, db)
        assert p.shape == (N_prot, 256), f"[{scorer}] p shape"
        assert d.shape == (N_drug, 256), f"[{scorer}] d shape"
        scores = model.scorer(p, d)
        assert scores.shape == (N_drug,), f"[{scorer}] scores shape"
        print(f"Unit A smoke test PASSED [{scorer}]")
