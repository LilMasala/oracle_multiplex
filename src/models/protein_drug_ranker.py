"""ProteinDrugRanker: biology-informed, modality-aware ranker with FiLM conditioning.

Uses GO term aggregation, ESM+CATH protein modality fusion, drug residual
connections, and a choice of bilinear / cosine / MLP interaction head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import PReLU
from torch_geometric.nn import SAGEConv, HeteroConv


class ProteinDrugRanker(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int = 256,
        num_layers: int = 3,
        protein_feat_dim: int = 2816,
        go_feat_dim: int = 200,
        drug_feat_dim: int = 512,
        protein_esm_dim: int = 2048,
        protein_cath_dim: int = 768,
        use_go_modality: bool = True,
        sage_aggr: str = "mean",
        hetero_aggr: str = "mean",
        dropout: bool = True,
        normalize: bool = True,
        conditioning: str = "film",   # "film" | "none"
        head: str = "bilinear",        # "bilinear" | "cosine" | "mlp"
        bilinear_rank: int = 128,
        film_strength: float = 0.2,
    ):
        super().__init__()
        self.hidden_channels = h = int(hidden_channels)
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        self.use_go_modality = use_go_modality
        self.conditioning = conditioning
        self.head = head
        self.bilinear_rank = bilinear_rank
        self.film_strength = film_strength
        self.protein_esm_dim = protein_esm_dim
        self.protein_cath_dim = protein_cath_dim

        # --- Protein modality encoders ---
        self.proj_protein_esm  = nn.Sequential(nn.Linear(protein_esm_dim, h), PReLU())
        self.proj_protein_cath = nn.Sequential(nn.Linear(protein_cath_dim, h), PReLU())

        if use_go_modality:
            self.proj_protein_go = nn.Sequential(nn.Linear(go_feat_dim, h), PReLU())
            gate_in = h * 3 + 1
            self.n_modalities = 3
        else:
            self.proj_protein_go = None
            gate_in = h * 2
            self.n_modalities = 2

        self.protein_gate = nn.Sequential(
            nn.Linear(gate_in, h), PReLU(), nn.Dropout(0.1),
            nn.Linear(h, self.n_modalities),
        )
        self.protein_fuse = nn.Sequential(
            nn.Linear(h, h), PReLU(), nn.Dropout(0.1), nn.Linear(h, h), PReLU(),
        )

        # --- Drug encoder (with residual skip) ---
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_feat_dim, h), PReLU(), nn.Dropout(0.1), nn.Linear(h, h), PReLU(),
        )

        # --- Message passing (no binding edges) ---
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList([PReLU() for _ in range(num_layers)])

        self.convs.append(HeteroConv({
            ("protein", "relates", "go"):      SAGEConv((h, go_feat_dim), h, aggr=sage_aggr),
            ("go", "rev_relates", "protein"):  SAGEConv((go_feat_dim, h), h, aggr=sage_aggr),
            ("drug", "similar", "drug"):       SAGEConv((h, h), h, aggr=sage_aggr),
        }, aggr=hetero_aggr))

        for _ in range(num_layers - 1):
            self.convs.append(HeteroConv({
                ("protein", "relates", "go"):      SAGEConv((h, h), h, aggr=sage_aggr, normalize=normalize),
                ("go", "rev_relates", "protein"):  SAGEConv((h, h), h, aggr=sage_aggr, normalize=normalize),
                ("drug", "similar", "drug"):       SAGEConv((h, h), h, aggr=sage_aggr, normalize=normalize),
            }, aggr=hetero_aggr))

        # --- FiLM conditioning ---
        if conditioning == "film":
            self.film = nn.Sequential(
                nn.Linear(h, h), PReLU(), nn.Linear(h, h * 2),
            )
        else:
            self.film = None

        # --- Interaction head ---
        if head == "bilinear":
            self.p_bilin = nn.Linear(h, bilinear_rank, bias=False)
            self.d_bilin = nn.Linear(h, bilinear_rank, bias=False)
        elif head == "mlp":
            self.mlp_head = nn.Sequential(
                nn.Linear(h * 4, 256), PReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), PReLU(), nn.Dropout(0.2),
                nn.Linear(128, 1),
            )

    # ------------------------------------------------------------------
    def _aggregate_go(self, x_go, edge_pg, num_proteins):
        if edge_pg is None or edge_pg.numel() == 0:
            return x_go.new_zeros(num_proteins, x_go.size(-1)), x_go.new_zeros(num_proteins, 1)
        src, dst = edge_pg[0].long(), edge_pg[1].long()
        go_sum = x_go.new_zeros(num_proteins, x_go.size(-1))
        go_sum.index_add_(0, src, x_go[dst])
        counts = x_go.new_zeros(num_proteins, 1)
        counts.index_add_(0, src, x_go.new_ones(dst.numel(), 1))
        go_present = (counts > 0).float()
        return go_sum / counts.clamp(min=1.0), go_present

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        xp = x_dict["protein"]
        esm  = xp[:, :self.protein_esm_dim]
        cath = xp[:, self.protein_esm_dim:self.protein_esm_dim + self.protein_cath_dim]

        h_esm  = self.proj_protein_esm(esm)
        h_cath = self.proj_protein_cath(cath)

        if self.use_go_modality and "go" in x_dict:
            x_go = x_dict["go"]
            e_pg = edge_index_dict.get(("protein", "relates", "go"), None)
            go_mean, go_present = self._aggregate_go(x_go, e_pg, xp.size(0))
            h_go = self.proj_protein_go(go_mean)
            gate_in = torch.cat([h_esm, h_cath, h_go, go_present], dim=-1)
            w = torch.softmax(self.protein_gate(gate_in), dim=-1)
            h_mix = w[:, 0:1] * h_esm + w[:, 1:2] * h_cath + w[:, 2:3] * h_go
        else:
            gate_in = torch.cat([h_esm, h_cath], dim=-1)
            w = torch.softmax(self.protein_gate(gate_in), dim=-1)
            h_mix = w[:, 0:1] * h_esm + w[:, 1:2] * h_cath

        x_dict["protein"] = self.protein_fuse(h_mix)
        x_dict["drug"] = self.drug_encoder(x_dict["drug"])
        drug_residual = x_dict["drug"]

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: self.activations[i](v) for k, v in x_dict.items()}
            if self.dropout:
                x_dict = {k: F.dropout(v, p=0.2, training=self.training) for k, v in x_dict.items()}

        x_dict["drug"] = x_dict["drug"] + drug_residual
        return x_dict

    def _apply_film(self, p_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tensor:
        if self.film is None:
            return d_emb
        gb = self.film(p_emb)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        gamma = 1.0 + self.film_strength * torch.tanh(gamma)
        beta  = self.film_strength * torch.tanh(beta)
        return gamma * d_emb + beta

    def predict_links(
        self,
        z_dict: dict,
        edge_index: torch.Tensor,
        edge_type: tuple = ("protein", "binds_activity", "drug"),
    ) -> torch.Tensor:
        src, dst = edge_index
        p_emb = z_dict[edge_type[0]][src]
        d_emb = self._apply_film(p_emb, z_dict[edge_type[2]][dst])

        if self.head == "cosine":
            return (F.normalize(p_emb, dim=-1) * F.normalize(d_emb, dim=-1)).sum(-1)
        if self.head == "bilinear":
            return (self.p_bilin(p_emb) * self.d_bilin(d_emb)).sum(-1)
        phi = torch.cat([p_emb, d_emb, p_emb * d_emb, (p_emb - d_emb).abs()], dim=-1)
        return self.mlp_head(phi).view(-1)
