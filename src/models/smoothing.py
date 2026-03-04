import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class MultiplexInductiveSmoother(nn.Module):
    """
    Multiplex smoother with hierarchical fusion:
      1) Build neighbor preference vectors per layer.
      2) Intra-layer attention aggregation -> v_form, v_role.
      3) Inter-layer mixer with trust priors -> v_prior.
    """

    def __init__(self, protein_dim, drug_dim, trust_dim=3, baseline_pic50=6.0):
        super().__init__()
        self.baseline = baseline_pic50
        self.drug_dim = drug_dim

        self.form_refiner = nn.Sequential(
            nn.Linear(protein_dim, protein_dim),
            nn.PReLU(),
            nn.Linear(protein_dim, protein_dim),
        )
        self.role_refiner = nn.Sequential(
            nn.Linear(protein_dim, protein_dim),
            nn.PReLU(),
            nn.Linear(protein_dim, protein_dim),
        )

        self.q_proj = nn.Linear(protein_dim, drug_dim)
        self.k_proj = nn.Linear(drug_dim, drug_dim)
        self.v_proj = nn.Linear(drug_dim, drug_dim)

        self.layer_emb = nn.Embedding(2, 16)
        attn_in_dim = (protein_dim * 2) + 16
        self.attn_net = nn.Sequential(
            nn.Linear(attn_in_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

        # Cross-layer mixer: [v_form, v_role, trust_vector] -> [w_form, w_role]
        self.layer_mixer = nn.Sequential(
            nn.Linear((2 * drug_dim) + trust_dim, 64),
            nn.PReLU(),
            nn.Linear(64, 2),
        )

        self.integration_mlp = nn.Sequential(
            nn.Linear(drug_dim, protein_dim),
            nn.PReLU(),
            nn.Linear(protein_dim, protein_dim),
        )
        self.norm = nn.LayerNorm(protein_dim)
        self.delta_norm = nn.LayerNorm(protein_dim, elementwise_affine=False)
        self.delta_gate = nn.Parameter(torch.tensor(0.0))

    def _build_preference_vectors(self, z_target, neighbor_ids, binds_ei, binds_y, binds_w, drug_features):
        device = drug_features.device
        num_neighbors = neighbor_ids.size(0)
        messages = torch.zeros((num_neighbors, self.drug_dim), device=device)
        if num_neighbors == 0 or binds_ei.numel() == 0:
            return messages

        sorted_neighbors, inverse = torch.sort(neighbor_ids)
        edge_src = binds_ei[0]
        pos = torch.searchsorted(sorted_neighbors, edge_src)
        clipped = pos.clamp(max=max(sorted_neighbors.numel() - 1, 0))
        valid = (pos < sorted_neighbors.numel()) & (sorted_neighbors[clipped] == edge_src)
        if not valid.any():
            return messages

        neighbor_pos = inverse[clipped[valid]]
        drug_idx = binds_ei[1][valid]
        affinities = (binds_y[valid] - self.baseline)
        reliabilities = binds_w[valid]

        q = self.q_proj(z_target).view(1, -1)
        d_feats = drug_features[drug_idx]
        k = self.k_proj(d_feats)
        v = self.v_proj(d_feats) * (affinities * reliabilities).unsqueeze(-1)
        logits = torch.matmul(k, q.t()).squeeze(-1) / math.sqrt(self.drug_dim)

        # Softmax by neighbor group using scatter primitives.
        max_per_neighbor = scatter(logits, neighbor_pos, dim=0, dim_size=num_neighbors, reduce="max")
        centered = logits - max_per_neighbor[neighbor_pos]
        exp_logits = torch.exp(centered)
        denom = scatter(exp_logits, neighbor_pos, dim=0, dim_size=num_neighbors, reduce="sum") + 1e-12
        attn = exp_logits / denom[neighbor_pos]

        weighted_v = v * attn.unsqueeze(-1)
        messages = scatter(weighted_v, neighbor_pos, dim=0, dim_size=num_neighbors, reduce="sum")
        return messages

    def _compute_attention(self, z_target, z_neighbors, layer_id):
        num_neighbors = z_neighbors.size(0)
        if num_neighbors == 0:
            return torch.empty((0,), device=z_target.device)

        z_target_exp = z_target.unsqueeze(0).expand(num_neighbors, -1)
        l_emb = self.layer_emb(torch.tensor([layer_id], device=z_target.device)).expand(num_neighbors, -1)
        attn_input = torch.cat([z_target_exp, z_neighbors, l_emb], dim=-1)
        return self.attn_net(attn_input).squeeze(-1)

    def _aggregate_layer(self, msgs, attn_logits, diff_w):
        if msgs.size(0) == 0:
            return torch.zeros((self.drug_dim,), device=msgs.device), torch.empty((0,), device=msgs.device)

        if diff_w.numel() == msgs.size(0):
            attn_logits = attn_logits + torch.log(diff_w.clamp_min(1e-12))
        attn = F.softmax(attn_logits, dim=0)
        v_layer = (attn.unsqueeze(-1) * msgs).sum(dim=0)
        return v_layer, attn

    def _aggregate_delta_layer(self, z_target, z_neighbors, attn):
        # z_target: [protein_dim], z_neighbors: [N_neighbors, protein_dim], attn: [N_neighbors]
        if z_neighbors.size(0) == 0:
            return torch.zeros_like(z_target)
        delta_neighbors = z_target.unsqueeze(0) - z_neighbors
        return (attn.unsqueeze(-1) * delta_neighbors).sum(dim=0)

    def forward(self, pillar_data, drug_features):
        z_target_form = self.form_refiner(pillar_data["target_features"])
        z_target_role = self.role_refiner(pillar_data["target_features"])
        z_target_refined = 0.5 * (z_target_form + z_target_role)

        form_feats_refined = self.form_refiner(pillar_data["form_features"])
        role_feats_refined = self.role_refiner(pillar_data["role_features"])

        form_msgs = self._build_preference_vectors(
            z_target_refined,
            pillar_data["form_neighbors"],
            pillar_data["form_binds_ei"],
            pillar_data["form_binds_y"],
            pillar_data["form_binds_w"],
            drug_features,
        )
        role_msgs = self._build_preference_vectors(
            z_target_refined,
            pillar_data["role_neighbors"],
            pillar_data["role_binds_ei"],
            pillar_data["role_binds_y"],
            pillar_data["role_binds_w"],
            drug_features,
        )

        form_logits = self._compute_attention(z_target_refined, form_feats_refined, layer_id=0)
        role_logits = self._compute_attention(z_target_refined, role_feats_refined, layer_id=1)

        v_form, form_attn = self._aggregate_layer(form_msgs, form_logits, pillar_data.get("form_diff_w", torch.empty(0, device=z_target_refined.device)))
        v_role, role_attn = self._aggregate_layer(role_msgs, role_logits, pillar_data.get("role_diff_w", torch.empty(0, device=z_target_refined.device)))

        trust_vector = pillar_data.get("trust_vector", torch.zeros(3, device=z_target_refined.device)).float()
        mixer_in = torch.cat([v_form, v_role, trust_vector], dim=0).unsqueeze(0)
        layer_weights = F.softmax(self.layer_mixer(mixer_in), dim=-1).squeeze(0)
        w_form, w_role = layer_weights[0], layer_weights[1]

        v_prior = (w_form * v_form) + (w_role * v_role)

        # Delta channel over same neighborhood sets used for v_prior.
        form_delta = self._aggregate_delta_layer(z_target_refined, form_feats_refined, form_attn)
        role_delta = self._aggregate_delta_layer(z_target_refined, role_feats_refined, role_attn)
        delta_raw = (w_form * form_delta) + (w_role * role_delta)
        delta_mean = self.delta_gate * self.delta_norm(delta_raw)

        z_refined = self.norm(z_target_refined + self.integration_mlp(v_prior))

        floor_stats = {
            "form_attn": form_attn,
            "role_attn": role_attn,
            "w_form": w_form.detach(),
            "w_role": w_role.detach(),
            "delta_norm": delta_mean.norm(p=2).detach(),
        }
        return z_refined, v_prior, delta_mean, floor_stats
