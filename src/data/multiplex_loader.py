import torch
from torch_geometric.utils import scatter


class MultiplexPillarSampler:
    """Multiplex pillar context via precomputed diffusion priors."""

    def __init__(self, hetero_data, binds_metric="binds_pic50", temporal_decay=0.01, priors_cache_path=None):
        self.data = hetero_data
        self.binds_metric = binds_metric
        self.temporal_decay = temporal_decay
        self.current_episode = 0

        self.form_ei = self.data["protein", "similar", "protein"].edge_index
        self.role_ei = self.data["protein", "go_shared", "protein"].edge_index
        self.binds_ei = self.data["protein", self.binds_metric, "drug"].edge_index
        self.binds_y = self.data["protein", self.binds_metric, "drug"].edge_label
        self.binds_w = getattr(self.data["protein", self.binds_metric, "drug"], "edge_weight", None)
        if self.binds_w is None:
            self.binds_w = torch.ones_like(self.binds_y, dtype=torch.float)

        self.edge_birth_t = torch.zeros(self.binds_y.size(0), device=self.binds_y.device)
        self.protein_x = self.data["protein"].x
        self.num_proteins = int(self.data["protein"].num_nodes)

        self.priors = torch.load(priors_cache_path, weights_only=False) if priors_cache_path is not None else None

        n = self.num_proteins
        self.form_hash = torch.unique((self.form_ei[0] * n + self.form_ei[1]).to(self.protein_x.device))
        self.role_hash = torch.unique((self.role_ei[0] * n + self.role_ei[1]).to(self.protein_x.device))

        self._refresh_bind_sorted_index()

    def _refresh_bind_sorted_index(self):
        if self.binds_ei.size(1) == 0:
            self.bind_src_sorted = torch.empty((0,), dtype=torch.long, device=self.binds_ei.device)
            self.bind_edge_perm = torch.empty((0,), dtype=torch.long, device=self.binds_ei.device)
            return
        self.bind_edge_perm = torch.argsort(self.binds_ei[0])
        self.bind_src_sorted = self.binds_ei[0][self.bind_edge_perm]

    def begin_episode(self, episode_idx):
        self.current_episode = int(episode_idx)

    def _get_diffused_neighbors(self, target_idx):
        device = self.protein_x.device
        if self.priors is None:
            mask_form = self.form_ei[0] == target_idx
            mask_role = self.role_ei[0] == target_idx
            form_neighbors = self.form_ei[1][mask_form]
            role_neighbors = self.role_ei[1][mask_role]
            return (
                form_neighbors,
                torch.ones_like(form_neighbors, dtype=torch.float32),
                role_neighbors,
                torch.ones_like(role_neighbors, dtype=torch.float32),
            )

        ppr_idx = self.priors["ppr_topk_indices"][target_idx].to(device)
        ppr_scores = self.priors["ppr_topk_scores"][target_idx].to(device)
        valid = ppr_idx >= 0
        ppr_idx = ppr_idx[valid]
        ppr_scores = ppr_scores[valid]

        base = torch.full_like(ppr_idx, fill_value=target_idx * self.num_proteins)
        pair_hash = base + ppr_idx
        in_form = torch.isin(pair_hash, self.form_hash)
        in_role = torch.isin(pair_hash, self.role_hash)

        return ppr_idx[in_form], ppr_scores[in_form], ppr_idx[in_role], ppr_scores[in_role]

    def _get_neighbor_binding_edges(self, neighbor_indices):
        if neighbor_indices.numel() == 0 or self.binds_ei.size(1) == 0:
            return (
                torch.empty((2, 0), dtype=torch.long, device=self.binds_ei.device),
                torch.empty((0,), dtype=torch.float, device=self.binds_y.device),
                torch.empty((0,), dtype=torch.float, device=self.binds_w.device),
            )

        n_sorted, _ = torch.sort(neighbor_indices)
        left = torch.searchsorted(self.bind_src_sorted, n_sorted, right=False)
        right = torch.searchsorted(self.bind_src_sorted, n_sorted, right=True)
        lengths = right - left
        valid = lengths > 0
        if not valid.any():
            return (
                torch.empty((2, 0), dtype=torch.long, device=self.binds_ei.device),
                torch.empty((0,), dtype=torch.float, device=self.binds_y.device),
                torch.empty((0,), dtype=torch.float, device=self.binds_w.device),
            )

        left = left[valid]
        lengths = lengths[valid]
        offsets = torch.arange(int(lengths.sum().item()), device=self.binds_ei.device)
        seg_start = torch.repeat_interleave(left, lengths)
        seg_offset = offsets - torch.repeat_interleave(torch.cumsum(lengths, dim=0) - lengths, lengths)
        sorted_positions = seg_start + seg_offset
        edge_ids = self.bind_edge_perm[sorted_positions]

        n_binds_ei = self.binds_ei[:, edge_ids]
        n_binds_y = self.binds_y[edge_ids]
        base_w = self.binds_w[edge_ids]
        age = (self.current_episode - self.edge_birth_t[edge_ids]).clamp(min=0)
        decay = torch.exp(-self.temporal_decay * age)
        return n_binds_ei, n_binds_y, base_w * decay

    def _build_trust_vector(self, target_idx, form_neighbors, role_neighbors):
        device = self.protein_x.device

        # neighbor_binding_density: fraction of structural neighbors with any revealed binding edge
        all_neighbors = torch.cat([form_neighbors, role_neighbors]).unique()
        if all_neighbors.numel() > 0 and self.binds_ei.size(1) > 0:
            n_sorted, _ = torch.sort(all_neighbors)
            left = torch.searchsorted(self.bind_src_sorted, n_sorted, right=False)
            right = torch.searchsorted(self.bind_src_sorted, n_sorted, right=True)
            n_with_data = ((right - left) > 0).sum().item()
            density = torch.tensor(n_with_data / all_neighbors.numel(), dtype=torch.float32, device=device)
        else:
            density = torch.tensor(0.0, device=device)

        if self.priors is not None:
            if "mean_ppr_score" in self.priors:
                mean_ppr = self.priors["mean_ppr_score"][target_idx].to(device).float()
            else:
                mean_ppr = torch.tensor(0.0, device=device)
            return torch.stack(
                [
                    self.priors["participation_coeff"][target_idx].to(device).float(),
                    self.priors["jaccard_overlap"][target_idx].to(device).float(),
                    self.priors["total_neighbor_count"][target_idx].to(device).float(),
                    mean_ppr,
                    density,
                ],
                dim=0,
            )

        form_deg = scatter(torch.ones(self.form_ei.size(1), device=device), self.form_ei[0], dim=0, dim_size=self.num_proteins, reduce="sum")
        role_deg = scatter(torch.ones(self.role_ei.size(1), device=device), self.role_ei[0], dim=0, dim_size=self.num_proteins, reduce="sum")
        kf = form_deg[target_idx]
        kr = role_deg[target_idx]
        kt = kf + kr
        participation = torch.where(kt > 0, 1.0 - ((kf / kt) ** 2 + (kr / kt) ** 2), torch.tensor(0.0, device=device))
        return torch.stack([participation.float(), torch.tensor(0.0, device=device), kt.float(),
                            torch.tensor(0.0, device=device), density], dim=0)

    def _get_ppr_centroid(self, target_idx):
        if self.priors is None or "ppr_protein_centroid" not in self.priors:
            return None
        return self.priors["ppr_protein_centroid"][target_idx].to(self.protein_x.device).float()

    def get_pillar_context(self, target_idx):
        device = self.protein_x.device
        form_neighbors, form_diff_w, role_neighbors, role_diff_w = self._get_diffused_neighbors(target_idx)

        form_binds_ei, form_binds_y, form_binds_w = self._get_neighbor_binding_edges(form_neighbors)
        role_binds_ei, role_binds_y, role_binds_w = self._get_neighbor_binding_edges(role_neighbors)

        form_features = self.protein_x[form_neighbors] if form_neighbors.numel() > 0 else torch.empty((0, self.protein_x.size(1)), device=device)
        role_features = self.protein_x[role_neighbors] if role_neighbors.numel() > 0 else torch.empty((0, self.protein_x.size(1)), device=device)

        return {
            "target_idx": target_idx,
            "target_features": self.protein_x[target_idx],
            "form_neighbors": form_neighbors,
            "form_features": form_features,
            "form_diff_w": form_diff_w,
            "form_binds_ei": form_binds_ei,
            "form_binds_y": form_binds_y,
            "form_binds_w": form_binds_w,
            "role_neighbors": role_neighbors,
            "role_features": role_features,
            "role_diff_w": role_diff_w,
            "role_binds_ei": role_binds_ei,
            "role_binds_y": role_binds_y,
            "role_binds_w": role_binds_w,
            "trust_vector": self._build_trust_vector(target_idx, form_neighbors, role_neighbors),
            "ppr_centroid": self._get_ppr_centroid(target_idx),
        }

    def add_revealed_edges(self, new_edges, new_labels, new_weights=None):
        device = self.binds_ei.device
        if new_weights is None:
            new_weights = torch.ones_like(new_labels, dtype=torch.float)

        self.binds_ei = torch.cat([self.binds_ei, new_edges.to(device)], dim=1)
        self.binds_y = torch.cat([self.binds_y, new_labels.to(device)], dim=0)
        self.binds_w = torch.cat([self.binds_w, new_weights.to(device)], dim=0)

        new_birth = torch.full((new_labels.size(0),), float(self.current_episode), device=device)
        self.edge_birth_t = torch.cat([self.edge_birth_t, new_birth], dim=0)
        self._refresh_bind_sorted_index()
