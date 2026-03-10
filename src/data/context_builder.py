import torch


class TNPContextBuilder:
    def __init__(
        self,
        drug_features: torch.Tensor,      # [N_drugs, drug_dim]
        max_context: int = 256,
        min_affinity_weight: float = 0.1,
    ):
        self.drug_features = drug_features
        self.max_context = max_context
        self.min_affinity_weight = min_affinity_weight

    def _collect_layer(self, neighbor_indices, neighbor_features, binds_ei, binds_y, binds_w):
        """Extract (protein, drug, affinity) triples from one multiplex layer."""
        if binds_ei.size(1) == 0 or neighbor_indices.numel() == 0:
            return [], [], []

        # Filter by weight
        keep = binds_w >= self.min_affinity_weight
        if not keep.any():
            return [], [], []

        ei = binds_ei[:, keep]   # [2, E] — row0=protein_idx, row1=drug_idx
        y  = binds_y[keep]
        drug_idx = ei[1]
        prot_idx = ei[0]

        # Guard against out-of-bounds drug indices
        valid = drug_idx < self.drug_features.size(0)
        if not valid.any():
            return [], [], []
        ei, y, drug_idx, prot_idx = ei[:, valid], y[valid], drug_idx[valid], prot_idx[valid]

        # Map full-graph protein indices → rows in neighbor_features
        # neighbor_indices[k] is the full-graph index of neighbor_features[k]
        prot_list, drug_list, aff_list = [], [], []
        for k, nidx in enumerate(neighbor_indices.tolist()):
            mask = prot_idx == nidx
            if not mask.any():
                continue
            n_edges = int(mask.sum())
            prot_list.append(neighbor_features[k].unsqueeze(0).expand(n_edges, -1))
            drug_list.append(self.drug_features[drug_idx[mask]])
            aff_list.append(y[mask].unsqueeze(1))

        return prot_list, drug_list, aff_list

    def build_context(self, pillar: dict):
        """
        Build TNP context set from a pillar dict returned by MultiplexPillarSampler.

        Returns:
            ctx_protein:  [N_ctx, protein_dim]
            ctx_drug:     [N_ctx, drug_dim]
            ctx_affinity: [N_ctx, 1]
        If no context available, returns zero-size tensors.
        """
        device = pillar["target_features"].device
        protein_dim = pillar["target_features"].size(0)
        drug_dim = self.drug_features.size(1)

        prot_parts, drug_parts, aff_parts = [], [], []

        for layer in ("form", "role"):
            neighbors = pillar[f"{layer}_neighbors"]
            features  = pillar[f"{layer}_features"]
            ei        = pillar[f"{layer}_binds_ei"]
            y         = pillar[f"{layer}_binds_y"]
            w         = pillar[f"{layer}_binds_w"]
            p, d, a   = self._collect_layer(neighbors, features, ei, y, w)
            prot_parts.extend(p)
            drug_parts.extend(d)
            aff_parts.extend(a)

        if not prot_parts:
            return (
                torch.zeros(0, protein_dim, device=device),
                torch.zeros(0, drug_dim, device=device),
                torch.zeros(0, 1, device=device),
            )

        ctx_protein  = torch.cat(prot_parts, dim=0)
        ctx_drug     = torch.cat(drug_parts, dim=0).to(device)
        ctx_affinity = torch.cat(aff_parts,  dim=0)

        # Subsample if over budget
        N = ctx_protein.size(0)
        if N > self.max_context:
            idx = torch.randperm(N, device=device)[:self.max_context]
            ctx_protein  = ctx_protein[idx]
            ctx_drug     = ctx_drug[idx]
            ctx_affinity = ctx_affinity[idx]

        return ctx_protein, ctx_drug, ctx_affinity


if __name__ == "__main__":
    drug_features = torch.randn(1000, 512)
    builder = TNPContextBuilder(drug_features, max_context=64)

    pillar = {
        "target_features": torch.randn(2816),
        "form_neighbors": torch.tensor([0, 1, 2]),
        "form_features": torch.randn(3, 2816),
        "form_binds_ei": torch.stack([torch.tensor([0, 0, 1, 2]),
                                       torch.tensor([0, 1, 2, 3])]),
        "form_binds_y": torch.randn(4),
        "form_binds_w": torch.ones(4),
        "role_neighbors": torch.zeros(0, dtype=torch.long),
        "role_features": torch.zeros(0, 2816),
        "role_binds_ei": torch.zeros(2, 0, dtype=torch.long),
        "role_binds_y": torch.zeros(0),
        "role_binds_w": torch.zeros(0),
        "trust_vector": torch.zeros(5),
        "ppr_centroid": torch.zeros(2816),
    }

    ctx_p, ctx_d, ctx_a = builder.build_context(pillar)
    assert ctx_p.shape[1] == 2816
    assert ctx_d.shape[1] == 512
    assert ctx_a.shape[1] == 1
    assert ctx_p.shape[0] == ctx_d.shape[0] == ctx_a.shape[0]
    print(f"Context size: {ctx_p.shape[0]} — PASSED")

    pillar_cold = {**pillar, "form_neighbors": torch.zeros(0, dtype=torch.long),
                   "form_features": torch.zeros(0, 2816),
                   "form_binds_ei": torch.zeros(2, 0, dtype=torch.long),
                   "form_binds_y": torch.zeros(0), "form_binds_w": torch.zeros(0)}
    ctx_p0, ctx_d0, ctx_a0 = builder.build_context(pillar_cold)
    assert ctx_p0.shape[0] == 0
    print("Cold-start: PASSED")
