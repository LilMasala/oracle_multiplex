import torch


class TNPContextBuilder:
    def __init__(
        self,
        drug_features: torch.Tensor,      # [N_drugs, drug_dim]
        max_context: int = 256,
        min_affinity_weight: float = 0.0,
    ):
        self.drug_features = drug_features
        self.max_context = max_context
        self.min_affinity_weight = min_affinity_weight

    def _collect_layer(self, neighbor_indices, neighbor_features, diff_w,
                       binds_ei, binds_y, binds_w, target_features):
        """Extract (protein, drug, affinity, ppr, delta, trust) tuples from one layer."""
        if binds_ei.size(1) == 0 or neighbor_indices.numel() == 0:
            return [], [], [], [], [], []

        keep = binds_w >= self.min_affinity_weight
        if not keep.any():
            return [], [], [], [], [], []

        ei = binds_ei[:, keep]
        y  = binds_y[keep]
        w  = binds_w[keep]
        drug_idx = ei[1]
        prot_idx = ei[0]

        valid = drug_idx < self.drug_features.size(0)
        if not valid.any():
            return [], [], [], [], [], []
        ei, y, w = ei[:, valid], y[valid], w[valid]
        drug_idx, prot_idx = drug_idx[valid], prot_idx[valid]

        prot_list, drug_list, aff_list = [], [], []
        ppr_list, delta_list, trust_list = [], [], []

        for k, nidx in enumerate(neighbor_indices.tolist()):
            mask = prot_idx == nidx
            if not mask.any():
                continue
            n_edges = int(mask.sum())
            prot_list.append(neighbor_features[k].unsqueeze(0).expand(n_edges, -1))
            drug_list.append(self.drug_features[drug_idx[mask]])
            aff_list.append(y[mask].unsqueeze(1))

            # PPR score for this neighbor (scalar broadcast across edges)
            ppr_val = diff_w[k].item() if diff_w.numel() > k else 1.0
            ppr_list.append(torch.full((n_edges,), ppr_val, device=binds_ei.device))

            # Structural delta: target - neighbor, broadcast across edges
            delta = (target_features - neighbor_features[k]).unsqueeze(0).expand(n_edges, -1)
            delta_list.append(delta)

            # Trust = temporal-decay-weighted edge weight
            trust_list.append(w[mask])

        return prot_list, drug_list, aff_list, ppr_list, delta_list, trust_list

    def build_context(self, pillar: dict):
        """
        Build TNP context set from a pillar dict returned by MultiplexPillarSampler.

        Returns:
            ctx_protein:  [N_ctx, protein_dim]
            ctx_drug:     [N_ctx, drug_dim]
            ctx_affinity: [N_ctx, 1]
            ctx_ppr:      [N_ctx]   — PPR score of each context protein to target
            ctx_delta:    [N_ctx, protein_dim] — target_features - neighbor_features
            ctx_trust:    [N_ctx]   — temporal-decay-weighted edge weight
        If no context available, returns zero-size tensors.
        """
        device = pillar["target_features"].device
        protein_dim = pillar["target_features"].size(0)
        drug_dim = self.drug_features.size(1)
        target_features = pillar["target_features"]

        prot_parts, drug_parts, aff_parts = [], [], []
        ppr_parts, delta_parts, trust_parts = [], [], []

        for layer in ("form", "role"):
            neighbors = pillar[f"{layer}_neighbors"]
            features  = pillar[f"{layer}_features"]
            diff_w    = pillar[f"{layer}_diff_w"]
            ei        = pillar[f"{layer}_binds_ei"]
            y         = pillar[f"{layer}_binds_y"]
            w         = pillar[f"{layer}_binds_w"]
            p, d, a, ppr, delta, trust = self._collect_layer(
                neighbors, features, diff_w, ei, y, w, target_features
            )
            prot_parts.extend(p)
            drug_parts.extend(d)
            aff_parts.extend(a)
            ppr_parts.extend(ppr)
            delta_parts.extend(delta)
            trust_parts.extend(trust)

        if not prot_parts:
            return (
                torch.zeros(0, protein_dim, device=device),
                torch.zeros(0, drug_dim, device=device),
                torch.zeros(0, 1, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, protein_dim, device=device),
                torch.zeros(0, device=device),
            )

        ctx_protein  = torch.cat(prot_parts, dim=0)
        ctx_drug     = torch.cat(drug_parts, dim=0).to(device)
        ctx_affinity = torch.cat(aff_parts,  dim=0)
        ctx_ppr      = torch.cat(ppr_parts,  dim=0)
        ctx_delta    = torch.cat(delta_parts, dim=0)
        ctx_trust    = torch.cat(trust_parts, dim=0)

        N = ctx_protein.size(0)
        if N > self.max_context:
            # Sample proportional to PPR score — prefer edges from graph-close neighbors
            weights = ctx_ppr.clamp(min=1e-8)
            idx = torch.multinomial(weights, self.max_context, replacement=False)
            ctx_protein  = ctx_protein[idx]
            ctx_drug     = ctx_drug[idx]
            ctx_affinity = ctx_affinity[idx]
            ctx_ppr      = ctx_ppr[idx]
            ctx_delta    = ctx_delta[idx]
            ctx_trust    = ctx_trust[idx]

        return ctx_protein, ctx_drug, ctx_affinity, ctx_ppr, ctx_delta, ctx_trust


if __name__ == "__main__":
    drug_features = torch.randn(1000, 512)
    builder = TNPContextBuilder(drug_features, max_context=64)
    protein_dim = 2816

    pillar = {
        "target_features": torch.randn(protein_dim),
        "form_neighbors": torch.tensor([0, 1, 2]),
        "form_features": torch.randn(3, protein_dim),
        "form_diff_w": torch.tensor([0.8, 0.5, 0.3]),
        "form_binds_ei": torch.stack([torch.tensor([0, 0, 1, 2]),
                                      torch.tensor([0, 1, 2, 3])]),
        "form_binds_y": torch.randn(4),
        "form_binds_w": torch.ones(4),
        "role_neighbors": torch.zeros(0, dtype=torch.long),
        "role_features": torch.zeros(0, protein_dim),
        "role_diff_w": torch.zeros(0),
        "role_binds_ei": torch.zeros(2, 0, dtype=torch.long),
        "role_binds_y": torch.zeros(0),
        "role_binds_w": torch.zeros(0),
        "trust_vector": torch.zeros(5),
        "ppr_centroid": torch.zeros(protein_dim),
    }

    ctx_p, ctx_d, ctx_a, ctx_ppr, ctx_delta, ctx_trust = builder.build_context(pillar)
    assert ctx_p.shape[1] == protein_dim
    assert ctx_d.shape[1] == 512
    assert ctx_a.shape[1] == 1
    N = ctx_p.shape[0]
    assert ctx_ppr.shape == (N,)
    assert ctx_delta.shape == (N, protein_dim)
    assert ctx_trust.shape == (N,)
    assert ctx_p.shape[0] == ctx_d.shape[0] == ctx_a.shape[0]
    print(f"Context size: {N} — PASSED")

    cold = {**pillar,
            "form_neighbors": torch.zeros(0, dtype=torch.long),
            "form_features": torch.zeros(0, protein_dim),
            "form_diff_w": torch.zeros(0),
            "form_binds_ei": torch.zeros(2, 0, dtype=torch.long),
            "form_binds_y": torch.zeros(0),
            "form_binds_w": torch.zeros(0)}
    out = builder.build_context(cold)
    assert out[0].shape[0] == 0
    print("Cold-start: PASSED")
