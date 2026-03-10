"""
TNP context builder with cold-start improvements:
  Unit 2 — Synthetic prior context token (cold-start)
  Unit 5 — Drug analog context injection (sparse context)
  Unit 6 — GNN embedding collection (aligned with context tokens)
"""
import torch
from typing import Optional


class TNPContextBuilder:
    def __init__(
        self,
        drug_features: torch.Tensor,      # [N_drugs, drug_dim]
        max_context: int = 256,
        min_affinity_weight: float = 0.0,
        # Unit 2: Synthetic prior
        global_drug_mean: Optional[torch.Tensor] = None,
        global_mean_affinity: float = 6.5,
        enable_synthetic_prior: bool = False,
        # Unit 5: Drug analog injection
        drug_analog_index=None,             # DrugAnalogIndex or None
        analog_inject_threshold: int = 8,   # inject when n_ctx < this
        analog_sim_threshold: float = 0.7,
        max_query_for_analog: int = 20,     # cap to avoid slowdown
        # Unit 6: GNN embeddings
        gnn_protein_embs: Optional[torch.Tensor] = None,  # [N_proteins, gnn_dim]
    ):
        self.drug_features = drug_features
        self.max_context = max_context
        self.min_affinity_weight = min_affinity_weight

        # Unit 2
        self.global_drug_mean = global_drug_mean
        self.global_mean_affinity = global_mean_affinity
        self.enable_synthetic_prior = enable_synthetic_prior

        # Unit 5
        self.drug_analog_index = drug_analog_index
        self.analog_inject_threshold = analog_inject_threshold
        self.analog_sim_threshold = analog_sim_threshold
        self.max_query_for_analog = max_query_for_analog

        # Unit 6
        self.gnn_protein_embs = gnn_protein_embs

    def _collect_layer(
        self,
        neighbor_indices: torch.Tensor,
        neighbor_features: torch.Tensor,
        diff_w: torch.Tensor,
        binds_ei: torch.Tensor,
        binds_y: torch.Tensor,
        binds_w: torch.Tensor,
    ):
        """
        Extract (protein, drug, affinity, ppr, trust, drug_idx, gnn_emb) tuples
        from one multiplex layer.

        Returns 7 lists: prot_list, drug_list, aff_list, ppr_list,
                          trust_list, didx_list, gnn_list
        """
        if binds_ei.size(1) == 0 or neighbor_indices.numel() == 0:
            return [], [], [], [], [], [], []

        keep = binds_w >= self.min_affinity_weight
        if not keep.any():
            return [], [], [], [], [], [], []

        ei = binds_ei[:, keep]
        y  = binds_y[keep]
        w  = binds_w[keep]
        drug_idx = ei[1]
        prot_idx = ei[0]

        valid = drug_idx < self.drug_features.size(0)
        if not valid.any():
            return [], [], [], [], [], [], []
        ei, y, w = ei[:, valid], y[valid], w[valid]
        drug_idx, prot_idx = drug_idx[valid], prot_idx[valid]

        prot_list, drug_list, aff_list = [], [], []
        ppr_list, trust_list, didx_list, gnn_list = [], [], [], []

        for k, nidx in enumerate(neighbor_indices.tolist()):
            mask = prot_idx == nidx
            if not mask.any():
                continue
            n_edges = int(mask.sum())
            prot_list.append(neighbor_features[k].unsqueeze(0).expand(n_edges, -1))
            drug_list.append(self.drug_features[drug_idx[mask]])
            aff_list.append(y[mask].unsqueeze(1))
            didx_list.append(drug_idx[mask])

            ppr_val = diff_w[k].item() if diff_w.numel() > k else 1.0
            ppr_list.append(torch.full((n_edges,), ppr_val, device=binds_ei.device))
            trust_list.append(w[mask])

            # Unit 6: GNN embeddings aligned with context tokens
            if self.gnn_protein_embs is not None:
                gnn_emb = self.gnn_protein_embs[nidx].unsqueeze(0).expand(n_edges, -1)
                gnn_list.append(gnn_emb)

        return prot_list, drug_list, aff_list, ppr_list, trust_list, didx_list, gnn_list

    def _inject_drug_analogs(
        self,
        query_drug_indices: torch.Tensor,
        ctx_prot_cat: torch.Tensor,
        ctx_drug_cat: torch.Tensor,
        ctx_aff_cat: torch.Tensor,
        ctx_ppr_cat: torch.Tensor,
        ctx_trust_cat: torch.Tensor,
        ctx_didx_cat: torch.Tensor,
        ctx_gnn_cat: Optional[torch.Tensor],
        prot_parts: list,
        drug_parts: list,
        aff_parts: list,
        ppr_parts: list,
        trust_parts: list,
        gnn_parts: list,
    ):
        """
        Unit 5: For each query drug, find chemically similar drugs that have
        known bindings with neighbor proteins. Inject pseudo-context tokens:
            (neighbor_protein_features, query_drug_features, analog_affinity,
             original_ppr * drug_sim, original_trust * drug_sim)
        """
        device = ctx_prot_cat.device
        q_list = query_drug_indices.tolist()
        if len(q_list) > self.max_query_for_analog:
            perm = torch.randperm(len(q_list), device=device)[:self.max_query_for_analog]
            q_list = [q_list[i] for i in perm.tolist()]

        for q_idx in q_list:
            sim_idx, sim_scores = self.drug_analog_index.get_analogs(q_idx)
            for j in range(sim_idx.size(0)):
                sim_j = float(sim_scores[j])
                if sim_j < self.analog_sim_threshold:
                    break  # sorted descending, safe to stop early
                didx_j = int(sim_idx[j])
                ctx_mask = ctx_didx_cat == didx_j
                if not ctx_mask.any():
                    continue
                n_m = int(ctx_mask.sum())

                # Pseudo-token: same protein/affinity as analog binding,
                # but with the QUERY drug's features and analog-scaled weights.
                prot_parts.append(ctx_prot_cat[ctx_mask])
                drug_parts.append(
                    self.drug_features[q_idx].unsqueeze(0).expand(n_m, -1).to(device)
                )
                aff_parts.append(ctx_aff_cat[ctx_mask])
                ppr_parts.append(ctx_ppr_cat[ctx_mask] * sim_j)
                trust_parts.append(ctx_trust_cat[ctx_mask] * sim_j)
                if ctx_gnn_cat is not None:
                    gnn_parts.append(ctx_gnn_cat[ctx_mask])

    def build_context(
        self,
        pillar: dict,
        query_drug_indices: Optional[torch.Tensor] = None,
    ):
        """
        Build TNP context set from a pillar dict.

        Args:
            pillar: dict from MultiplexPillarSampler.get_pillar_context()
            query_drug_indices: [N_qry] drug indices being scored (for Unit 5)

        Returns 6-tuple:
            ctx_protein:  [N_ctx, protein_dim]
            ctx_drug:     [N_ctx, drug_dim]
            ctx_affinity: [N_ctx, 1]
            ctx_ppr:      [N_ctx]
            ctx_trust:    [N_ctx]
            ctx_gnn_emb:  [N_ctx, gnn_dim] or None
        If no context, returns zero-size tensors (+ synthetic prior if enabled).
        """
        device = pillar["target_features"].device
        protein_dim = pillar["target_features"].size(0)
        drug_dim = self.drug_features.size(1)

        prot_parts, drug_parts, aff_parts = [], [], []
        ppr_parts, trust_parts, didx_parts, gnn_parts = [], [], [], []

        for layer in ("form", "role"):
            neighbors = pillar[f"{layer}_neighbors"]
            features  = pillar[f"{layer}_features"]
            diff_w    = pillar[f"{layer}_diff_w"]
            ei        = pillar[f"{layer}_binds_ei"]
            y         = pillar[f"{layer}_binds_y"]
            w         = pillar[f"{layer}_binds_w"]
            p, d, a, ppr, trust, didx, gnn = self._collect_layer(
                neighbors, features, diff_w, ei, y, w
            )
            prot_parts.extend(p)
            drug_parts.extend(d)
            aff_parts.extend(a)
            ppr_parts.extend(ppr)
            trust_parts.extend(trust)
            didx_parts.extend(didx)
            gnn_parts.extend(gnn)

        # Unit 5: Drug analog injection (only when sparse but not cold)
        if (
            query_drug_indices is not None
            and self.drug_analog_index is not None
            and prot_parts
        ):
            n_ctx_so_far = sum(p.size(0) for p in prot_parts)
            if n_ctx_so_far < self.analog_inject_threshold:
                ctx_prot_cat  = torch.cat(prot_parts,  dim=0)
                ctx_drug_cat  = torch.cat(drug_parts,  dim=0)
                ctx_aff_cat   = torch.cat(aff_parts,   dim=0)
                ctx_ppr_cat   = torch.cat(ppr_parts,   dim=0)
                ctx_trust_cat = torch.cat(trust_parts, dim=0)
                ctx_didx_cat  = torch.cat(didx_parts,  dim=0)
                ctx_gnn_cat   = torch.cat(gnn_parts, dim=0) if gnn_parts else None

                self._inject_drug_analogs(
                    query_drug_indices,
                    ctx_prot_cat, ctx_drug_cat, ctx_aff_cat,
                    ctx_ppr_cat, ctx_trust_cat, ctx_didx_cat, ctx_gnn_cat,
                    prot_parts, drug_parts, aff_parts,
                    ppr_parts, trust_parts, gnn_parts,
                )

        # Cold-start: still no context after possible analog injection
        if not prot_parts:
            # Unit 2: inject synthetic prior token to activate cross-attention path
            if (
                self.enable_synthetic_prior
                and self.global_drug_mean is not None
                and pillar.get("ppr_centroid") is not None
            ):
                ppr_centroid = pillar["ppr_centroid"].to(device)
                syn_protein  = ppr_centroid.unsqueeze(0)                     # [1, protein_dim]
                syn_drug     = self.global_drug_mean.to(device).unsqueeze(0) # [1, drug_dim]
                syn_affinity = torch.full((1, 1), self.global_mean_affinity, device=device)
                syn_ppr      = torch.full((1,), 0.5, device=device)
                syn_trust    = torch.full((1,), 0.3, device=device)
                syn_gnn      = None
                if self.gnn_protein_embs is not None:
                    syn_gnn = self.gnn_protein_embs.mean(0, keepdim=True).to(device)
                return syn_protein, syn_drug, syn_affinity, syn_ppr, syn_trust, syn_gnn

            return (
                torch.zeros(0, protein_dim, device=device),
                torch.zeros(0, drug_dim, device=device),
                torch.zeros(0, 1, device=device),
                torch.zeros(0, device=device),
                torch.zeros(0, device=device),
                None,
            )

        ctx_protein  = torch.cat(prot_parts, dim=0)
        ctx_drug     = torch.cat(drug_parts, dim=0).to(device)
        ctx_affinity = torch.cat(aff_parts,  dim=0)
        ctx_ppr      = torch.cat(ppr_parts,  dim=0)
        ctx_trust    = torch.cat(trust_parts, dim=0)
        ctx_gnn_emb  = torch.cat(gnn_parts, dim=0) if gnn_parts else None

        N = ctx_protein.size(0)
        if N > self.max_context:
            weights = ctx_ppr.clamp(min=1e-8)
            idx = torch.multinomial(weights, self.max_context, replacement=False)
            ctx_protein  = ctx_protein[idx]
            ctx_drug     = ctx_drug[idx]
            ctx_affinity = ctx_affinity[idx]
            ctx_ppr      = ctx_ppr[idx]
            ctx_trust    = ctx_trust[idx]
            if ctx_gnn_emb is not None:
                ctx_gnn_emb = ctx_gnn_emb[idx]

        return ctx_protein, ctx_drug, ctx_affinity, ctx_ppr, ctx_trust, ctx_gnn_emb


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

    ctx_p, ctx_d, ctx_a, ctx_ppr, ctx_trust, ctx_gnn = builder.build_context(pillar)
    assert ctx_p.shape[1] == protein_dim
    assert ctx_d.shape[1] == 512
    assert ctx_a.shape[1] == 1
    N = ctx_p.shape[0]
    assert ctx_ppr.shape == (N,)
    assert ctx_trust.shape == (N,)
    assert ctx_p.shape[0] == ctx_d.shape[0] == ctx_a.shape[0]
    assert ctx_gnn is None
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
    print("Cold-start (no prior): PASSED")

    # Test Unit 2: synthetic prior
    builder2 = TNPContextBuilder(
        drug_features, max_context=64,
        global_drug_mean=drug_features.mean(0),
        global_mean_affinity=6.5,
        enable_synthetic_prior=True,
    )
    out2 = builder2.build_context(cold)
    assert out2[0].shape[0] == 1, f"Expected 1 synthetic token, got {out2[0].shape[0]}"
    print("Cold-start (synthetic prior): PASSED")
