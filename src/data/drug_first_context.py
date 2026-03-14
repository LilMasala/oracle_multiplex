"""
Drug-first context builder for the GP-inspired affinity model.

Retrieves context for each query (protein A, drug D) via a product kernel:

  score(A, D; X, Y) = k_prot(A, X) × k_drug(D, Y)

where:
  k_prot(A, X) = ESM cosine similarity, boosted to role_boost for GO-similar proteins
  k_drug(D, Y) = drug feature cosine similarity

Top-K scored (protein, drug, affinity) triples are returned as context. The
observation drug Y for each slot is passed through so GPAffinityModel can encode
the correct cross-pair kernel enc(X, Y) rather than enc(X, D_query).

This replaces the old level-1/2/3 hierarchy:
  - Exact (A, D) matches → k_drug = 1.0 → always top-scored
  - Drug analogs (A, D') → k_drug = drug_sim → scored proportionally
  - GO-similar proteins (X, D) → k_prot = go_sim (via role_boost) → correctly weighted
  - Cross-transfer (X, D') → k_prot × k_drug → both similarity axes apply

No value scaling (aff × drug_sim) is applied — the cross-pair kernel
enc(A,D)^T enc(X,Y) handles drug transfer weighting in the model.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


class DrugFirstContextBuilder:
    """
    Maintains a revealed binding history and builds per-query context via
    product-kernel retrieval.

    Usage:
        builder = DrugFirstContextBuilder(protein_features, drug_features)
        builder.add_revealed(ep.edges, ep.residuals)
        ctx_p, ctx_d, ctx_a, ctx_mask = builder.build_context(
            qry_protein_idx, query_drug_indices, device, pillar=pillar
        )
    """

    def __init__(
        self,
        protein_features: torch.Tensor,  # [N_proteins, protein_dim]
        drug_features: torch.Tensor,     # [N_drugs, drug_dim]
        max_k: int = 32,
        max_pool: int = 4096,  # cap on cross-pair pool (random-sampled when exceeded)
        # Legacy args kept for call-site compat — not used
        drug_analog_index=None,
        analog_min_sim: float = 0.5,
    ):
        self.protein_features = protein_features
        self.drug_features = drug_features
        self.max_k = max_k
        self.max_pool = max_pool

        # Exact drug-match lookup: drug_idx → [(prot_idx, residual_aff)]
        # Guarantees level-1 coverage even when random subsampling the pool.
        self._drug_to_bindings: dict[int, list[tuple[int, float]]] = defaultdict(list)

        # Flat lists for cross-pair pool (indexed together)
        self._pool_prot: list[int] = []
        self._pool_drug: list[int] = []
        self._pool_aff: list[float] = []

    def reset(self):
        self._drug_to_bindings.clear()
        self._pool_prot.clear()
        self._pool_drug.clear()
        self._pool_aff.clear()

    def add_revealed(self, edges: torch.Tensor, labels: torch.Tensor):
        """
        Register newly revealed (protein, drug, residual_affinity) bindings.

        edges:  [2, n_edges]  edges[0]=protein_idx, edges[1]=drug_idx
        labels: [n_edges]     residual affinities (label - prior) stored by caller
        """
        for i in range(edges.size(1)):
            prot_idx = int(edges[0, i])
            drug_idx = int(edges[1, i])
            aff = float(labels[i])
            self._drug_to_bindings[drug_idx].append((prot_idx, aff))
            self._pool_prot.append(prot_idx)
            self._pool_drug.append(drug_idx)
            self._pool_aff.append(aff)

    def build_context(
        self,
        qry_protein_idx: int,
        query_drug_indices: torch.Tensor,  # [n_qry]
        device: torch.device,
        pillar: dict | None = None,
        role_boost: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build per-query context tensors via product-kernel retrieval.

        Returns:
            ctx_proteins:   [n_qry, K, protein_dim]
            ctx_drugs:      [n_qry, K, drug_dim]      observation drug per context slot
            ctx_affinities: [n_qry, K]                residual affinities, unscaled
            ctx_mask:       [n_qry, K] bool            True = valid slot
        """
        n_qry = query_drug_indices.size(0)
        K = self.max_k
        protein_dim = self.protein_features.size(1)
        drug_dim = self.drug_features.size(1)
        N = len(self._pool_prot)

        ctx_proteins   = torch.zeros(n_qry, K, protein_dim, device=device)
        ctx_drugs      = torch.zeros(n_qry, K, drug_dim, device=device)
        ctx_affinities = torch.zeros(n_qry, K, device=device)
        ctx_mask       = torch.zeros(n_qry, K, dtype=torch.bool, device=device)

        if N == 0:
            return ctx_proteins, ctx_drugs, ctx_affinities, ctx_mask

        qry_prot_feat = self.protein_features[qry_protein_idx].to(device)

        # GO-similar protein set (from pillar's role layer)
        role_prot_set: set[int] = set()
        if pillar is not None:
            role_ei = pillar.get("role_binds_ei")
            if role_ei is not None and role_ei.size(1) > 0:
                role_prot_set = set(role_ei[0].tolist())

        # Build cross-pair pool (random subsample when N > max_pool)
        pool_prot_t = torch.tensor(self._pool_prot, dtype=torch.long)
        pool_drug_t = torch.tensor(self._pool_drug, dtype=torch.long)
        pool_aff_t  = torch.tensor(self._pool_aff,  dtype=torch.float32)

        if N > self.max_pool:
            perm = torch.randperm(N)[: self.max_pool]
            pool_prot_t = pool_prot_t[perm]
            pool_drug_t = pool_drug_t[perm]
            pool_aff_t  = pool_aff_t[perm]

        pool_prot_feats = self.protein_features[pool_prot_t].to(device)  # [pool, P]
        pool_drug_feats = self.drug_features[pool_drug_t].to(device)      # [pool, D]
        pool_aff_dev    = pool_aff_t.to(device)

        # k_prot: ESM cosine sim, boosted to role_boost for GO-similar proteins.
        # Precompute once — same for all query drugs.
        k_prot_esm = (
            F.cosine_similarity(qry_prot_feat.unsqueeze(0), pool_prot_feats, dim=-1) + 1.0
        ) / 2.0
        if role_prot_set:
            is_role = torch.tensor(
                [int(p) in role_prot_set for p in pool_prot_t.tolist()],
                dtype=torch.float32, device=device,
            )
            k_prot = torch.maximum(k_prot_esm, is_role * role_boost)
        else:
            k_prot = k_prot_esm

        for i, d_idx in enumerate(query_drug_indices.tolist()):
            d_idx = int(d_idx)

            # Always include exact drug matches (guarantees level-1 coverage even
            # when the cross-pair pool is randomly subsampled)
            exact = self._drug_to_bindings.get(d_idx, [])

            qry_drug_feat = self.drug_features[d_idx].to(device)
            k_drug = (
                F.cosine_similarity(qry_drug_feat.unsqueeze(0), pool_drug_feats, dim=-1) + 1.0
            ) / 2.0
            pool_scores = k_prot * k_drug

            if exact:
                ex_prot_t = torch.tensor([p for p, _ in exact], dtype=torch.long)
                ex_aff_t  = torch.tensor([a for _, a in exact], dtype=torch.float32, device=device)
                ex_prot_feats = self.protein_features[ex_prot_t].to(device)
                ex_drug_feat  = qry_drug_feat.unsqueeze(0).expand(len(exact), -1)

                ex_esm = (
                    F.cosine_similarity(qry_prot_feat.unsqueeze(0), ex_prot_feats, dim=-1) + 1.0
                ) / 2.0
                if role_prot_set:
                    ex_is_role = torch.tensor(
                        [int(p) in role_prot_set for p in ex_prot_t.tolist()],
                        dtype=torch.float32, device=device,
                    )
                    ex_k_prot = torch.maximum(ex_esm, ex_is_role * role_boost)
                else:
                    ex_k_prot = ex_esm
                # k_drug = 1.0 for exact matches
                ex_scores = ex_k_prot

                all_prot_feats = torch.cat([ex_prot_feats, pool_prot_feats], dim=0)
                all_drug_feats = torch.cat([ex_drug_feat,  pool_drug_feats], dim=0)
                all_affs       = torch.cat([ex_aff_t,      pool_aff_dev],    dim=0)
                all_scores     = torch.cat([ex_scores,     pool_scores],      dim=0)
            else:
                all_prot_feats = pool_prot_feats
                all_drug_feats = pool_drug_feats
                all_affs       = pool_aff_dev
                all_scores     = pool_scores

            take = min(K, all_scores.size(0))
            top_idx = torch.topk(all_scores, take).indices

            ctx_proteins[i, :take]   = all_prot_feats[top_idx]
            ctx_drugs[i, :take]      = all_drug_feats[top_idx]
            ctx_affinities[i, :take] = all_affs[top_idx]
            ctx_mask[i, :take] = True

        return ctx_proteins, ctx_drugs, ctx_affinities, ctx_mask

    def apply_neighborhood_fallback(self, *args, **kwargs):
        """Deprecated — product-kernel retrieval in build_context handles all sources."""
        pass
