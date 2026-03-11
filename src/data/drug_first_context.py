"""
Drug-first context builder for the GP-inspired affinity model.

Organizes revealed binding history around drugs, not protein neighborhoods:
  Level 1: D's exact binding profile  {(protX, affinity_X)}
  Level 2: D' analog profiles         {(protX, affinity_X × drug_sim(D, D'))}
  Level 3: no context (model falls back to BindingEncoder)

For each query drug D_i, we collect candidate (protein, scaled_affinity) pairs,
score them by protein_sim(protX, A) × drug_weight, and take the top-K.
The protein similarity pre-filter is a coarse ranking step; the drug-conditional
cross-attention in GPAffinityModel does the fine-grained relevance weighting.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


class DrugFirstContextBuilder:
    """
    Maintains a drug-indexed binding history and builds per-query-drug context.

    Usage:
        builder = DrugFirstContextBuilder(protein_features, drug_analog_index)
        # after each episode:
        builder.add_revealed(ep.edges, ep.labels)
        # at prediction time:
        ctx_proteins, ctx_affinities, ctx_mask = builder.build_context(
            qry_protein_idx, query_drug_indices, device
        )
    """

    def __init__(
        self,
        protein_features: torch.Tensor,  # [N_proteins, protein_dim]
        drug_analog_index=None,           # DrugAnalogIndex or None (for level 2)
        max_k: int = 32,
        analog_min_sim: float = 0.5,
    ):
        self.protein_features = protein_features
        self.drug_analog_index = drug_analog_index
        self.max_k = max_k
        self.analog_min_sim = analog_min_sim

        # drug_idx → [(prot_idx, affinity)]  maintained incrementally
        self._drug_to_bindings: dict[int, list[tuple[int, float]]] = defaultdict(list)

    def reset(self):
        self._drug_to_bindings.clear()

    def add_revealed(self, edges: torch.Tensor, labels: torch.Tensor):
        """
        Register newly revealed (protein, drug, affinity) bindings.

        edges:  [2, n_edges]  edges[0]=protein_idx, edges[1]=drug_idx
        labels: [n_edges]     affinity values
        """
        for i in range(edges.size(1)):
            prot_idx = int(edges[0, i])
            drug_idx = int(edges[1, i])
            aff = float(labels[i])
            self._drug_to_bindings[drug_idx].append((prot_idx, aff))

    def build_context(
        self,
        qry_protein_idx: int,
        query_drug_indices: torch.Tensor,  # [n_qry]
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build per-query-drug context tensors for the GP model.

        For each query drug D_i:
          Level 1: gather (protX, affinity_X) where drug == D_i  (weight=1.0)
          Level 2: gather (protX, affinity_X) from analog drugs D', scaled by
                   drug_sim(D_i, D')  (weight=drug_sim)
          Score candidates by cosine_sim(protX, A) × drug_weight; take top-K.

        Returns:
            ctx_proteins:   [n_qry, K, protein_dim]
            ctx_affinities: [n_qry, K]  (drug_sim-scaled at level 2)
            ctx_mask:       [n_qry, K] bool, True = valid slot
        """
        n_qry = query_drug_indices.size(0)
        K = self.max_k
        protein_dim = self.protein_features.size(1)

        ctx_proteins = torch.zeros(n_qry, K, protein_dim, device=device)
        ctx_affinities = torch.zeros(n_qry, K, device=device)
        ctx_mask = torch.zeros(n_qry, K, dtype=torch.bool, device=device)

        qry_prot_feat = self.protein_features[qry_protein_idx].to(device)

        for i, d_idx in enumerate(query_drug_indices.tolist()):
            d_idx = int(d_idx)
            # (prot_idx, scaled_affinity, drug_weight)
            candidates: list[tuple[int, float, float]] = []

            # Level 1: exact drug
            for prot_idx, aff in self._drug_to_bindings.get(d_idx, []):
                candidates.append((prot_idx, aff, 1.0))

            # Level 2: drug analogs, value-scaled by drug_sim
            if self.drug_analog_index is not None:
                sim_idx, sim_scores = self.drug_analog_index.get_analogs(d_idx)
                for j in range(sim_idx.size(0)):
                    sim = float(sim_scores[j])
                    if sim < self.analog_min_sim:
                        break  # sorted descending, safe to stop early
                    d_prime = int(sim_idx[j])
                    if d_prime == d_idx:
                        continue
                    for prot_idx, aff in self._drug_to_bindings.get(d_prime, []):
                        candidates.append((prot_idx, aff * sim, sim))

            if not candidates:
                continue  # level 3: mask stays False, model uses BindingEncoder

            # Score by protein_sim(protX, A) × drug_weight, take top-K
            prot_idxs = torch.tensor([c[0] for c in candidates], dtype=torch.long)
            scaled_affs = torch.tensor(
                [c[1] for c in candidates], dtype=torch.float32, device=device
            )
            drug_weights = torch.tensor(
                [c[2] for c in candidates], dtype=torch.float32, device=device
            )

            cand_feats = self.protein_features[prot_idxs].to(device)
            prot_sims = F.cosine_similarity(qry_prot_feat.unsqueeze(0), cand_feats, dim=-1)
            scores = (prot_sims + 1.0) / 2.0 * drug_weights  # both in [0,1]

            take = min(K, scores.size(0))
            top_idx = torch.topk(scores, take).indices

            ctx_proteins[i, :take] = cand_feats[top_idx]
            ctx_affinities[i, :take] = scaled_affs[top_idx]
            ctx_mask[i, :take] = True

        return ctx_proteins, ctx_affinities, ctx_mask
