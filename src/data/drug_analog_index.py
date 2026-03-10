"""Unit 5: Precomputed drug-drug cosine similarity top-K index for analog context injection."""
import torch
import torch.nn.functional as F


class DrugAnalogIndex:
    """
    Precomputes and stores top-K cosine-similar drugs for each drug.

    Usage:
        index = DrugAnalogIndex(drug_features, top_k=32)
        similar_idx, sim_scores = index.get_analogs(query_drug_idx)
    """

    def __init__(self, drug_features: torch.Tensor, top_k: int = 32, batch_size: int = 512):
        """
        Args:
            drug_features: [N_drugs, drug_dim] float tensor
            top_k: number of similar drugs to store per drug (excluding self)
            batch_size: batch size for similarity computation (to avoid OOM)
        """
        self.top_k = top_k
        N = drug_features.size(0)
        device = drug_features.device
        k = min(top_k, N - 1)

        # Normalize for cosine similarity
        normed = F.normalize(drug_features.float(), dim=1)  # [N, D]

        all_indices = torch.zeros(N, k, dtype=torch.long, device=device)
        all_scores = torch.zeros(N, k, dtype=torch.float, device=device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = normed[start:end]                      # [B, D]
            sims = torch.mm(batch, normed.t())             # [B, N]
            # Zero out self-similarity to avoid selecting self
            for j in range(end - start):
                sims[j, start + j] = -2.0
            vals, idx = sims.topk(k, dim=1)               # [B, k]
            all_indices[start:end] = idx
            all_scores[start:end] = vals

        self.indices = all_indices  # [N, top_k]
        self.scores = all_scores    # [N, top_k]

    def get_analogs(self, drug_idx: int):
        """Returns (similar_drug_indices [top_k], similarity_scores [top_k])."""
        return self.indices[drug_idx], self.scores[drug_idx]
