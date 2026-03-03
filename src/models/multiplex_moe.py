import torch
import torch.nn as nn


class MultiplexMoE(nn.Module):
    def __init__(self, smoother, router):
        super().__init__()
        self.smoother = smoother
        self.router = router

    def forward(self, pillar_data, drug_features, target_drug_indices):
        z_refined, v_prior, floor_stats = self.smoother(pillar_data, drug_features)
        target_drug_feats = drug_features[target_drug_indices]

        trust_vector = pillar_data.get("trust_vector", torch.zeros(3, device=z_refined.device))
        scores, gate_probs, expert_tensor = self.router(
            z_refined=z_refined,
            protein_raw_features=pillar_data["target_features"],
            v_prior=v_prior,
            query_drug_features=target_drug_feats,
            trust_vector=trust_vector,
        )
        return scores, gate_probs, expert_tensor
