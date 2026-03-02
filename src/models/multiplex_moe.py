import torch
import torch.nn as nn


class MultiplexMoE(nn.Module):
    def __init__(self, smoother, router):
        super().__init__()
        self.smoother = smoother
        self.router = router

    def forward(self, pillar_data, drug_features, target_drug_indices):
        z_refined, form_footprints, role_footprints, floor_stats = self.smoother(pillar_data, drug_features)
        target_drug_feats = drug_features[target_drug_indices]
        scores, gate_probs, expert_tensor = self.router(
            z_refined,
            form_footprints,
            role_footprints,
            target_drug_feats,
            floor_stats=floor_stats,
            cross_floor_jaccard=pillar_data.get("cross_floor_jaccard", torch.tensor(0.0, device=z_refined.device)),
        )
        return scores, gate_probs, expert_tensor
