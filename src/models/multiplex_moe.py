import torch
import torch.nn as nn

class MultiplexMoE(nn.Module):
    def __init__(self, smoother, router):
        super().__init__()
        self.smoother = smoother
        self.router = router

    def forward(self, pillar_data, drug_features, target_drug_indices):
        # 1. Phase 2: Get the Refined Identity AND the raw footprints
        z_refined, form_footprints, role_footprints = self.smoother(pillar_data, drug_features)
        
        # 2. Extract the features for the specific drugs we are ranking
        target_drug_feats = drug_features[target_drug_indices]
        
        # 3. Phase 3: Route and Score
        scores, gate_probs, expert_tensor = self.router(
            z_refined, form_footprints, role_footprints, target_drug_feats
        )
        return scores, gate_probs, expert_tensor