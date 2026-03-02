import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiplexInductiveSmoother(nn.Module):
    def __init__(self, protein_dim, drug_dim, baseline_pic50=6.0):
        super().__init__()
        self.baseline = baseline_pic50
        self.drug_dim = drug_dim
        
        # Embeddings to tell the attention network WHICH floor we are on
        # 0 = Form (CATH/ESM), 1 = Role (GO)
        self.layer_emb = nn.Embedding(2, 16) 
        
        # The "Commonality Filter" (Attention network)
        # Input: Target Prot + Neighbor Prot + Floor Embedding
        attn_in_dim = (protein_dim * 2) + 16
        self.attn_net = nn.Sequential(
            nn.Linear(attn_in_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
        # Translates the aggregated chemical footprint back into protein latent space
        self.integration_mlp = nn.Sequential(
            nn.Linear(drug_dim, protein_dim),
            nn.PReLU(),
            nn.Linear(protein_dim, protein_dim)
        )
        self.norm = nn.LayerNorm(protein_dim)

    def _build_preference_vectors(self, neighbor_ids, binds_ei, binds_y, drug_features):
        """
        Translates raw pIC50 edges into dense chemical footprints (Binding Preference Vectors).
        """
        device = drug_features.device
        num_neighbors = neighbor_ids.size(0)
        messages = torch.zeros((num_neighbors, self.drug_dim), device=device)
        
        if binds_ei.numel() == 0:
            return messages # Return empty signals if no neighbors have labels yet
            
        # Shift affinities: >6.0 is positive (binds), <6.0 is negative (rejects)
        centered_y = binds_y - self.baseline
        
        # For each neighbor, aggregate the chemical features of the drugs it interacted with
        for i, n_idx in enumerate(neighbor_ids):
            mask = binds_ei[0] == n_idx
            if mask.any():
                drug_idx = binds_ei[1][mask]
                affinities = centered_y[mask]
                
                # Fetch MolCLR features for these specific drugs
                d_feats = drug_features[drug_idx]
                
                # Weight drug features by affinity and sum into a single vector
                messages[i] = torch.sum(d_feats * affinities.unsqueeze(-1), dim=0)
                
        return messages

    def _compute_attention(self, z_target, z_neighbors, layer_id):
        """
        The Commonality Filter: Figures out WHICH neighbor's features actually matter.
        """
        num_neighbors = z_neighbors.size(0)
        if num_neighbors == 0:
            return torch.empty((0, 1), device=z_target.device)
            
        # Expand target features to match neighbor count
        z_target_exp = z_target.unsqueeze(0).expand(num_neighbors, -1)
        
        # Get the floor embedding (Form or Role)
        l_emb = self.layer_emb(torch.tensor([layer_id], device=z_target.device)).expand(num_neighbors, -1)
        
        # Concatenate: [Target || Neighbor || Floor]
        attn_input = torch.cat([z_target_exp, z_neighbors, l_emb], dim=-1)
        
        # Compute raw attention scores (the e_{i,j})
        raw_scores = self.attn_net(attn_input)
        return raw_scores

    def forward(self, pillar_data, drug_features):
        """
        pillar_data: The dictionary output from Phase 1 (MultiplexPillarSampler)
        drug_features: The global matrix of MolCLR drug embeddings [num_total_drugs, drug_dim]
        """
        z_target = pillar_data["target_features"]
        
        # 1. Build the Chemical Footprints (The Messages)
        form_msgs = self._build_preference_vectors(
            pillar_data["form_neighbors"], pillar_data["form_binds_ei"], 
            pillar_data["form_binds_y"], drug_features
        )
        role_msgs = self._build_preference_vectors(
            pillar_data["role_neighbors"], pillar_data["role_binds_ei"], 
            pillar_data["role_binds_y"], drug_features
        )
        
        # 2. Apply the Commonality Filter (Attention)
        # layer_id 0 = Form, 1 = Role
        form_attn = self._compute_attention(z_target, pillar_data["form_features"], layer_id=0)
        role_attn = self._compute_attention(z_target, pillar_data["role_features"], layer_id=1)
        
        # Combine all attention scores to run a global Softmax across BOTH floors
        all_msgs = torch.cat([form_msgs, role_msgs], dim=0)
        all_attn = torch.cat([form_attn, role_attn], dim=0)
        
        if all_msgs.size(0) == 0:
            return z_target, form_msgs, role_msgs # form and role msgs are already empty tensors!
            
        attn_weights = F.softmax(all_attn, dim=0)
        
        # 3. The Inductive Leak
        # Weight the chemical footprints by the attention and sum them up
        v_prior = torch.sum(attn_weights * all_msgs, dim=0)
        
        # 4. Refine the Identity
        # Mix the raw protein features with the aggregated chemical prior
        z_refined = self.norm(z_target + self.integration_mlp(v_prior))
        
        return z_refined, form_msgs, role_msgs