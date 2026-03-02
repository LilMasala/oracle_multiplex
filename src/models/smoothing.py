import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiplexInductiveSmoother(nn.Module):
    def __init__(self, protein_dim, drug_dim, baseline_pic50=6.0):
        super().__init__()
        self.baseline = baseline_pic50
        self.drug_dim = drug_dim
        
        # Refiner to allow positioning to shift even without neighbors
        self.prot_refiner = nn.Sequential(
            nn.Linear(protein_dim, protein_dim),
            nn.PReLU(),
            nn.Linear(protein_dim, protein_dim)
        )
        
        # --- NEW: CROSS-ATTENTION DICTIONARY SCANNERS ---
        # Query: The target protein looking for relevant chemical features
        self.q_proj = nn.Linear(protein_dim, drug_dim)
        # Key: The drug features in the neighbor's history
        self.k_proj = nn.Linear(drug_dim, drug_dim)
        # Value: What we actually extract (drug features scaled by affinity)
        self.v_proj = nn.Linear(drug_dim, drug_dim)
        
        self.layer_emb = nn.Embedding(2, 16) 
        
        # Deeper attn_net for complex similarities (The Floor Filter)
        attn_in_dim = (protein_dim * 2) + 16
        self.attn_net = nn.Sequential(
            nn.Linear(attn_in_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
        self.integration_mlp = nn.Sequential(
            nn.Linear(drug_dim, protein_dim),
            nn.PReLU(),
            nn.Linear(protein_dim, protein_dim)
        )
        self.norm = nn.LayerNorm(protein_dim)

    def _build_preference_vectors(self, z_target, neighbor_ids, binds_ei, binds_y, drug_features):
        """
        The Dictionary Scan: Uses Cross-Attention so the target protein can 
        selectively pull the most relevant drug features from a neighbor's history.
        """
        device = drug_features.device
        num_neighbors = neighbor_ids.size(0)
        messages = torch.zeros((num_neighbors, self.drug_dim), device=device)
        
        if binds_ei.numel() == 0:
            return messages 
            
        centered_y = binds_y - self.baseline
        
        # 1. Project the Target Protein into the search space (The Query)
        # z_target is [protein_dim] -> Q is [1, drug_dim]
        Q = self.q_proj(z_target).unsqueeze(0) 
        
        for i, n_idx in enumerate(neighbor_ids):
            mask = binds_ei[0] == n_idx
            if mask.any():
                drug_idx = binds_ei[1][mask]
                affinities = centered_y[mask]
                d_feats = drug_features[drug_idx] # [N_drugs, drug_dim]
                
                # 2. Project the neighbor's drugs (The Keys and Values)
                K = self.k_proj(d_feats) # [N_drugs, drug_dim]
                
                # We multiply the Values by the affinity so the model knows 
                # if it was a strong bind or a strong rejection.
                V = self.v_proj(d_feats) * affinities.unsqueeze(-1) # [N_drugs, drug_dim]
                
                # 3. Calculate "Dictionary" Match Scores (Dot Product Attention)
                # How well does the target protein match each drug the neighbor saw?
                scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.drug_dim) # [1, N_drugs]
                
                # Convert scores to probabilities (The "OH SHOOT LIKE THIS!" weights)
                attn_weights = F.softmax(scores, dim=-1) # [1, N_drugs]
                
                # 4. Pull the relevant information
                # Instead of a dumb average, we do a weighted sum based on relevance
                messages[i] = torch.matmul(attn_weights, V).squeeze(0)
                
        return messages

    def _compute_attention(self, z_target, z_neighbors, layer_id):
        num_neighbors = z_neighbors.size(0)
        if num_neighbors == 0:
            return torch.empty((0, 1), device=z_target.device)
            
        z_target_exp = z_target.unsqueeze(0).expand(num_neighbors, -1)
        l_emb = self.layer_emb(torch.tensor([layer_id], device=z_target.device)).expand(num_neighbors, -1)
        
        attn_input = torch.cat([z_target_exp, z_neighbors, l_emb], dim=-1)
        raw_scores = self.attn_net(attn_input)
        return raw_scores

    def forward(self, pillar_data, drug_features):
        # 1. Align the latent spaces!
        z_target_refined = self.prot_refiner(pillar_data["target_features"])
        form_feats_refined = self.prot_refiner(pillar_data["form_features"])
        role_feats_refined = self.prot_refiner(pillar_data["role_features"])
        
        # 2. Run the Dictionary Scan (Notice we now pass z_target_refined)
        form_msgs = self._build_preference_vectors(
            z_target_refined, pillar_data["form_neighbors"], 
            pillar_data["form_binds_ei"], pillar_data["form_binds_y"], drug_features
        )
        role_msgs = self._build_preference_vectors(
            z_target_refined, pillar_data["role_neighbors"], 
            pillar_data["role_binds_ei"], pillar_data["role_binds_y"], drug_features
        )
        
        # 3. Apply the Global Floor Filter
        form_attn = self._compute_attention(z_target_refined, form_feats_refined, layer_id=0)
        role_attn = self._compute_attention(z_target_refined, role_feats_refined, layer_id=1)
        
        all_msgs = torch.cat([form_msgs, role_msgs], dim=0)
        all_attn = torch.cat([form_attn, role_attn], dim=0)
        
        if all_msgs.size(0) == 0:
            return self.norm(z_target_refined), form_msgs, role_msgs
            
        attn_weights = F.softmax(all_attn, dim=0)
        v_prior = torch.sum(attn_weights * all_msgs, dim=0)
        
        # 4. Final Inductive Mix
        z_refined = self.norm(z_target_refined + self.integration_mlp(v_prior))
        
        return z_refined, form_msgs, role_msgs