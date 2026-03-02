import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertScorer(nn.Module):
    """
    A single 'Binding Theory' physics engine.
    Evaluates the compatibility between a refined protein and a query drug.
    """
    def __init__(self, protein_dim, drug_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        # Interaction features: [Prot, Drug, Prot*Drug, |Prot-Drug|]
        # Assumes protein_dim == drug_dim for the element-wise ops, 
        # or you can project them to the same dim first. 
        # Let's project them to a shared interaction space to be safe.
        self.p_proj = nn.Linear(protein_dim, hidden_dim)
        self.d_proj = nn.Linear(drug_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, z_p, z_d):
        # Project to shared space
        p = self.p_proj(z_p)  # [1, H] or [B, H]
        d = self.d_proj(z_d)  # [M, H] (M = number of query drugs)
        
        # Broadcast protein to match number of query drugs
        if p.size(0) == 1 and d.size(0) > 1:
            p = p.expand(d.size(0), -1)
            
        # Build interaction vector
        interaction = torch.cat([p, d, p * d, torch.abs(p - d)], dim=-1)
        return self.mlp(interaction).squeeze(-1) # [M]


class MultiplexMoEGate(nn.Module):
    """
    The Meta-Gate. Looks at the protein and the Trust signals to pick an expert.
    """
    def __init__(self, protein_dim, num_experts, hidden_dim=128):
        super().__init__()
        # Input: [Z_refined (protein_dim) + Trust_form (1) + Trust_role (1)]
        self.router = nn.Sequential(
            nn.Linear(protein_dim + 2, hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, z_refined, trust_form, trust_role):
        # z_refined: [1, protein_dim]
        # trust_form, trust_role: scalars converted to [1, 1] tensors
        gate_input = torch.cat([
            z_refined, 
            trust_form.view(1, 1), 
            trust_role.view(1, 1)
        ], dim=-1)
        
        logits = self.router(gate_input)
        return F.softmax(logits, dim=-1) # [1, num_experts]


class MultiplexRoutingHead(nn.Module):
    """
    Wraps the Trust Calculation, the Gate, and the Experts.
    """
    def __init__(self, protein_dim, drug_dim, num_experts=4):
        super().__init__()
        self.gate = MultiplexMoEGate(protein_dim, num_experts)
        self.experts = nn.ModuleList([
            ExpertScorer(protein_dim, drug_dim) for _ in range(num_experts)
        ])

    def _calculate_trust(self, footprints):
        """
        Calculates the reliability (inverse variance) of a multiplex floor.
        """
        num_neighbors = footprints.size(0)
        
        if num_neighbors < 2:
            # If 0 or 1 neighbor, we have no consensus. Trust is zero.
            return torch.tensor(0.0, device=footprints.device)
            
        # Calculate variance across the neighbors for each feature dimension
        # Shape: [drug_dim] -> mean() -> scalar
        variance = torch.var(footprints, dim=0, unbiased=False).mean()
        
        # Convert variance to a bounded Trust score between 0 and 1
        # Low variance -> High Trust (~1.0)
        # High variance -> Low Trust (~0.0)
        trust = 1.0 / (1.0 + variance)
        return trust

    def forward(self, z_refined, form_footprints, role_footprints, query_drug_features):
        """
        z_refined: [1, protein_dim] from Phase 2
        form_footprints: [N_form, drug_dim] the raw messages before Phase 2 attention
        role_footprints: [N_role, drug_dim] the raw messages before Phase 2 attention
        query_drug_features: [M, drug_dim] the drugs we are actually trying to rank
        """
        if z_refined.dim() == 1:
            z_refined = z_refined.unsqueeze(0)

        # 1. Audit the Floors (Calculate Trust)
        trust_form = self._calculate_trust(form_footprints)
        trust_role = self._calculate_trust(role_footprints)
        
        # 2. Route the Protein
        # gate_probs: [1, num_experts]
        gate_probs = self.gate(z_refined, trust_form, trust_role)
        
        # 3. Consult the Experts
        # Each expert ranks all M query drugs.
        # expert_predictions: List of tensors, each [M]
        expert_predictions = [expert(z_refined, query_drug_features) for expert in self.experts]
        
        # Stack into [M, num_experts]
        expert_tensor = torch.stack(expert_predictions, dim=1) 
        
        # 4. Final Prediction
        # Multiply each expert's prediction by the gate's confidence in that expert
        # gate_probs is [1, K], expert_tensor is [M, K]. 
        # Broadcasting handles the M query drugs automatically!
        final_scores = torch.sum(expert_tensor * gate_probs, dim=-1) # [M]
        
        # We also return gate_probs and expert_tensor for the Phase 5 EBL Loss!
        return final_scores, gate_probs, expert_tensor