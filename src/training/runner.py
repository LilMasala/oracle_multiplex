import torch
import torch.nn.functional as F
import numpy as np

def run_prequential_stream(model, episodes, loader, drug_features, optimizer, config):
    """
    Executes the linear timeline. 
    model: The MultiplexMoE wrapper.
    episodes: List of ProteinEpisode objects from Phase 1.
    loader: MultiplexPillarSampler from Phase 1.
    """
    model.train()
    
    # Tracking metrics over the stream
    stream_results = {"ci": [], "mse": []}
    
    for ep_num, ep in enumerate(episodes):
        optimizer.zero_grad()
        
        # ==========================================
        # STEP 1: The Arrival & The Consultation
        # Fetch the neighborhood context for this protein.
        # ==========================================
        pillar = loader.get_pillar_context(ep.protein_idx)
        
        # ==========================================
        # STEP 2: The Moment of Truth (Zero-Shot Eval)
        # We test the model on the query edges blindly.
        # ==========================================
        with torch.no_grad():
            query_drug_indices = ep.query_edges[1]
            query_preds, _, _ = model(pillar, drug_features, query_drug_indices)
            
            # Record the true cold-start performance
            mse_val = F.mse_loss(query_preds, ep.query_labels).item()
            stream_results["mse"].append(mse_val)
            
            # TODO: Add your Concordance Index (CI) calculation here
            # ci_val = calculate_ci(query_preds, ep.query_labels)
            # stream_results["ci"].append(ci_val)
            
            print(f"Episode {ep_num} | Prot {ep.protein_idx} | Query MSE: {mse_val:.4f}")

        # ==========================================
        # STEP 3: The Autopsy & Policy Update (Support Training)
        # The blindfold comes off. We learn from the support edges.
        # ==========================================
        support_drug_indices = ep.support_edges[1]
        support_preds, gate_probs, expert_tensor = model(pillar, drug_features, support_drug_indices)
        
        # Base regression loss
        base_loss = F.mse_loss(support_preds, ep.support_labels)
        
        # --- PHASE 5 PREVIEW: The EBL Penalty ---
        # Did the Gate route to a stupid expert? Let's find out.
        with torch.no_grad():
            # Which expert actually had the lowest error for these labels?
            per_expert_err = (expert_tensor - ep.support_labels.unsqueeze(-1)) ** 2
            target_gate_probs = F.softmax(-per_expert_err, dim=-1) # Ideal routing distribution
            
        # Cross-Entropy between what the gate chose and what it SHOULD have chosen
        gate_loss = -torch.sum(target_gate_probs * torch.log(gate_probs + 1e-12), dim=-1).mean()
        
        # Total Loss
        loss = base_loss + (config.get('ebl_alpha', 0.1) * gate_loss)
        loss.backward()
        optimizer.step()
        
        # ==========================================
        # STEP 4: Expand the Memory Bank (Living Graph Update)
        # ==========================================
        # Now that we've processed this protein, its support edges become 
        # PUBLIC KNOWLEDGE for the next proteins in the stream.
        loader.add_revealed_edges(ep.support_edges, ep.support_labels)
        
    return stream_results