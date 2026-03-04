import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np

from src.data.multiplex_loader import MultiplexPillarSampler
from src.models.smoothing import MultiplexInductiveSmoother
from src.models.routing import MultiplexRoutingHead
from src.models.multiplex_moe import MultiplexMoE
from src.training.ebl_loss import EBLLoss

def generate_random_noise_graph(num_proteins=50, num_drugs=200, prot_dim=128, drug_dim=128, device='cpu'):
    """Generates a graph where features and labels are pure random noise."""
    print("🎲 Generating Random Noise Graph...")
    data = HeteroData()
    # Random normal features
    data["protein"].x = torch.randn(num_proteins, prot_dim)
    data["drug"].x = torch.randn(num_drugs, drug_dim)

    # Random topology: Form edges (Structural similarities)
    form_src = torch.randint(0, num_proteins, (100,))
    form_dst = torch.randint(0, num_proteins, (100,))
    data["protein", "similar", "protein"].edge_index = torch.stack([form_src, form_dst])

    # Random topology: Role edges (Shared GO Terms)
    role_src = torch.randint(0, num_proteins, (150,))
    role_dst = torch.randint(0, num_proteins, (150,))
    data["protein", "go_shared", "protein"].edge_index = torch.stack([role_src, role_dst])

    # Random Binding Edges (We generate a lot so we can split them)
    num_binds = 800
    binds_src = torch.randint(0, num_proteins, (num_binds,))
    binds_dst = torch.randint(0, num_drugs, (num_binds,))
    data["protein", "binds_pic50", "drug"].edge_index = torch.stack([binds_src, binds_dst])
    
    # Random pIC50 labels (Mean 6.0, Std 1.5)
    data["protein", "binds_pic50", "drug"].edge_label = torch.randn(num_binds) * 1.5 + 6.0
    
    return data.to(device)

def run_noise_test():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 Running on device: {device}")

    # 1. Setup Data & Architecture
    prot_dim, drug_dim, num_experts = 128, 128, 4
    data = generate_random_noise_graph(prot_dim=prot_dim, drug_dim=drug_dim, device=device)
    
    # Notice we initialize the loader with NO binding edges at first (Strict Timeline!)
    # We will reveal them prequentially.
    empty_binds_data = data.clone()
    empty_binds_data["protein", "binds_pic50", "drug"].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    empty_binds_data["protein", "binds_pic50", "drug"].edge_label = torch.empty((0,), dtype=torch.float, device=device)
    
    loader = MultiplexPillarSampler(empty_binds_data)
    
    smoother = MultiplexInductiveSmoother(prot_dim, drug_dim).to(device)
    router = MultiplexRoutingHead(prot_dim, drug_dim, num_experts).to(device)
    model = MultiplexMoE(smoother, router).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = EBLLoss(ebl_alpha=0.1)

    # 2. Extract episodes (Simulating src/protocol/prequential.py)
    binds_ei = data["protein", "binds_pic50", "drug"].edge_index
    binds_y = data["protein", "binds_pic50", "drug"].edge_label
    
    print("\n⚔️ ENTERING THE ARENA (Prequential Training Loop) ⚔️")
    print("Expectation: Support Loss might wiggle. Query Loss should stay stubbornly high (~2.0 - 2.5).")
    print("-" * 75)
    
    for ep_num in range(30): # Run for 30 proteins
        prot_idx = ep_num
        
        # Get all true edges for this protein from the FULL graph
        mask = binds_ei[0] == prot_idx
        p_edges = binds_ei[:, mask]
        p_labels = binds_y[mask]
        
        # If this protein has fewer than 4 edges, skip it (can't split effectively)
        if p_edges.size(1) < 4:
            continue
            
        # Split into Query (Blind Test) and Support (Training)
        split_idx = p_edges.size(1) // 2
        query_edges, query_labels = p_edges[:, :split_idx], p_labels[:split_idx]
        support_edges, support_labels = p_edges[:, split_idx:], p_labels[split_idx:]

        optimizer.zero_grad()
        
        # --- ZERO-SHOT EVALUATION (THE BLINDFOLD) ---
        pillar = loader.get_pillar_context(prot_idx)
        with torch.no_grad():
            query_preds, _, _, _ = model(pillar, data["drug"].x, query_edges[1])
            query_mse = F.mse_loss(query_preds, query_labels).item()

        # --- SUPPORT TRAINING (THE AUTOPSY) ---
        support_preds, gate_probs, expert_tensor, _ = model(pillar, data["drug"].x, support_edges[1])
        losses = loss_fn(support_preds, support_labels, gate_probs, expert_tensor)

        losses["total_loss"].backward()
        # Gradient clipping because MoE gates can get unstable on pure noise
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # --- MEMORY BANK EXPANSION ---
        loader.add_revealed_edges(support_edges, support_labels)

        print(
            f"Prot {prot_idx:02d} | Query MSE (Blind): {query_mse:.4f} "
            f"| Supp Expert Loss: {losses['expert_loss'].item():.4f} "
            f"| Gate Pen: {losses['gate_loss'].item():.4f}"
        )

if __name__ == "__main__":
    run_noise_test()