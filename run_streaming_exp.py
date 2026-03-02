import torch
import os 
import numpy as np
from torch.optim import Adam
from src.data.multiplex_loader import MultiplexPillarSampler
from src.protocol.prequential import build_multiplex_stream
from src.models.smoothing import MultiplexInductiveSmoother
from src.models.routing import MultiplexRoutingHead
from src.models.multiplex_moe import MultiplexMoE
from src.training.ebl_loss import EBLLoss
from src.training.metrics import calculate_ci, calculate_ef_at_k
import pandas as pd 


def create_pactivity_edges(data):
    """
    Merges pIC50, pKi, and pKd edges into a single 'binds_activity' edge type.
    """
    ei_list = []
    y_list = []
    
    # 1. Extract all available metrics
    metrics = ["binds_pic50", "binds_pki", "binds_pkd"]
    for m in metrics:
        if ("protein", m, "drug") in data.edge_types:
            ei_list.append(data["protein", m, "drug"].edge_index)
            y_list.append(data["protein", m, "drug"].edge_label)
            
    if not ei_list:
        return data
        
    combined_ei = torch.cat(ei_list, dim=1)
    combined_y = torch.cat(y_list, dim=0)
    
    # 2. Deduplicate: (protein_idx * MAX_DRUGS) + drug_idx
    max_drug = data["drug"].num_nodes
    edge_hashes = combined_ei[0] * max_drug + combined_ei[1]
    
    # Use numpy for stable first-occurrence indexing
    _, unique_idx = np.unique(edge_hashes.cpu().numpy(), return_index=True)
    
    final_ei = combined_ei[:, unique_idx]
    final_y = combined_y[unique_idx]
    
    # 3. Inject the new super-edge into the graph
    data["protein", "binds_activity", "drug"].edge_index = final_ei
    data["protein", "binds_activity", "drug"].edge_label = final_y
    
    return data

def main():
    # 1. Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing Oracle Multiplex on {device}...")

    # 2. Load and Merge Data
    print("📦 Loading final_graph_data_not_normalized.pt...")
    # The graph object contains UniProt ID to Index mappings for post-mortem analysis
    data = torch.load("data/final_graph_data_not_normalized.pt", weights_only=False).to(device)
    
    print("🧬 Merging pIC50, pKi, and pKd into a single pActivity metric...")
    # Combining metrics provides the MoE experts with a richer diversity of chemical signatures
    data = create_pactivity_edges(data)
    
    prot_dim = data["protein"].x.size(1)
    drug_dim = data["drug"].x.size(1)
    num_experts = 4
    lr = 5e-4 

    # 3. Initialize Pipeline with pActivity metric
    loader = MultiplexPillarSampler(data, binds_metric="binds_activity")
    # Prequential protocol simulates a real-world discovery timeline
    episodes = build_multiplex_stream(data, binds_metric="binds_activity", min_edges=15)
    print(f"🧬 Stream built: {len(episodes)} protein episodes ready.")

    smoother = MultiplexInductiveSmoother(prot_dim, drug_dim).to(device)
    router = MultiplexRoutingHead(prot_dim, drug_dim, num_experts).to(device)
    model = MultiplexMoE(smoother, router).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr)
    # Aggressive routing parameters for the EBL autopsy
    loss_fn = EBLLoss(ebl_alpha=0.3, temperature=0.1, eps=0.15, rank_weight=0.3)

    # 4. The Pure Cold-Start Arena
    print("\n🎬 STARTING REAL PREQUENTIAL STREAM (PURE COLD-START)")
    print("-" * 80)
    
    # Initialize an Episode Log for post-run UniProt analysis
    episode_log = []
    
    # Start the stream with zero prior binding knowledge
    loader.binds_ei = torch.empty((2, 0), dtype=torch.long, device=device)
    loader.binds_y = torch.empty((0,), dtype=torch.float, device=device)
    loader.binds_w = torch.empty((0,), dtype=torch.float, device=device)
    loader.edge_birth_t = torch.empty((0,), dtype=torch.float, device=device)
    loader._build_bind_index_cache()

    support_batch_size = 128

    for i, ep in enumerate(episodes):
        loader.begin_episode(i)
        # The Inductive Smoother consults structural and functional neighbors
        pillar = loader.get_pillar_context(ep.protein_idx)
        
        # Retrieve UniProt ID for logging
        uniprot_id = data["protein"].index_to_uniprot_id[ep.protein_idx]
        
        # Track "Connectivity" (Sum of neighbors in the inductive pillar)
        n_neighbors = pillar["form_neighbors"].numel() + pillar["role_neighbors"].numel()

        # Phase 1: Blind Evaluation (The Screener Test)
        model.eval()
        with torch.no_grad():
            query_preds, q_gate_probs, _ = model(pillar, data["drug"].x, ep.edges[1])
            ci_val = calculate_ci(ep.labels, query_preds)
            # EF10 measures if we successfully ranked true binders in the top 10%
            ef10_val = calculate_ef_at_k(ep.labels, query_preds, k=0.1)

            # --- basic episode difficulty / class-balance diagnostics ---
            n_preds = int(ep.labels.numel())
            n_pos_ge6 = int((ep.labels >= 6.0).sum().item())
            n_pos_ge7 = int((ep.labels >= 7.0).sum().item())
            pos_rate_ge6 = (n_pos_ge6 / n_preds) if n_preds > 0 else 0.0
            pos_rate_ge7 = (n_pos_ge7 / n_preds) if n_preds > 0 else 0.0
            
            # Record which expert the router favored for this protein family
            winning_expert = torch.argmax(q_gate_probs.mean(dim=0)).item()

        # Phase 2: Full Backprop (Absorb Knowledge)
        model.train()
        optimizer.zero_grad()
        loss_fn.step_schedule(i, len(episodes))

        rank_losses = []
        num_edges = ep.edges.size(1)
        n_batches = max(1, (num_edges + support_batch_size - 1) // support_batch_size)
        for b in range(n_batches):
            start = b * support_batch_size
            end = min((b + 1) * support_batch_size, num_edges)
            mb_drugs = ep.edges[1][start:end]
            mb_labels = ep.labels[start:end]

            sup_preds, s_gate_probs, s_experts = model(pillar, data["drug"].x, mb_drugs)
            losses = loss_fn(sup_preds, mb_labels, s_gate_probs, s_experts)
            (losses["total_loss"] / n_batches).backward()
            rank_losses.append(losses["rank_loss"].item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        rank_loss = float(np.mean(rank_losses)) if rank_losses else 0.0
        
        # Phase 3: Update Global Graph
        # Revealed edges allow future proteins to 'leak' affinity data
        loader.add_revealed_edges(ep.edges, ep.labels)
        
        # Record the episode data
        episode_log.append({
            "episode": i,
            "uniprot_id": uniprot_id,
            "ci": ci_val,
            "ef10": ef10_val,
            "rank_loss": rank_loss,
            "expert_id": winning_expert,
            "connectivity": n_neighbors,
            "n_preds": n_preds,
            "n_pos_ge6": n_pos_ge6,
            "n_pos_ge7": n_pos_ge7,
            "pos_rate_ge6": pos_rate_ge6,
            "pos_rate_ge7": pos_rate_ge7
        })

        if i % 5 == 0:
            gate_str = ", ".join([f"{p:.2f}" for p in q_gate_probs[0].tolist()])
            ram_gb = torch.mps.current_allocated_memory() / (1024**3)
            print(
                f"Ep {i:03d} | {uniprot_id} | CI: {ci_val:.3f} | EF10: {ef10_val:.2f} "
                f"| n={n_preds} pos>=6:{n_pos_ge6} ({pos_rate_ge6:.2%}) pos>=7:{n_pos_ge7} ({pos_rate_ge7:.2%}) "
                f"| Expert: {winning_expert} | RAM: {ram_gb:.2f}GB"
            )

    print("\n💾 Saving trained Oracle Multiplex...")
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Save the analysis CSV
    df_results = pd.DataFrame(episode_log)
    df_results.to_csv("results/stream_analysis_v1.csv", index=False)
    
    torch.save(model.state_dict(), "models/oracle_multiplex_v1.pt")
    print("✅ Weights secured and analysis log saved to results/stream_analysis_v1.csv")

if __name__ == "__main__":
    main()