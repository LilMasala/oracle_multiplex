import torch
import yaml
import os 
from torch.optim import Adam
from src.data.multiplex_loader import MultiplexPillarSampler
from src.protocol.prequential import build_multiplex_stream
from src.models.smoothing import MultiplexInductiveSmoother
from src.models.routing import MultiplexRoutingHead
from src.models.multiplex_moe import MultiplexMoE
from src.training.ebl_loss import EBLLoss
from src.training.metrics import calculate_ci

def main():
    # 1. Setup Device (MPS for your Mac, CUDA for Colab)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing Oracle Multiplex on {device}...")

    # 2. Load Real Data
    print("📦 Loading final_graph_data_not_normalized.pt...")
    data = torch.load("data/final_graph_data_not_normalized.pt", weights_only=False).to(device)
    
    # 3. Hyperparameters (Ideally move these to config/default_config.yaml later)
    prot_dim = data["protein"].x.size(1)
    drug_dim = data["drug"].x.size(1)
    num_experts = 2
    ebl_alpha = 0.5
    lr = 1e-4

    # 4. Initialize Pipeline Components
    loader = MultiplexPillarSampler(data)
    
    # We build the stream (Phase 1)
    # This identifies which proteins are valid for testing (min 10 edges)
    episodes = build_multiplex_stream(data, support_k=8, min_edges=10)
    print(f"🧬 Stream built: {len(episodes)} protein episodes ready.")

    smoother = MultiplexInductiveSmoother(prot_dim, drug_dim).to(device)
    router = MultiplexRoutingHead(prot_dim, drug_dim, num_experts).to(device)
    model = MultiplexMoE(smoother, router).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = EBLLoss(ebl_alpha=ebl_alpha)

    # 5. The Real Arena
    print("\n🎬 STARTING REAL PREQUENTIAL STREAM")
    print("-" * 80)
    
    # We start with a completely empty "revealed" set in the loader
    # because we want to prove we can learn from scratch.
    loader.binds_ei = torch.empty((2, 0), dtype=torch.long, device=device)
    loader.binds_y = torch.empty((0,), dtype=torch.float, device=device)

    for i, ep in enumerate(episodes):
        optimizer.zero_grad()
        
        # Phase 2 & 3: Consultation
        pillar = loader.get_pillar_context(ep.protein_idx)
        
        # ZERO-SHOT EVAL (Before weight update)
        with torch.no_grad():
            query_preds, q_gate_probs, _ = model(pillar, data["drug"].x, ep.query_edges[1])
            query_mse = torch.nn.functional.mse_loss(query_preds, ep.query_labels).item()

        # SUPPORT TRAINING (Learning from revealed labels)
        sup_preds, s_gate_probs, s_experts = model(pillar, data["drug"].x, ep.support_edges[1])
        total_loss, mse, gate_pen = loss_fn(sup_preds, ep.support_labels, s_gate_probs, s_experts)
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # PHASE 4: UPDATE THE GRAPH
        loader.add_revealed_edges(ep.support_edges, ep.support_labels)

        if i % 5 == 0:
            gate_str = ", ".join([f"{p:.2f}" for p in q_gate_probs[0].tolist()])
            # Change your print line to this:
            ci_val = calculate_ci(ep.query_labels, query_preds)
            print(f"Ep {i:03d} | Query CI: {ci_val:.3f} | Prot {ep.protein_idx} | Query MSE: {query_mse:.4f} | Gate: [{gate_str}]")

    print("\n💾 Saving trained Oracle Multiplex...")
    save_path = "models/oracle_multiplex_v1.pt"
    os.makedirs("models", exist_ok=True)
    
    torch.save({
        'smoother': smoother.state_dict(),
        'router': router.state_dict(),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': {
            'prot_dim': prot_dim,
            'drug_dim': drug_dim,
            'num_experts': num_experts
        }
    }, save_path)
    print(f"✅ Weights secured at {save_path}")

if __name__ == "__main__":
    main()