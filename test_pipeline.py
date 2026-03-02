import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

# Import your newly minted architecture!
from src.data.multiplex_loader import MultiplexPillarSampler
from src.models.smoothing import MultiplexInductiveSmoother
from src.models.routing import MultiplexRoutingHead
from src.models.multiplex_moe import MultiplexMoE
from src.training.ebl_loss import EBLLoss

def run_sanity_check():
    # 1. Set up Metal Performance Shaders (MPS) for Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔥 Running on device: {device}")

    # 2. Mock some dimensions
    prot_dim = 128
    drug_dim = 128
    num_experts = 4
    target_prot_idx = 0
    
    # 3. Create a tiny fake HeteroData graph (Phase 1 context)
    data = HeteroData()
    data["protein"].x = torch.randn(10, prot_dim) # 10 proteins
    data["drug"].x = torch.randn(50, drug_dim)    # 50 drugs
    
    # Mock some topology: Prot 0 is similar to Prot 1 & 2 (Form)
    data["protein", "similar", "protein"].edge_index = torch.tensor([[0, 0], [1, 2]])
    # Prot 0 shares GO terms with Prot 3 & 4 (Role)
    data["protein", "go_shared", "protein"].edge_index = torch.tensor([[0, 0], [3, 4]])
    
    # Mock some binding history for the neighbors
    # Prot 1 binds to Drug 10 (pIC50=8.0), Prot 3 binds to Drug 20 (pIC50=4.0)
    data["protein", "binds_pic50", "drug"].edge_index = torch.tensor([[1, 3], [10, 20]])
    data["protein", "binds_pic50", "drug"].edge_label = torch.tensor([8.0, 4.0])
    
    data = data.to(device)

    # 4. Initialize the Pipeline
    loader = MultiplexPillarSampler(data)
    smoother = MultiplexInductiveSmoother(prot_dim, drug_dim).to(device)
    router = MultiplexRoutingHead(prot_dim, drug_dim, num_experts).to(device)
    model = MultiplexMoE(smoother, router).to(device)
    loss_fn = EBLLoss(ebl_alpha=0.2)

    print("\n[Phase 1] Fetching the Multiplex Pillar...")
    pillar = loader.get_pillar_context(target_prot_idx)
    print(f"  Target Prot Features: {pillar['target_features'].shape}")
    print(f"  Form Neighbors found: {pillar['form_neighbors'].shape[0]}")
    print(f"  Role Neighbors found: {pillar['role_neighbors'].shape[0]}")

    # 5. Mock a query/support split (Phase 4 timeline)
    # Let's say we want to rank 5 query drugs, and we have 2 known support drugs to learn from
    query_drug_indices = torch.tensor([5, 6, 7, 8, 9]).to(device)
    support_drug_indices = torch.tensor([11, 12]).to(device)
    true_support_labels = torch.tensor([7.5, 5.2]).to(device) # What the model *should* predict

    print("\n[Phase 2 & 3] Running the Inductive Leak & MoE Gate (Zero-Shot Eval)...")
    # We pass the full drug matrix so the Smoother can build the chemical footprints
    query_preds, gate_probs, _ = model(pillar, data["drug"].x, query_drug_indices)
    
    print(f"  Gate Probabilities (Routing): {gate_probs.detach().cpu().numpy()}")
    print(f"  Predicted pIC50s for {len(query_drug_indices)} Query Drugs: {query_preds.shape}")

    print("\n[Phase 5] Running the Autopsy (EBL Loss on Support Data)...")
    support_preds, sup_gate_probs, sup_expert_tensor = model(pillar, data["drug"].x, support_drug_indices)
    
    total_loss, mse, gate_penalty = loss_fn(support_preds, true_support_labels, sup_gate_probs, sup_expert_tensor)
    
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  --> Base MSE: {mse:.4f}")
    print(f"  --> Gate Routing Penalty: {gate_penalty:.4f}")
    
    print("\n✅ PIPELINE SURVIVED! TENSORS ALIGNED PERFECTLY.")

if __name__ == "__main__":
    run_sanity_check()