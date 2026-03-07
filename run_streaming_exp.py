import os
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from src.data.multiplex_loader import MultiplexPillarSampler
from src.protocol.prequential import build_multiplex_stream
from src.models.smoothing import MultiplexInductiveSmoother
from src.models.routing import MultiplexRoutingHead
from src.models.multiplex_moe import MultiplexMoE
from src.training.ebl_loss import EBLLoss
from src.training.metrics import calculate_ci, calculate_ef_at_k


def create_pactivity_edges(data):
    """Merges pIC50, pKi, and pKd edges into a single 'binds_activity' edge type."""
    ei_list, y_list = [], []
    for m in ["binds_pic50", "binds_pki", "binds_pkd"]:
        if ("protein", m, "drug") in data.edge_types:
            ei_list.append(data["protein", m, "drug"].edge_index)
            y_list.append(data["protein", m, "drug"].edge_label)

    if not ei_list:
        return data

    combined_ei = torch.cat(ei_list, dim=1)
    combined_y = torch.cat(y_list, dim=0)
    max_drug = data["drug"].num_nodes
    edge_hashes = combined_ei[0] * max_drug + combined_ei[1]
    _, unique_idx = np.unique(edge_hashes.cpu().numpy(), return_index=True)

    data["protein", "binds_activity", "drug"].edge_index = combined_ei[:, unique_idx]
    data["protein", "binds_activity", "drug"].edge_label = combined_y[unique_idx]
    return data


def sample_replay_batch(loader, current_protein_idx, replay_edges_per_protein=256, max_replay_proteins=1):
    """Protein-level replay sampler from historical amortized buffer.

    Returns list of tuples: (replay_protein_idx, replay_edges[2, R], replay_labels[R], replay_pillar)
    """
    if loader.binds_ei.size(1) == 0:
        return []

    hist_proteins = torch.unique(loader.binds_ei[0])
    hist_proteins = hist_proteins[hist_proteins != int(current_protein_idx)]
    if hist_proteins.numel() == 0:
        return []

    n_select = min(max_replay_proteins, hist_proteins.numel())
    perm = torch.randperm(hist_proteins.numel(), device=hist_proteins.device)[:n_select]
    replay_proteins = hist_proteins[perm]

    replay_batches = []
    for replay_protein_idx in replay_proteins.tolist():
        mask = loader.binds_ei[0] == replay_protein_idx
        edge_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if edge_idx.numel() == 0:
            continue

        k = min(replay_edges_per_protein, edge_idx.numel())
        chosen = edge_idx[torch.randperm(edge_idx.numel(), device=edge_idx.device)[:k]]
        replay_edges = loader.binds_ei[:, chosen]
        replay_labels = loader.binds_y[chosen]
        replay_pillar = loader.get_pillar_context(int(replay_protein_idx))
        replay_batches.append((int(replay_protein_idx), replay_edges, replay_labels, replay_pillar))

    return replay_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Oracle Multiplex on {device}...")

    print("📦 Loading final_graph_data_not_normalized.pt...")
    data = torch.load("data/final_graph_data_not_normalized.pt", weights_only=False).to(device)
    print("🧬 Merging pIC50, pKi, and pKd into a single pActivity metric...")
    data = create_pactivity_edges(data)

    prot_dim = data["protein"].x.size(1)
    drug_dim = data["drug"].x.size(1)
    num_experts, lr = 4, 5e-4

    loader = MultiplexPillarSampler(data, binds_metric="binds_activity", priors_cache_path="data/multiplex_priors.pt")
    episodes = build_multiplex_stream(data, binds_metric="binds_activity", min_edges=15)
    print(f"🧬 Stream built: {len(episodes)} protein episodes ready.")

    smoother = MultiplexInductiveSmoother(prot_dim, drug_dim).to(device)
    router = MultiplexRoutingHead(prot_dim, drug_dim, num_experts).to(device)
    model = MultiplexMoE(smoother, router).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = EBLLoss(ebl_alpha=0.3, temperature=0.1, eps=0.15, rank_weight=0.3)

    print("\n🎬 STARTING REAL PREQUENTIAL STREAM (PURE COLD-START)")
    print("-" * 80)

    os.makedirs("models", exist_ok=True)

    episode_log = []
    loader.binds_ei = torch.empty((2, 0), dtype=torch.long, device=device)
    loader.binds_y = torch.empty((0,), dtype=torch.float, device=device)
    loader.binds_w = torch.empty((0,), dtype=torch.float, device=device)
    loader.edge_birth_t = torch.empty((0,), dtype=torch.float, device=device)
    loader._refresh_bind_sorted_index()

    support_batch_size = 128
    replay_weight = 0.25

    for i, ep in enumerate(episodes):
        loader.begin_episode(i)
        pillar = loader.get_pillar_context(ep.protein_idx)
        uniprot_id = data["protein"].index_to_uniprot_id[ep.protein_idx]
        n_neighbors = pillar["form_neighbors"].numel() + pillar["role_neighbors"].numel()

        model.eval()
        with torch.no_grad():
            query_preds, q_gate_probs, _, q_stats = model(pillar, data["drug"].x, ep.edges[1])
            ci_val = calculate_ci(ep.labels, query_preds)
            ef10_val = calculate_ef_at_k(ep.labels, query_preds, k=0.1)
            delta_norm = float(q_stats["delta_norm"].item())

            n_preds = int(ep.labels.numel())
            n_pos_ge6 = int((ep.labels >= 6.0).sum().item())
            n_pos_ge7 = int((ep.labels >= 7.0).sum().item())
            pos_rate_ge6 = (n_pos_ge6 / n_preds) if n_preds > 0 else 0.0
            pos_rate_ge7 = (n_pos_ge7 / n_preds) if n_preds > 0 else 0.0
            winning_expert = torch.argmax(q_gate_probs.mean(dim=0)).item()

        model.train()
        optimizer.zero_grad()
        loss_fn.step_schedule(i, len(episodes))

        rank_losses = []
        replay_losses = []
        current_losses = []
        num_edges = ep.edges.size(1)
        n_batches = max(1, (num_edges + support_batch_size - 1) // support_batch_size)

        # Protein-level replay sampling from historical buffer.
        replay_batches = sample_replay_batch(
            loader,
            current_protein_idx=ep.protein_idx,
            replay_edges_per_protein=256,
            max_replay_proteins=1,
        )

        for b in range(n_batches):
            start, end = b * support_batch_size, min((b + 1) * support_batch_size, num_edges)
            mb_drugs = ep.edges[1][start:end]
            mb_labels = ep.labels[start:end]

            # Current episode loss (keeps episode-specific gate dynamics / ELBO path).
            sup_preds, s_gate_probs, s_experts, s_stats = model(pillar, data["drug"].x, mb_drugs)
            curr_losses = loss_fn(sup_preds, mb_labels, s_gate_probs, s_experts, protein_level_gate=True)
            total = curr_losses["total_loss"] / n_batches
            current_losses.append(float(curr_losses["total_loss"].detach().item()))
            rank_losses.append(curr_losses["rank_loss"].item())

            # Replay is supervised-only EBLLoss term (no ELBO call here).
            if replay_batches:
                replay_total = 0.0
                for replay_protein_idx, replay_edges, replay_labels, replay_pillar in replay_batches:
                    replay_drugs = replay_edges[1]
                    r_preds, r_gate_probs, r_experts, _ = model(replay_pillar, data["drug"].x, replay_drugs)
                    r_loss = loss_fn(r_preds, replay_labels, r_gate_probs, r_experts)
                    replay_total = replay_total + r_loss["total_loss"]

                replay_total = replay_total / max(len(replay_batches), 1)
                replay_losses.append(float(replay_total.detach().item()))
                total = total + replay_weight * (replay_total / n_batches)

            total.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if i % 50 == 0 and i > 0:
            torch.save(model.state_dict(), f"models/checkpoint_ep{i:04d}.pt")

        rank_loss = float(np.mean(rank_losses)) if rank_losses else 0.0
        replay_loss = float(np.mean(replay_losses)) if replay_losses else 0.0
        current_loss = float(np.mean(current_losses)) if current_losses else 0.0

        loader.add_revealed_edges(ep.edges, ep.labels)

        episode_log.append(
            {
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
                "pos_rate_ge7": pos_rate_ge7,
                "delta_norm": delta_norm,
                "current_loss": current_loss,
                "replay_loss": replay_loss,
            }
        )

        if i % 5 == 0:
            print(
                f"Ep {i:03d} | {uniprot_id} | CI: {ci_val:.3f} | EF10: {ef10_val:.2f} "
                f"| ||delta||: {delta_norm:.3f} | current_loss: {current_loss:.3f} | replay_loss: {replay_loss:.3f} "
                f"| n={n_preds} pos>=6:{n_pos_ge6} ({pos_rate_ge6:.2%}) pos>=7:{n_pos_ge7} ({pos_rate_ge7:.2%}) "
                f"| Expert: {winning_expert}"
            )

    print("\n💾 Saving trained Oracle Multiplex...")
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    pd.DataFrame(episode_log).to_csv("results/stream_analysis_v1.csv", index=False)
    torch.save(model.state_dict(), "models/oracle_multiplex_v1.pt")
    print("✅ Weights secured and analysis log saved to results/stream_analysis_v1.csv")


if __name__ == "__main__":
    main()
