import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from src.data.multiplex_loader import MultiplexPillarSampler
from src.data.context_builder import TNPContextBuilder
from src.protocol.prequential import build_multiplex_stream
from src.models.tnp import ProteinLigandTNP
from src.training.tnp_loss import TNPLoss
from src.training.metrics import calculate_ci, calculate_ef_at_k


def create_pactivity_edges(data):
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


def run_episode(model, builder, drug_features, pillar, query_drug_indices):
    """Forward pass for one protein episode. Returns (mu, sigma)."""
    ctx_p, ctx_d, ctx_a, ctx_ppr, ctx_delta, ctx_trust = builder.build_context(pillar)
    target = pillar["target_features"]
    n_q = query_drug_indices.size(0)
    qry_protein = target.unsqueeze(0).expand(n_q, -1)
    qry_drug = drug_features[query_drug_indices]
    return model(ctx_p, ctx_d, ctx_a, qry_protein, qry_drug, ctx_ppr, ctx_delta, ctx_trust)


def main():
    parser = argparse.ArgumentParser(description="TNP prequential streaming experiment")
    parser.add_argument("--data", default="data/final_graph_data_not_normalized.pt")
    parser.add_argument("--priors", default="data/multiplex_priors.pt")
    parser.add_argument("--n-episodes", type=int, default=None, help="Limit number of episodes (default: all)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--replay-weight", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {args.data}...")
    data = torch.load(args.data, weights_only=False).to(device)
    data = create_pactivity_edges(data)

    prot_dim = data["protein"].x.size(1)
    drug_dim = data["drug"].x.size(1)
    drug_features = data["drug"].x

    loader = MultiplexPillarSampler(data, binds_metric="binds_activity",
                                    priors_cache_path=args.priors,
                                    temporal_decay=0.0)
    episodes = build_multiplex_stream(data, binds_metric="binds_activity", min_edges=15)
    if args.n_episodes is not None:
        episodes = episodes[:args.n_episodes]
    print(f"Stream: {len(episodes)} episodes")

    builder = TNPContextBuilder(drug_features, max_context=args.max_context)
    model = ProteinLigandTNP(prot_dim, drug_dim, token_dim=args.token_dim).to(device)
    loss_fn = TNPLoss(rank_weight=0.3)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print("\nSTARTING TNP PREQUENTIAL STREAM")
    print("-" * 70)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Keep full dataset binding edges as prior context (ChEMBL/PubChem historical knowledge).
    # Prequential constraint is on the *current* protein only: evaluate before revealing its labels.
    # episode_log will still measure zero-shot ranking performance on each protein.

    episode_log = []
    ci_history  = []
    ef10_history = []
    ROLL = 50  # rolling window size

    for i, ep in enumerate(episodes):
        loader.begin_episode(i)
        pillar = loader.get_pillar_context(ep.protein_idx)
        n_ctx = (pillar["form_binds_ei"].size(1) + pillar["role_binds_ei"].size(1))

        # --- EVALUATION (cold-start, labels not yet revealed) ---
        model.eval()
        with torch.no_grad():
            mu_eval, sigma_eval = run_episode(model, builder, drug_features, pillar, ep.edges[1])
        ci_val   = calculate_ci(ep.labels, mu_eval)
        ef10_val = calculate_ef_at_k(ep.labels, mu_eval, k=0.1)
        mean_sigma = float(sigma_eval.mean().detach())

        ci_history.append(ci_val)
        ef10_history.append(ef10_val)
        ci_roll   = float(sum(ci_history[-ROLL:])   / len(ci_history[-ROLL:]))
        ef10_roll = float(sum(ef10_history[-ROLL:]) / len(ef10_history[-ROLL:]))

        # --- TRAINING ---
        model.train()
        optimizer.zero_grad()
        loss_fn.step_schedule(i, len(episodes))

        # Current episode supervised loss
        mu_train, sigma_train = run_episode(model, builder, drug_features, pillar, ep.edges[1])
        train_result = loss_fn(mu_train, sigma_train, ep.labels.to(device))
        train_result["total_loss"].backward()
        train_loss = float(train_result["total_loss"].detach())
        nll_val    = float(train_result["nll"].detach())

        # Experience replay
        replay_batches = sample_replay_batch(loader, ep.protein_idx,
                                             replay_edges_per_protein=256, max_replay_proteins=1)
        replay_loss_val = 0.0
        if replay_batches:
            replay_total = torch.tensor(0.0, device=device)
            for _, r_edges, r_labels, r_pillar in replay_batches:
                r_mu, r_sigma = run_episode(model, builder, drug_features, r_pillar, r_edges[1])
                r_result = loss_fn(r_mu, r_sigma, r_labels.to(device))
                replay_total = replay_total + r_result["total_loss"]
            replay_total = replay_total / len(replay_batches)
            (args.replay_weight * replay_total).backward()
            replay_loss_val = float(replay_total.detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loader.add_revealed_edges(ep.edges, ep.labels)

        if i % 50 == 0 and i > 0:
            torch.save(model.state_dict(), f"models/tnp_checkpoint_ep{i:04d}.pt")

        episode_log.append({
            "episode": i,
            "protein_idx": ep.protein_idx,
            "ci": ci_val,
            "ef10": ef10_val,
            "ci_roll50": ci_roll,
            "ef10_roll50": ef10_roll,
            "mean_sigma": mean_sigma,
            "total_loss": train_loss,
            "nll": nll_val,
            "replay_loss": replay_loss_val,
            "n_ctx": n_ctx,
            "n_preds": ep.labels.numel(),
        })

        if i % 5 == 0:
            print(
                f"ep {i:04d}/{len(episodes)} "
                f"| CI: {ci_val:.3f} (roll50: {ci_roll:.3f}) "
                f"| EF10: {ef10_val:.2f} (roll50: {ef10_roll:.2f}) "
                f"| σ: {mean_sigma:.3f} | nll: {nll_val:.3f} "
                f"| ctx: {n_ctx}"
            )

    print("\nSaving TNP model...")
    torch.save(model.state_dict(), "models/oracle_tnp_v1.pt")
    pd.DataFrame(episode_log).to_csv("results/stream_tnp_v1.csv", index=False)
    print("Saved: models/oracle_tnp_v1.pt  results/stream_tnp_v1.csv")


if __name__ == "__main__":
    main()
