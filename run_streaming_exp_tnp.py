"""
TNP prequential streaming experiment with cold-start improvements:
  Unit 1 — Cold-start evaluation harness (regime tracking)
  Unit 2 — Synthetic prior context token (--enable-synthetic-prior)
  Unit 3 — Context density gating (always on, in model architecture)
  Unit 4 — PPR centroid interpolation (always on, in model architecture)
  Unit 5 — Drug analog context injection (--drug-analogs)
  Unit 6 — Protein GNN pre-encoder (--use-gnn)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

from src.data.multiplex_loader import MultiplexPillarSampler
from src.data.context_builder import TNPContextBuilder
from src.data.diverse_replay_buffer import DiverseReplayBuffer
from src.protocol.prequential import build_multiplex_stream
from src.models.tnp import ProteinLigandTNP
from src.training.tnp_loss import TNPLoss
from src.training.metrics import calculate_ci, calculate_ef_at_k
from src.training.cold_start_metrics import classify_regime, summarize_cold_start


def precompute_go_fingerprints(data):
    """
    Unit 9: Mean-pool anc2vec GO embeddings per protein.
    Returns [N_proteins, go_dim] tensor, or None if GO data unavailable.
    """
    edge_key = ("protein", "relates", "go")
    if "go" not in data.node_types or edge_key not in data.edge_types:
        return None
    go_x    = data["go"].x                                      # [N_go, go_dim]
    ei      = data[edge_key].edge_index                         # [2, N_edges]
    prot_idx, go_idx = ei[0], ei[1]
    N_p, go_dim = data["protein"].x.size(0), go_x.size(1)
    device  = data["protein"].x.device
    go_fp   = torch.zeros(N_p, go_dim, device=device)
    count   = torch.zeros(N_p, device=device)
    go_fp.index_add_(0, prot_idx, go_x[go_idx].to(device))
    count.index_add_(0, prot_idx, torch.ones(prot_idx.size(0), device=device))
    count   = count.clamp(min=1.0)
    go_fp   = go_fp / count.unsqueeze(1)
    n_annotated = int((count > 1).sum())
    print(f"  GO fingerprints: {n_annotated}/{N_p} proteins annotated (dim={go_dim})")
    return go_fp


def create_pactivity_edges(data):
    ei_list, y_list = [], []
    for m in ["binds_pic50", "binds_pki", "binds_pkd"]:
        if ("protein", m, "drug") in data.edge_types:
            ei_list.append(data["protein", m, "drug"].edge_index)
            y_list.append(data["protein", m, "drug"].edge_label)
    if not ei_list:
        return data
    combined_ei = torch.cat(ei_list, dim=1)
    combined_y  = torch.cat(y_list, dim=0)
    max_drug = data["drug"].num_nodes
    edge_hashes = combined_ei[0] * max_drug + combined_ei[1]
    _, unique_idx = np.unique(edge_hashes.cpu().numpy(), return_index=True)
    data["protein", "binds_activity", "drug"].edge_index = combined_ei[:, unique_idx]
    data["protein", "binds_activity", "drug"].edge_label = combined_y[unique_idx]
    return data


def sample_replay_batch(
    loader, replay_protein_indices, replay_edges_per_protein=256
):
    """Fetch replay batches for the given protein indices from the loader history."""
    replay_batches = []
    for replay_protein_idx in replay_protein_indices:
        mask     = loader.binds_ei[0] == replay_protein_idx
        edge_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if edge_idx.numel() == 0:
            continue
        k      = min(replay_edges_per_protein, edge_idx.numel())
        chosen = edge_idx[torch.randperm(edge_idx.numel(), device=edge_idx.device)[:k]]
        replay_edges  = loader.binds_ei[:, chosen]
        replay_labels = loader.binds_y[chosen]
        replay_pillar = loader.get_pillar_context(int(replay_protein_idx))
        replay_batches.append((int(replay_protein_idx), replay_edges, replay_labels, replay_pillar))
    return replay_batches


def run_episode(model, builder, drug_features, pillar, query_drug_indices,
                gnn_all_embs=None, global_mean_affinity=6.5, per_query_k=0):
    """
    Forward pass for one protein episode. Returns (mu, sigma).

    Args:
        gnn_all_embs:       [N_proteins, gnn_dim] precomputed GNN embeddings or None
        global_mean_affinity: fallback anchor for cold-start (n_ctx == 0)
        per_query_k:        >0 → use per-query dynamic context (Unit 8)
    """
    target       = pillar["target_features"]
    ppr_centroid = pillar.get("ppr_centroid")
    n_q          = query_drug_indices.size(0)
    qry_protein  = target.unsqueeze(0).expand(n_q, -1)
    qry_drug     = drug_features[query_drug_indices]

    qry_gnn_emb = None
    if gnn_all_embs is not None:
        target_idx  = int(pillar["target_idx"])
        qry_gnn_emb = gnn_all_embs[target_idx].unsqueeze(0).expand(n_q, -1)

    # Unit 9: GO fingerprint for the target protein
    qry_go_fp = None
    if builder.go_fingerprints is not None:
        target_idx = int(pillar["target_idx"])
        qry_go_fp  = builder.go_fingerprints[target_idx].unsqueeze(0).expand(n_q, -1)

    if per_query_k > 0:
        # Unit 8: per-query dynamic context
        pq_p, pq_d, pq_a, pq_ppr, pq_trust, pq_gnn, pq_aff_mean, pq_go_fp = \
            builder.build_per_query_context(pillar, query_drug_indices, per_query_k)
        return model.forward_per_query(
            pq_p, pq_d, pq_a, pq_ppr, pq_trust,
            qry_protein, qry_drug, pq_aff_mean,
            pq_gnn_emb=pq_gnn,
            qry_gnn_emb=qry_gnn_emb,
            pq_go_fp=pq_go_fp,
            qry_go_fp=qry_go_fp,
            ppr_centroid=ppr_centroid,
        )

    # Shared context path (default)
    ctx_p, ctx_d, ctx_a, ctx_ppr, ctx_trust, ctx_gnn, ctx_go_fp = builder.build_context(
        pillar, query_drug_indices
    )
    return model(
        ctx_p, ctx_d, ctx_a,
        qry_protein, qry_drug,
        ctx_ppr, ctx_trust,
        ppr_centroid=ppr_centroid,
        ctx_gnn_emb=ctx_gnn,
        qry_gnn_emb=qry_gnn_emb,
        ctx_go_fp=ctx_go_fp,
        qry_go_fp=qry_go_fp,
        global_mean_affinity=global_mean_affinity,
    )


def main():
    parser = argparse.ArgumentParser(description="TNP prequential streaming experiment")
    parser.add_argument("--data",     default="data/final_graph_data_not_normalized.pt")
    parser.add_argument("--priors",   default="data/multiplex_priors.pt")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Limit number of episodes (default: all)")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--token-dim",  type=int,   default=256)
    parser.add_argument("--max-context", type=int,  default=256)
    parser.add_argument("--replay-weight", type=float, default=0.5)

    # Unit 1: cold-start only mode
    parser.add_argument("--cold-start-only", action="store_true",
                        help="Evaluation-only mode; skip training (Unit 1)")

    # Unit 2: synthetic prior
    parser.add_argument("--enable-synthetic-prior", action="store_true",
                        help="Inject synthetic prior token at cold-start (Unit 2)")

    # Unit 5: drug analog injection
    parser.add_argument("--drug-analogs", action="store_true",
                        help="Enable drug analog context injection (Unit 5)")
    parser.add_argument("--analog-top-k", type=int, default=32,
                        help="Top-K analogues per drug (Unit 5)")

    # Unit 9: GO functional fingerprints (anc2vec mean-pool)
    parser.add_argument("--use-go", action="store_true",
                        help="Inject GO anc2vec fingerprints into protein encoder (Unit 9)")

    # Unit 8: per-query dynamic context
    parser.add_argument("--per-query-k", type=int, default=0,
                        help="Per-query context size >0 enables Unit 8 (default: 0 = shared)")

    # Unit 6: GNN pre-encoder
    parser.add_argument("--use-gnn", action="store_true",
                        help="Enable protein GNN pre-encoder (Unit 6)")
    parser.add_argument("--gnn-out-dim", type=int, default=256,
                        help="GNN embedding dimension (Unit 6)")
    parser.add_argument("--gnn-refresh-interval", type=int, default=100,
                        help="Episodes between GNN embedding refreshes (Unit 6)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {args.data}...")
    data = torch.load(args.data, weights_only=False).to(device)
    data = create_pactivity_edges(data)

    prot_dim    = data["protein"].x.size(1)
    drug_dim    = data["drug"].x.size(1)
    drug_features = data["drug"].x

    loader   = MultiplexPillarSampler(data, binds_metric="binds_activity",
                                      priors_cache_path=args.priors,
                                      temporal_decay=0.0)
    replay_buffer = DiverseReplayBuffer(max_size=1000, protein_dim=prot_dim, device=device)
    episodes = build_multiplex_stream(data, binds_metric="binds_activity", min_edges=15)
    if args.n_episodes is not None:
        episodes = episodes[:args.n_episodes]
    print(f"Stream: {len(episodes)} episodes")

    # --- Unit 2: Compute global statistics for synthetic prior ---
    global_drug_mean     = drug_features.mean(0)
    global_mean_affinity = float(
        data["protein", "binds_activity", "drug"].edge_label.mean()
    )
    print(f"Global mean affinity: {global_mean_affinity:.3f}")

    # --- Unit 5: Drug analog index ---
    drug_analog_index = None
    if args.drug_analogs:
        print(f"Building DrugAnalogIndex (top_k={args.analog_top_k})...")
        from src.data.drug_analog_index import DrugAnalogIndex
        drug_analog_index = DrugAnalogIndex(drug_features, top_k=args.analog_top_k)
        print("DrugAnalogIndex ready.")

    # --- Unit 6: Protein GNN pre-encoder ---
    gnn_all_embs = None
    protein_gnn  = None
    gnn_emb_dim  = 0
    if args.use_gnn:
        print(f"Initialising ProteinGNN (out_dim={args.gnn_out_dim})...")
        from src.models.protein_gnn import ProteinGNN, compute_all_embeddings
        protein_gnn = ProteinGNN(
            in_dim=prot_dim, hidden_dim=128, out_dim=args.gnn_out_dim
        ).to(device)
        gnn_emb_dim = args.gnn_out_dim
        form_ei = data["protein", "similar",   "protein"].edge_index
        role_ei = data["protein", "go_shared", "protein"].edge_index
        gnn_all_embs = compute_all_embeddings(
            protein_gnn, data["protein"].x, form_ei, role_ei
        )
        print(f"GNN embeddings: {gnn_all_embs.shape}")

    # --- Unit 9: GO fingerprints ---
    go_fingerprints = None
    go_fp_dim       = 0
    if args.use_go:
        go_fingerprints = precompute_go_fingerprints(data)
        if go_fingerprints is not None:
            go_fp_dim = go_fingerprints.size(1)

    # --- Build context builder ---
    builder = TNPContextBuilder(
        drug_features,
        max_context=args.max_context,
        global_drug_mean=global_drug_mean,
        global_mean_affinity=global_mean_affinity,
        enable_synthetic_prior=args.enable_synthetic_prior,
        drug_analog_index=drug_analog_index,
        gnn_protein_embs=gnn_all_embs,
    )
    builder.go_fingerprints = go_fingerprints

    # --- Build model ---
    model = ProteinLigandTNP(
        prot_dim, drug_dim,
        token_dim=args.token_dim,
        gnn_emb_dim=gnn_emb_dim,
        go_fp_dim=go_fp_dim,
    ).to(device)


    loss_fn   = TNPLoss().to(device)
    model_params = list(model.parameters())
    if protein_gnn is not None:
        model_params += list(protein_gnn.parameters())
    # log_var scalars are three lightweight calibration parameters — they need
    # to adapt much faster than the transformer weights to be effective.
    optimizer = Adam([
        {"params": model_params,               "lr": args.lr},
        {"params": list(loss_fn.parameters()), "lr": args.lr * 10},
    ])

    print("\nSTARTING TNP PREQUENTIAL STREAM")
    print(f"  synthetic prior: {args.enable_synthetic_prior}")
    print(f"  drug analogs:    {args.drug_analogs}")
    print(f"  GNN pre-encoder: {args.use_gnn}")
    print(f"  cold-start only: {args.cold_start_only}")
    print(f"  per-query-k:     {args.per_query_k} {'(Unit 8 enabled)' if args.per_query_k > 0 else '(shared context)'}")
    print(f"  GO fingerprints: {args.use_go} (dim={go_fp_dim})")
    print("-" * 70)

    os.makedirs("models",   exist_ok=True)
    os.makedirs("results",  exist_ok=True)

    episode_log  = []
    ci_history   = []
    ef10_history = []
    ROLL = 100

    for i, ep in enumerate(episodes):
        loader.begin_episode(i)
        pillar = loader.get_pillar_context(ep.protein_idx)
        n_ctx  = int(
            pillar["form_binds_ei"].size(1) + pillar["role_binds_ei"].size(1)
        )

        # Unit 1: classify regime
        regime = classify_regime(n_ctx)

        # --- EVALUATION (cold-start, labels not yet revealed) ---
        model.eval()
        with torch.no_grad():
            mu_eval, sigma_eval = run_episode(
                model, builder, drug_features, pillar, ep.edges[1], gnn_all_embs,
                global_mean_affinity=global_mean_affinity,
                per_query_k=args.per_query_k,
            )
        ci_val     = calculate_ci(ep.labels, mu_eval)
        ef10_val   = calculate_ef_at_k(ep.labels, mu_eval, k=0.1)
        mean_sigma = float(sigma_eval.mean().detach())

        ci_history.append(ci_val)
        ef10_history.append(ef10_val)
        ci_roll   = float(sum(ci_history[-ROLL:])   / len(ci_history[-ROLL:]))
        ef10_roll = float(sum(ef10_history[-ROLL:]) / len(ef10_history[-ROLL:]))

        train_loss      = 0.0
        nll_val         = 0.0
        replay_loss_val = 0.0
        w_nll = w_listnet = w_lambda = 0.0

        if not args.cold_start_only:
            # --- TRAINING ---
            model.train()
            optimizer.zero_grad()
            loss_fn.step_schedule(i, len(episodes))

            mu_train, sigma_train = run_episode(
                model, builder, drug_features, pillar, ep.edges[1], gnn_all_embs,
                global_mean_affinity=global_mean_affinity,
                per_query_k=args.per_query_k,
            )
            train_result = loss_fn(mu_train, sigma_train, ep.labels.to(device))
            train_result["total_loss"].backward()

            train_loss = float(train_result["total_loss"].detach())
            nll_val    = float(train_result["nll"].detach())
            w_nll      = float(train_result["w_nll"])
            w_listnet  = float(train_result["w_listnet"])
            w_lambda   = float(train_result["w_lambda"])

            # Experience replay — diverse coreset sampling
            replay_protein_indices = replay_buffer.sample(25)
            replay_batches = sample_replay_batch(
                loader, replay_protein_indices,
                replay_edges_per_protein=256,
            )
            if replay_batches:
                replay_total = torch.tensor(0.0, device=device)
                for _, r_edges, r_labels, r_pillar in replay_batches:
                    r_mu, r_sigma = run_episode(
                        model, builder, drug_features, r_pillar, r_edges[1], gnn_all_embs,
                        global_mean_affinity=global_mean_affinity,
                        per_query_k=args.per_query_k,
                    )
                    r_result     = loss_fn(r_mu, r_sigma, r_labels.to(device))
                    replay_total = replay_total + r_result["total_loss"]
                replay_total  = replay_total / len(replay_batches)
                (args.replay_weight * replay_total).backward()
                replay_loss_val = float(replay_total.detach())

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(loss_fn.parameters()), 1.0
            )
            optimizer.step()

            # Unit 6: refresh GNN embeddings periodically
            if (
                protein_gnn is not None
                and i % args.gnn_refresh_interval == 0
                and i > 0
            ):
                from src.models.protein_gnn import compute_all_embeddings
                gnn_all_embs = compute_all_embeddings(
                    protein_gnn, data["protein"].x, form_ei, role_ei
                )
                builder.gnn_protein_embs = gnn_all_embs

        loader.add_revealed_edges(ep.edges, ep.labels)
        replay_buffer.add(ep.protein_idx, pillar["target_features"])

        if i % 50 == 0 and i > 0:
            torch.save(model.state_dict(), f"models/tnp_checkpoint_ep{i:04d}.pt")

        episode_log.append({
            "episode":      i,
            "protein_idx":  ep.protein_idx,
            "regime":       regime,           # Unit 1
            "ci":           ci_val,
            "ef10":         ef10_val,
            "ci_roll100":   ci_roll,
            "ef10_roll100": ef10_roll,
            "mean_sigma":   mean_sigma,
            "total_loss":   train_loss,
            "nll":          nll_val,
            "replay_loss":  replay_loss_val,
            "n_ctx":        n_ctx,
            "n_preds":      ep.labels.numel(),
            "w_nll":        w_nll,
            "w_listnet":    w_listnet,
            "w_lambda":     w_lambda,
        })

        if i % 5 == 0:
            print(
                f"ep {i:04d}/{len(episodes)} [{regime:6s}]"
                f" | CI: {ci_val:.3f} (roll100: {ci_roll:.3f})"
                f" | σ: {mean_sigma:.3f}"
                f" | n_ctx: {n_ctx}"
                f" | weights -> NLL: {w_nll:.4f}, ListNet: {w_listnet:.4f}, Lambda: {w_lambda:.4f}"
            )

    print("\nSaving TNP model...")
    torch.save(model.state_dict(), "models/oracle_tnp_v1.pt")

    df = pd.DataFrame(episode_log)
    df.to_csv("results/stream_tnp_v1.csv", index=False)
    print("Saved: models/oracle_tnp_v1.pt  results/stream_tnp_v1.csv")

    # Unit 1: write cold-start summary
    summary = summarize_cold_start(episode_log)
    if not summary.empty:
        summary.to_csv("results/cold_start_summary.csv")
        print("Saved: results/cold_start_summary.csv")
        print("\nCold-start summary:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
