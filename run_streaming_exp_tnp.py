"""
Strict prequential TNP experiment runner with debug-oriented baselines.

This script keeps the cold-start evaluator honest:
  - revealed binding history is explicit (`--history-mode`)
  - duplicate protein-drug reveals are deduplicated
  - `binds_activity` merging uses an explicit reduction policy
  - simple baselines can be compared against the full TNP in the same harness
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.data.binds_activity import merge_activity_edges
from src.data.context_builder import TNPContextBuilder
from src.data.diverse_replay_buffer import DiverseReplayBuffer
from src.data.drug_first_context import DrugFirstContextBuilder
from src.data.multiplex_loader import MultiplexPillarSampler
from src.models.gp_affinity import GPAffinityModel
from src.models.neighbor_transfer import NeighborTransferModel
from src.models.tnp import BindingEncoder, BindingOnlyAffinityModel, ProteinLigandTNP
from src.protocol.prequential import build_multiplex_stream
from src.training.cold_start_metrics import classify_regime, summarize_cold_start
from src.training.metrics import calculate_ci, calculate_ef_at_k
from src.training.tnp_loss import TNPLoss


def _concordance_index(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Harrell's C-index: fraction of concordant pairs among all orderable pairs."""
    n = pred.size(0)
    if n < 2:
        return 0.5
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dp = float(pred[i]) - float(pred[j])
            dt = float(target[i]) - float(target[j])
            if dt == 0:
                continue
            if dp * dt > 0:
                concordant += 1
            elif dp * dt < 0:
                discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


def pretrain_binding_encoder(
    protein_features: torch.Tensor,  # [N_prot, prot_dim]
    drug_features: torch.Tensor,     # [N_drug, drug_dim]
    edges: torch.Tensor,             # [2, n_edges]
    labels: torch.Tensor,            # [n_edges]
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    hidden: int = 256,
    val_frac: float = 0.2,
    extra_epochs: int = 0,
) -> "BindingEncoder":
    """Train a BindingEncoder offline with LR scheduling and return it frozen."""
    prot_dim = protein_features.size(1)
    drug_dim = drug_features.size(1)

    prot_feats = protein_features[edges[0]].to(device)
    drug_feats = drug_features[edges[1]].to(device)
    affinities = labels.to(device)

    n = affinities.size(0)
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    model = BindingEncoder(prot_dim, drug_dim, hidden=hidden).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=lr * 1e-3
    )

    total_epochs = epochs + extra_epochs
    for e in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred_train = model(prot_feats[train_idx], drug_feats[train_idx])
        train_loss = F.mse_loss(pred_train, affinities[train_idx])
        train_loss.backward()
        optimizer.step()

        if e % 10 == 0 or e == total_epochs:
            model.eval()
            with torch.no_grad():
                pred_val = model(prot_feats[val_idx], drug_feats[val_idx])
                val_mse = float(F.mse_loss(pred_val, affinities[val_idx]))
                # CI on a subsample to keep it fast (max 2000 pairs)
                ci_idx = val_idx[:2000] if val_idx.size(0) > 2000 else val_idx
                ci = _concordance_index(
                    model(prot_feats[ci_idx], drug_feats[ci_idx]),
                    affinities[ci_idx],
                )
            scheduler.step(val_mse)
            current_lr = optimizer.param_groups[0]["lr"]
            phase = "base" if e <= epochs else "ext "
            print(
                f"  [pretrain/{phase}] epoch {e}/{total_epochs} | "
                f"train_mse={float(train_loss):.4f} | val_mse={val_mse:.4f} | "
                f"val_CI={ci:.3f} | lr={current_lr:.2e}"
            )

    for p in model.parameters():
        p.requires_grad_(False)
    return model


def compute_prior(
    frozen_be,           # BindingEncoder or None
    qry_protein,         # [n_qry, prot_dim]
    qry_drug,            # [n_qry, drug_dim]
    global_mean: float,
) -> torch.Tensor:
    if frozen_be is None:
        return torch.full((qry_protein.size(0),), global_mean, device=qry_protein.device)
    with torch.no_grad():
        return frozen_be(qry_protein, qry_drug) + global_mean


def compute_residuals(
    frozen_be,           # BindingEncoder or None
    prot_feats,          # [n, prot_dim]
    drug_feats,          # [n, drug_dim]
    labels,              # [n]
    global_mean: float,
) -> torch.Tensor:
    if frozen_be is None:
        return labels
    with torch.no_grad():
        return labels - frozen_be(prot_feats, drug_feats) - global_mean


class GlobalMeanAffinityModel(nn.Module):
    """Constant floor baseline: every query gets the global mean affinity."""

    def __init__(self):
        super().__init__()
        self.last_forward_stats = {}

    def forward(
        self,
        qry_protein: torch.Tensor,
        qry_drug: torch.Tensor,
        global_mean_affinity: float = 6.5,
    ):
        mu = torch.full(
            (qry_protein.size(0),),
            float(global_mean_affinity),
            dtype=qry_protein.dtype,
            device=qry_protein.device,
        )
        sigma = torch.ones_like(mu)
        self.last_forward_stats = {
            "mu_std": 0.0,
            "binding_prior_std": 0.0,
            "log_ppr_alpha": 0.0,
            "centroid_alpha": 0.0,
            "density": 0.0,
        }
        return mu, sigma


def precompute_go_fingerprints(data):
    """Mean-pool anc2vec GO embeddings per protein."""
    edge_key = ("protein", "relates", "go")
    if "go" not in data.node_types or edge_key not in data.edge_types:
        return None

    go_x = data["go"].x
    edge_index = data[edge_key].edge_index
    prot_idx, go_idx = edge_index[0], edge_index[1]
    n_proteins, go_dim = data["protein"].x.size(0), go_x.size(1)
    device = data["protein"].x.device

    pooled = torch.zeros(n_proteins, go_dim, device=device)
    counts = torch.zeros(n_proteins, device=device)
    pooled.index_add_(0, prot_idx, go_x[go_idx].to(device))
    counts.index_add_(0, prot_idx, torch.ones(prot_idx.size(0), device=device))
    annotated = int((counts > 0).sum().item())
    pooled = pooled / counts.clamp(min=1.0).unsqueeze(1)
    print(f"  GO fingerprints: {annotated}/{n_proteins} proteins annotated (dim={go_dim})")
    return pooled


def sample_replay_batch(loader, replay_protein_indices, replay_edges_per_protein=256):
    """Fetch replay batches for the given protein indices from revealed history."""
    replay_batches = []
    for replay_protein_idx in replay_protein_indices:
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


def split_stream_episodes(episodes, historical_frac: float):
    """Split a shuffled protein stream into historical and streamed episodes."""
    if historical_frac <= 0.0:
        return [], episodes
    if historical_frac >= 1.0:
        raise ValueError("historical_protein_frac must be < 1.0 so at least one streamed protein remains")

    historical_count = int(len(episodes) * historical_frac)
    if historical_count <= 0:
        return [], episodes
    if historical_count >= len(episodes):
        raise ValueError("historical_protein_frac leaves no streamed proteins")
    return episodes[:historical_count], episodes[historical_count:]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Strict TNP prequential streaming experiment")
    parser.add_argument("--data", default="data/final_graph_data_not_normalized.pt")
    parser.add_argument("--priors", default="data/multiplex_priors.pt")
    parser.add_argument("--run-name", default=None, help="Output stem for model/results files")
    parser.add_argument("--n-episodes", type=int, default=None, help="Limit number of episodes (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Protein stream seed")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--merge-reduce", choices=["amax", "mean"], default="amax")
    parser.add_argument(
        "--model-kind",
        choices=["tnp", "binding-only", "global-mean", "neighbor-transfer", "gp"],
        default="tnp",
    )
    parser.add_argument("--history-mode", choices=["empty", "full"], default="empty")
    parser.add_argument("--strict-baseline", action="store_true", help="Force the minimal strict-debug configuration")
    parser.add_argument("--cold-start-only", action="store_true", help="Evaluation-only mode; skip parameter updates")
    parser.add_argument("--enable-synthetic-prior", action="store_true", help="Inject synthetic prior token at cold-start")
    parser.add_argument("--drug-analogs", action="store_true", help="Enable drug analog context injection")
    parser.add_argument("--analog-top-k", type=int, default=32, help="Top-K analogues per drug")
    parser.add_argument("--use-go", action="store_true", help="Inject GO anc2vec fingerprints into the protein encoder")
    parser.add_argument("--per-query-k", type=int, default=0, help="Per-query context size (>0 enables dynamic context)")
    parser.add_argument("--use-gnn", action="store_true", help="Backward-compatible shortcut for --gnn-mode frozen")
    parser.add_argument("--gnn-mode", choices=["off", "frozen", "trainable"], default="off")
    parser.add_argument("--gnn-out-dim", type=int, default=256, help="Protein GNN embedding dimension")
    parser.add_argument("--train-scope", choices=["full", "head-only"], default="full")
    parser.add_argument("--unfreeze-after", type=int, default=None, help="Episode index at which TNP switches from head-only to full")
    parser.add_argument("--warmstart-checkpoint", default=None, help="Optional full-model warmstart checkpoint")
    parser.add_argument("--neighbor-k", type=int, default=8, help="Top-k exact-drug neighbors for neighbor-transfer model")
    parser.add_argument(
        "--historical-protein-frac",
        type=float,
        default=0.0,
        help="Protein-disjoint fraction consumed before the streamed evaluation and used to seed history",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=100,
                        help="Epochs for offline BindingEncoder pretraining (GP model only)")
    parser.add_argument("--pretrain-lr", type=float, default=1e-3,
                        help="Learning rate for offline BindingEncoder pretraining")
    parser.add_argument("--pretrain-extra-epochs", type=int, default=0,
                        help="Additional epochs to run after LR scheduler has converged")
    return parser


def apply_presets(args):
    if args.use_gnn and args.gnn_mode == "off":
        args.gnn_mode = "frozen"

    if args.strict_baseline:
        args.history_mode = "empty"
        args.enable_synthetic_prior = False
        args.drug_analogs = False
        args.use_go = False
        args.per_query_k = 0
        args.gnn_mode = "off"

    if args.model_kind not in {"tnp"} and args.gnn_mode != "off":
        print(f"Ignoring gnn_mode={args.gnn_mode} for model_kind={args.model_kind}")
        args.gnn_mode = "off"

    if args.model_kind != "tnp" and args.train_scope != "full":
        print(f"Ignoring train_scope={args.train_scope} for model_kind={args.model_kind}")
        args.train_scope = "full"

    if args.model_kind != "tnp" and args.per_query_k != 0:
        print(f"Ignoring per_query_k={args.per_query_k} for model_kind={args.model_kind}")
        args.per_query_k = 0

    if args.model_kind == "gp":
        # GP has its own drug-analog handling (level 2); disable TNP-specific features
        args.gnn_mode = "off"
        args.use_go = False
        args.enable_synthetic_prior = False
        args.per_query_k = 0

    if args.historical_protein_frac > 0 and args.history_mode != "empty":
        print(
            f"Using history_mode=empty because historical_protein_frac={args.historical_protein_frac} "
            "already seeds history explicitly."
        )
        args.history_mode = "empty"

    return args


def default_run_name(args):
    parts = ["stream", args.model_kind.replace("-", "_"), args.history_mode]
    if args.strict_baseline:
        parts.append("strict")
    if args.per_query_k > 0:
        parts.append(f"pq{args.per_query_k}")
    if args.model_kind == "neighbor-transfer":
        parts.append(f"nk{args.neighbor_k}")
    if args.historical_protein_frac > 0:
        parts.append(f"hist{int(round(args.historical_protein_frac * 100))}")
    if args.enable_synthetic_prior:
        parts.append("syn")
    if args.use_go:
        parts.append("go")
    if args.drug_analogs:
        parts.append("analogs")
    if args.gnn_mode != "off":
        parts.append(f"gnn_{args.gnn_mode}")
    return "_".join(parts)


def set_tnp_train_scope(model, scope: str):
    if not isinstance(model, ProteinLigandTNP):
        return
    if scope == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    if scope != "head-only":
        raise ValueError(f"Unsupported train_scope='{scope}'")

    for param in model.parameters():
        param.requires_grad = False
    for module in (model.output_head, model.binding_encoder, model.prior_proj):
        for param in module.parameters():
            param.requires_grad = True
    model.cold_start_bias.requires_grad = True


def maybe_load_model_warmstart(model, path, device):
    if path is None:
        return
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded model warmstart from {path} | missing={len(missing)} unexpected={len(unexpected)}")


def build_gnn_runtime(args, data, prot_dim, device):
    if args.gnn_mode == "off":
        return None, 0

    from src.models.protein_gnn import ProteinGNN, compute_all_embeddings

    print(f"Initialising ProteinGNN (mode={args.gnn_mode}, out_dim={args.gnn_out_dim})...")
    gnn_model = ProteinGNN(in_dim=prot_dim, hidden_dim=128, out_dim=args.gnn_out_dim).to(device)
    form_ei = data["protein", "similar", "protein"].edge_index
    role_ei = data["protein", "go_shared", "protein"].edge_index
    cached_embs = None
    if args.gnn_mode == "frozen":
        cached_embs = compute_all_embeddings(gnn_model, data["protein"].x, form_ei, role_ei)
        print(f"GNN embeddings cached: {cached_embs.shape}")

    return {
        "mode": args.gnn_mode,
        "model": gnn_model,
        "protein_x": data["protein"].x,
        "form_ei": form_ei,
        "role_ei": role_ei,
        "cached_embs": cached_embs,
    }, args.gnn_out_dim


def current_gnn_embeddings(gnn_runtime):
    if gnn_runtime is None:
        return None
    if gnn_runtime["mode"] == "frozen":
        return gnn_runtime["cached_embs"]
    return gnn_runtime["model"](
        gnn_runtime["protein_x"],
        gnn_runtime["form_ei"],
        gnn_runtime["role_ei"],
    )


def build_model(args, prot_dim, drug_dim, gnn_emb_dim, go_fp_dim, device):
    if args.model_kind == "global-mean":
        model = GlobalMeanAffinityModel()
    elif args.model_kind == "binding-only":
        model = BindingOnlyAffinityModel(prot_dim, drug_dim)
    elif args.model_kind == "neighbor-transfer":
        model = NeighborTransferModel(prot_dim, drug_dim, go_fp_dim=go_fp_dim)
    elif args.model_kind == "gp":
        # --token-dim repurposed as encoder/attention hidden width for the GP model
        hidden = args.token_dim
        model = GPAffinityModel(prot_dim, drug_dim, hidden_dim=hidden, out_dim=hidden // 2)
    else:
        model = ProteinLigandTNP(
            prot_dim,
            drug_dim,
            token_dim=args.token_dim,
            gnn_emb_dim=gnn_emb_dim,
            go_fp_dim=go_fp_dim,
        )

    model = model.to(device)
    maybe_load_model_warmstart(model, args.warmstart_checkpoint, device)
    set_tnp_train_scope(model, args.train_scope)
    return model


def build_optimizer(args, model, loss_fn, gnn_runtime):
    if args.cold_start_only or args.model_kind == "global-mean":
        return None

    param_groups = []
    model_params = list(model.parameters())
    if model_params:
        param_groups.append({"params": model_params, "lr": args.lr})
    if gnn_runtime is not None and gnn_runtime["mode"] == "trainable":
        param_groups.append({"params": list(gnn_runtime["model"].parameters()), "lr": args.lr})
    if loss_fn is not None:
        param_groups.append({"params": list(loss_fn.parameters()), "lr": args.lr * 10})
    return Adam(param_groups)


def collect_forward_stats(model):
    stats = getattr(model, "last_forward_stats", {})
    return {
        "mu_std": float(stats.get("mu_std", 0.0)),
        "binding_prior_std": float(stats.get("binding_prior_std", 0.0)),
        "log_ppr_alpha": float(stats.get("log_ppr_alpha", 0.0)),
        "centroid_alpha": float(stats.get("centroid_alpha", 0.0)),
        "density": float(stats.get("density", 0.0)),
    }


def optimize_episode(
    args,
    model,
    loss_fn,
    optimizer,
    builder,
    drug_features,
    pillar,
    query_drug_indices,
    labels,
    replay_buffer,
    loader,
    gnn_runtime,
    global_mean_affinity,
    total_episodes,
    episode_idx,
):
    train_loss = float("nan")
    nll_val = float("nan")
    replay_loss_val = float("nan")
    w_nll = float("nan")
    w_listnet = float("nan")
    w_lambda = float("nan")

    if optimizer is None or loss_fn is None:
        return train_loss, nll_val, replay_loss_val, w_nll, w_listnet, w_lambda

    model.train()
    loss_fn.train()
    optimizer.zero_grad()
    loss_fn.step_schedule(episode_idx, total_episodes)

    mu_train, sigma_train, _ = run_episode(
        args,
        model,
        builder,
        drug_features,
        pillar,
        query_drug_indices,
        gnn_runtime=gnn_runtime,
        global_mean_affinity=global_mean_affinity,
    )
    train_result = loss_fn(mu_train, sigma_train, labels.to(query_drug_indices.device))
    train_result["total_loss"].backward()

    train_loss = float(train_result["total_loss"].detach())
    nll_val = float(train_result["nll"].detach())
    w_nll = float(train_result["w_nll"])
    w_listnet = float(train_result["w_listnet"])
    w_lambda = float(train_result["w_lambda"])

    replay_protein_indices = replay_buffer.sample(25)
    replay_batches = sample_replay_batch(loader, replay_protein_indices, replay_edges_per_protein=256)
    if replay_batches:
        replay_total = torch.tensor(0.0, device=query_drug_indices.device)
        for _, replay_edges, replay_labels, replay_pillar in replay_batches:
            replay_mu, replay_sigma, _ = run_episode(
                args,
                model,
                builder,
                drug_features,
                replay_pillar,
                replay_edges[1],
                gnn_runtime=gnn_runtime,
                global_mean_affinity=global_mean_affinity,
            )
            replay_result = loss_fn(replay_mu, replay_sigma, replay_labels.to(query_drug_indices.device))
            replay_total = replay_total + replay_result["total_loss"]
        replay_total = replay_total / len(replay_batches)
        (args.replay_weight * replay_total).backward()
        replay_loss_val = float(replay_total.detach())

    clip_params = list(model.parameters()) + list(loss_fn.parameters())
    if gnn_runtime is not None and gnn_runtime["mode"] == "trainable":
        clip_params += list(gnn_runtime["model"].parameters())
    torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
    optimizer.step()

    return train_loss, nll_val, replay_loss_val, w_nll, w_listnet, w_lambda


def run_episode_gp(
    model: GPAffinityModel,
    gp_builder: DrugFirstContextBuilder,
    drug_features: torch.Tensor,
    pillar: dict,
    query_drug_indices: torch.Tensor,
    global_mean_affinity: float = 6.5,
    frozen_be=None,
):
    target = pillar["target_features"]
    target_idx = int(pillar["target_idx"])
    n_qry = query_drug_indices.size(0)
    device = query_drug_indices.device

    qry_protein = target.unsqueeze(0).expand(n_qry, -1)
    qry_drug = drug_features[query_drug_indices]

    ctx_proteins, ctx_affinities, ctx_mask = gp_builder.build_context(
        target_idx, query_drug_indices, device
    )
    gp_builder.apply_neighborhood_fallback(
        pillar, query_drug_indices, ctx_proteins, ctx_affinities, ctx_mask, device
    )
    prior = compute_prior(frozen_be, qry_protein, qry_drug, global_mean_affinity)
    mu, sigma = model(qry_protein, qry_drug, ctx_proteins, ctx_affinities, ctx_mask, prior)

    stats = collect_forward_stats(model)
    stats["n_ctx"] = int(ctx_mask.float().sum(dim=1).mean().item())
    return mu, sigma, stats


def optimize_episode_gp(
    model: GPAffinityModel,
    loss_fn,
    optimizer,
    gp_builder: DrugFirstContextBuilder,
    drug_features: torch.Tensor,
    pillar: dict,
    query_drug_indices: torch.Tensor,
    labels: torch.Tensor,
    replay_buffer,
    loader,
    global_mean_affinity: float,
    replay_weight: float,
    episode_idx: int,
    total_episodes: int,
    replay_edges_per_protein: int = 256,
    frozen_be=None,
):
    train_loss = nll_val = replay_loss_val = float("nan")
    w_nll = w_listnet = w_lambda = float("nan")

    if optimizer is None or loss_fn is None:
        return train_loss, nll_val, replay_loss_val, w_nll, w_listnet, w_lambda

    model.train()
    loss_fn.train()
    optimizer.zero_grad()
    loss_fn.step_schedule(episode_idx, total_episodes)

    mu, sigma, _ = run_episode_gp(
        model, gp_builder, drug_features, pillar, query_drug_indices, global_mean_affinity,
        frozen_be=frozen_be,
    )
    result = loss_fn(mu, sigma, labels.to(query_drug_indices.device))
    result["total_loss"].backward()

    train_loss = float(result["total_loss"].detach())
    nll_val = float(result["nll"].detach())
    w_nll = float(result["w_nll"])
    w_listnet = float(result["w_listnet"])
    w_lambda = float(result["w_lambda"])

    replay_protein_indices = replay_buffer.sample(25)
    replay_batches = sample_replay_batch(loader, replay_protein_indices, replay_edges_per_protein)
    if replay_batches:
        replay_total = torch.tensor(0.0, device=query_drug_indices.device)
        for _, replay_edges, replay_labels, replay_pillar in replay_batches:
            r_mu, r_sigma, _ = run_episode_gp(
                model, gp_builder, drug_features, replay_pillar, replay_edges[1], global_mean_affinity,
                frozen_be=frozen_be,
            )
            r_result = loss_fn(r_mu, r_sigma, replay_labels.to(query_drug_indices.device))
            replay_total = replay_total + r_result["total_loss"]
        replay_total = replay_total / len(replay_batches)
        (replay_weight * replay_total).backward()
        replay_loss_val = float(replay_total.detach())

    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_fn.parameters()), 1.0)
    optimizer.step()

    return train_loss, nll_val, replay_loss_val, w_nll, w_listnet, w_lambda


def run_episode(
    args,
    model,
    builder,
    drug_features,
    pillar,
    query_drug_indices,
    gnn_runtime=None,
    global_mean_affinity=6.5,
):
    target = pillar["target_features"]
    ppr_centroid = pillar.get("ppr_centroid")
    target_idx = int(pillar["target_idx"])
    n_qry = query_drug_indices.size(0)
    qry_protein = target.unsqueeze(0).expand(n_qry, -1)
    qry_drug = drug_features[query_drug_indices]

    all_gnn_embs = current_gnn_embeddings(gnn_runtime)
    builder.gnn_protein_embs = all_gnn_embs

    qry_gnn_emb = None
    if all_gnn_embs is not None:
        qry_gnn_emb = all_gnn_embs[target_idx].unsqueeze(0).expand(n_qry, -1)

    qry_go_fp = None
    if builder.go_fingerprints is not None:
        qry_go_fp = builder.go_fingerprints[target_idx].unsqueeze(0).expand(n_qry, -1)

    if args.model_kind == "neighbor-transfer":
        (
            nt_neighbor_protein,
            nt_neighbor_drug,
            nt_neighbor_affinity,
            nt_neighbor_ppr,
            nt_neighbor_trust,
            nt_neighbor_mask,
            nt_matched_counts,
            nt_neighbor_go_fp,
        ) = builder.build_neighbor_transfer_context(pillar, query_drug_indices, top_k=args.neighbor_k)
        mu, sigma = model(
            nt_neighbor_protein,
            nt_neighbor_drug,
            nt_neighbor_affinity,
            nt_neighbor_ppr,
            nt_neighbor_trust,
            nt_neighbor_mask,
            qry_protein,
            qry_drug,
            qry_go_fp=qry_go_fp,
            neighbor_go_fp=nt_neighbor_go_fp,
            global_mean_affinity=global_mean_affinity,
        )
        n_ctx = int(round(nt_matched_counts.float().mean().item()))
    elif args.per_query_k > 0:
        pq_p, pq_d, pq_a, pq_ppr, pq_trust, pq_gnn, pq_aff_mean, pq_go_fp = builder.build_per_query_context(
            pillar, query_drug_indices, args.per_query_k
        )
        n_ctx = int(pq_p.size(1))
        if args.model_kind == "tnp":
            mu, sigma = model.forward_per_query(
                pq_p,
                pq_d,
                pq_a,
                pq_ppr,
                pq_trust,
                qry_protein,
                qry_drug,
                pq_aff_mean,
                pq_gnn_emb=pq_gnn,
                qry_gnn_emb=qry_gnn_emb,
                pq_go_fp=pq_go_fp,
                qry_go_fp=qry_go_fp,
                ppr_centroid=ppr_centroid,
            )
        else:
            mu, sigma = model(qry_protein, qry_drug, global_mean_affinity=global_mean_affinity)
    else:
        ctx_p, ctx_d, ctx_a, ctx_ppr, ctx_trust, ctx_gnn, ctx_go_fp = builder.build_context(
            pillar, query_drug_indices
        )
        n_ctx = int(ctx_p.size(0))
        if args.model_kind == "tnp":
            mu, sigma = model(
                ctx_p,
                ctx_d,
                ctx_a,
                qry_protein,
                qry_drug,
                ctx_ppr,
                ctx_trust,
                ppr_centroid=ppr_centroid,
                ctx_gnn_emb=ctx_gnn,
                qry_gnn_emb=qry_gnn_emb,
                ctx_go_fp=ctx_go_fp,
                qry_go_fp=qry_go_fp,
                global_mean_affinity=global_mean_affinity,
            )
        else:
            mu, sigma = model(qry_protein, qry_drug, global_mean_affinity=global_mean_affinity)

    stats = collect_forward_stats(model)
    stats["n_ctx"] = n_ctx
    return mu, sigma, stats


def main():
    parser = build_arg_parser()
    args = apply_presets(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading data from {args.data}...")
    data = torch.load(args.data, weights_only=False).to(device)
    data = merge_activity_edges(data, reduce=args.merge_reduce)

    prot_dim = data["protein"].x.size(1)
    drug_dim = data["drug"].x.size(1)
    drug_features = data["drug"].x
    run_name = args.run_name or default_run_name(args)

    loader = MultiplexPillarSampler(
        data,
        binds_metric="binds_activity",
        priors_cache_path=args.priors,
        temporal_decay=0.0,
        history_mode=args.history_mode,
    )
    replay_buffer = DiverseReplayBuffer(max_size=1000, protein_dim=prot_dim, device=device)
    episodes = build_multiplex_stream(data, binds_metric="binds_activity", min_edges=15, seed=args.seed)
    if args.n_episodes is not None:
        episodes = episodes[:args.n_episodes]
    historical_episodes, stream_episodes = split_stream_episodes(episodes, args.historical_protein_frac)
    print(f"Stream: {len(episodes)} episodes")
    if historical_episodes:
        print(
            f"  Historical seed split: {len(historical_episodes)} proteins ({args.historical_protein_frac:.0%})"
            f" | streamed eval: {len(stream_episodes)} proteins"
        )
    else:
        stream_episodes = episodes

    global_drug_mean = drug_features.mean(0)
    global_mean_affinity = float(data["protein", "binds_activity", "drug"].edge_label.mean())
    print(f"Global mean affinity: {global_mean_affinity:.3f}")

    drug_analog_index = None
    if args.drug_analogs:
        print(f"Building DrugAnalogIndex (top_k={args.analog_top_k})...")
        from src.data.drug_analog_index import DrugAnalogIndex

        drug_analog_index = DrugAnalogIndex(drug_features, top_k=args.analog_top_k)
        print("DrugAnalogIndex ready.")

    gnn_runtime, gnn_emb_dim = build_gnn_runtime(args, data, prot_dim, device)

    go_fingerprints = precompute_go_fingerprints(data) if args.use_go else None
    go_fp_dim = 0 if go_fingerprints is None else int(go_fingerprints.size(1))

    builder = TNPContextBuilder(
        drug_features,
        max_context=args.max_context,
        global_drug_mean=global_drug_mean,
        global_mean_affinity=global_mean_affinity,
        enable_synthetic_prior=args.enable_synthetic_prior,
        drug_analog_index=drug_analog_index,
        gnn_protein_embs=None if gnn_runtime is None else gnn_runtime.get("cached_embs"),
    )
    builder.go_fingerprints = go_fingerprints

    model = build_model(args, prot_dim, drug_dim, gnn_emb_dim, go_fp_dim, device)
    loss_fn = None if args.model_kind == "global-mean" else TNPLoss().to(device)
    optimizer = build_optimizer(args, model, loss_fn, gnn_runtime)

    gp_builder = None
    if args.model_kind == "gp":
        gp_builder = DrugFirstContextBuilder(
            protein_features=data["protein"].x,
            drug_features=drug_features,
            drug_analog_index=drug_analog_index,
            max_k=args.max_context,
            analog_min_sim=0.5,
        )

    frozen_be = None
    if args.model_kind == "gp" and args.historical_protein_frac > 0 and historical_episodes:
        print(f"\nPretraining BindingEncoder on {len(historical_episodes)} historical proteins...")
        all_edges = torch.cat([ep.edges for ep in historical_episodes], dim=1)
        all_labels = torch.cat([ep.labels for ep in historical_episodes])
        frozen_be = pretrain_binding_encoder(
            data["protein"].x, drug_features,
            all_edges, all_labels,
            epochs=args.pretrain_epochs,
            device=device,
            lr=args.pretrain_lr,
            hidden=args.token_dim,
            extra_epochs=args.pretrain_extra_epochs,
        )
        print("BindingEncoder pretrained and frozen.")

        print("Populating GP history with residuals from historical proteins...")
        for ep in historical_episodes:
            prot_feats = data["protein"].x[ep.edges[0]]
            drug_feats_ep = drug_features[ep.edges[1]]
            store_labels = compute_residuals(frozen_be, prot_feats, drug_feats_ep, ep.labels, global_mean_affinity)
            gp_builder.add_revealed(ep.edges, store_labels)
        print(f"GP history populated with {all_labels.size(0)} residual bindings.")

    stream_title = "HISTORICAL-SEEDED PREQUENTIAL STREAM" if historical_episodes else "STARTING STRICT PREQUENTIAL STREAM"
    print(f"\n{stream_title}")
    print(f"  run name:        {run_name}")
    print(f"  model kind:      {args.model_kind}")
    print(f"  history mode:    {args.history_mode}")
    print(f"  train scope:     {args.train_scope}")
    print(f"  gnn mode:        {args.gnn_mode}")
    print(f"  synthetic prior: {args.enable_synthetic_prior}")
    print(f"  drug analogs:    {args.drug_analogs}")
    print(f"  GO fingerprints: {args.use_go} (dim={go_fp_dim})")
    print(f"  per-query-k:     {args.per_query_k} {'(dynamic context)' if args.per_query_k > 0 else '(shared context)'}")
    print(f"  neighbor-k:      {args.neighbor_k}")
    print(f"  historical frac: {args.historical_protein_frac:.0%}")
    print(f"  cold-start only: {args.cold_start_only}")
    print("-" * 72)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    model_path = os.path.join("models", f"{run_name}.pt")
    results_path = os.path.join("results", f"{run_name}.csv")
    summary_path = os.path.join("results", f"{run_name}_cold_start_summary.csv")

    episode_log = []
    ci_history = []
    ef10_history = []
    regime_counts = {"cold": 0, "sparse": 0, "warm": 0}
    roll_window = 100
    train_scope_upgraded = False
    total_episode_budget = len(historical_episodes) + len(stream_episodes)

    if historical_episodes:
        print("\nConsuming historical split before streamed evaluation...")
        for hist_i, ep in enumerate(historical_episodes):
            loader.begin_episode(hist_i)
            if (
                isinstance(model, ProteinLigandTNP)
                and args.train_scope == "head-only"
                and args.unfreeze_after is not None
                and hist_i >= args.unfreeze_after
                and not train_scope_upgraded
            ):
                set_tnp_train_scope(model, "full")
                train_scope_upgraded = True
                print(f"Historical episode {hist_i}: switching TNP train scope from head-only to full")

            pillar = loader.get_pillar_context(ep.protein_idx)
            if gp_builder is not None:
                optimize_episode_gp(
                    model, loss_fn, optimizer, gp_builder, drug_features,
                    pillar, ep.edges[1], ep.labels, replay_buffer, loader,
                    global_mean_affinity, args.replay_weight, hist_i, total_episode_budget,
                    frozen_be=frozen_be,
                )
            else:
                optimize_episode(
                    args, model, loss_fn, optimizer, builder, drug_features,
                    pillar, ep.edges[1], ep.labels, replay_buffer, loader,
                    gnn_runtime, global_mean_affinity, total_episode_budget, hist_i,
                )
            loader.add_revealed_edges(ep.edges, ep.labels)
            if gp_builder is not None and frozen_be is None:
                # When frozen_be is set, gp_builder was pre-populated with residuals above
                gp_builder.add_revealed(ep.edges, ep.labels)
            replay_buffer.add(ep.protein_idx, pillar["target_features"])

            if hist_i % 100 == 0 or hist_i == len(historical_episodes) - 1:
                hist_stats = loader.history_stats()
                print(
                    f"  hist {hist_i + 1:04d}/{len(historical_episodes)}"
                    f" | seeded edges: {hist_stats['revealed_edge_count']}"
                    f" | unique: {hist_stats['unique_revealed_edge_count']}"
                )

        seeded_stats = loader.history_stats()
        print(
            f"Historical split ready: seeded history has {seeded_stats['revealed_edge_count']} edges"
            f" ({seeded_stats['unique_revealed_edge_count']} unique)"
        )

    for i, ep in enumerate(stream_episodes):
        global_episode_idx = len(historical_episodes) + i
        loader.begin_episode(global_episode_idx)
        if (
            isinstance(model, ProteinLigandTNP)
            and args.train_scope == "head-only"
            and args.unfreeze_after is not None
            and global_episode_idx >= args.unfreeze_after
            and not train_scope_upgraded
        ):
            set_tnp_train_scope(model, "full")
            train_scope_upgraded = True
            print(f"Episode {i}: switching TNP train scope from head-only to full")

        pillar = loader.get_pillar_context(ep.protein_idx)
        history_before = loader.history_stats()

        model.eval()
        with torch.no_grad():
            if gp_builder is not None:
                mu_eval, sigma_eval, eval_stats = run_episode_gp(
                    model, gp_builder, drug_features, pillar, ep.edges[1], global_mean_affinity,
                    frozen_be=frozen_be,
                )
            else:
                mu_eval, sigma_eval, eval_stats = run_episode(
                    args, model, builder, drug_features, pillar, ep.edges[1],
                    gnn_runtime=gnn_runtime, global_mean_affinity=global_mean_affinity,
                )

        n_ctx = int(eval_stats["n_ctx"])
        regime = classify_regime(n_ctx)
        regime_counts[regime] += 1
        if args.history_mode == "empty" and not historical_episodes and i == 0 and n_ctx != 0:
            raise RuntimeError(
                f"Episode 0 should be cold under strict history_mode=empty, but n_ctx={n_ctx}. "
                "This indicates preloaded history leakage."
            )

        ci_val = calculate_ci(ep.labels, mu_eval)
        ef10_val = calculate_ef_at_k(ep.labels, mu_eval, k=0.1)
        mean_sigma = float(sigma_eval.mean().detach())

        ci_history.append(ci_val)
        ef10_history.append(ef10_val)
        ci_roll = float(sum(ci_history[-roll_window:]) / len(ci_history[-roll_window:]))
        ef10_roll = float(sum(ef10_history[-roll_window:]) / len(ef10_history[-roll_window:]))

        train_loss = float("nan")
        nll_val = float("nan")
        replay_loss_val = float("nan")
        w_nll = float("nan")
        w_listnet = float("nan")
        w_lambda = float("nan")

        if gp_builder is not None:
            train_loss, nll_val, replay_loss_val, w_nll, w_listnet, w_lambda = optimize_episode_gp(
                model, loss_fn, optimizer, gp_builder, drug_features,
                pillar, ep.edges[1], ep.labels, replay_buffer, loader,
                global_mean_affinity, args.replay_weight, global_episode_idx, total_episode_budget,
                frozen_be=frozen_be,
            )
        else:
            train_loss, nll_val, replay_loss_val, w_nll, w_listnet, w_lambda = optimize_episode(
                args, model, loss_fn, optimizer, builder, drug_features,
                pillar, ep.edges[1], ep.labels, replay_buffer, loader,
                gnn_runtime, global_mean_affinity, total_episode_budget, global_episode_idx,
            )

        loader.add_revealed_edges(ep.edges, ep.labels)
        if gp_builder is not None:
            prot_feats = data["protein"].x[ep.edges[0]]
            drug_feats_ep = drug_features[ep.edges[1]]
            store_labels = compute_residuals(frozen_be, prot_feats, drug_feats_ep, ep.labels, global_mean_affinity)
            gp_builder.add_revealed(ep.edges, store_labels)
        replay_buffer.add(ep.protein_idx, pillar["target_features"])
        history_after = loader.history_stats()

        episode_log.append(
            {
                "episode": i,
                "protein_idx": ep.protein_idx,
                "regime": regime,
                "ci": ci_val,
                "ef10": ef10_val,
                "ci_roll100": ci_roll,
                "ef10_roll100": ef10_roll,
                "mean_sigma": mean_sigma,
                "total_loss": train_loss,
                "nll": nll_val,
                "replay_loss": replay_loss_val,
                "n_ctx": n_ctx,
                "n_preds": ep.labels.numel(),
                "revealed_edge_count": history_before["revealed_edge_count"],
                "unique_revealed_edge_count": history_before["unique_revealed_edge_count"],
                "duplicate_revealed_edges": history_before["duplicate_revealed_edges"],
                "revealed_edge_count_after": history_after["revealed_edge_count"],
                "unique_revealed_edge_count_after": history_after["unique_revealed_edge_count"],
                "mu_std": eval_stats["mu_std"],
                "binding_prior_std": eval_stats["binding_prior_std"],
                "log_ppr_alpha": eval_stats["log_ppr_alpha"],
                "centroid_alpha": eval_stats["centroid_alpha"],
                "density": eval_stats["density"],
                "regime_count_cold": regime_counts["cold"],
                "regime_count_sparse": regime_counts["sparse"],
                "regime_count_warm": regime_counts["warm"],
                "w_nll": w_nll,
                "w_listnet": w_listnet,
                "w_lambda": w_lambda,
            }
        )

        if i % 5 == 0:
            print(
                f"ep {i:04d}/{len(stream_episodes)} [{regime:6s}]"
                f" | CI: {ci_val:.3f} (roll100: {ci_roll:.3f})"
                f" | σ: {mean_sigma:.3f}"
                f" | n_ctx: {n_ctx}"
                f" | history: {history_before['revealed_edge_count']} -> {history_after['revealed_edge_count']}"
                f" | dup: {history_after['duplicate_revealed_edges']}"
                f" | mu_std: {eval_stats['mu_std']:.3f}"
            )

    if args.history_mode == "empty" and not historical_episodes and regime_counts["cold"] + regime_counts["sparse"] == 0:
        raise RuntimeError("Strict prequential run produced only warm episodes; expected cold/sparse regimes as leakage check.")

    if args.model_kind != "global-mean":
        print(f"\nSaving model to {model_path}...")
        torch.save(model.state_dict(), model_path)

    df = pd.DataFrame(episode_log)
    df.to_csv(results_path, index=False)
    print(f"Saved results: {results_path}")

    summary = summarize_cold_start(episode_log)
    if not summary.empty:
        summary.to_csv(summary_path)
        print(f"Saved summary: {summary_path}")
        print("\nCold-start summary:")
        print(summary.to_string())

    print("\nFinal regime counts:")
    for regime_name in ("cold", "sparse", "warm"):
        print(f"  {regime_name}: {regime_counts[regime_name]}")


if __name__ == "__main__":
    main()
