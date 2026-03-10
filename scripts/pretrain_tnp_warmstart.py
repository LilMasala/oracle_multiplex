"""
Offline warm-start pretraining for the TNP stack.

Stage 1: pretrain the direct BindingEncoder on historical protein-drug labels.
Stage 2: pretrain the full TNP episodically on proteins using full revealed
history as context and held-out query drugs as the supervision target.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_streaming_exp_tnp import precompute_go_fingerprints, run_episode
from src.data.binds_activity import merge_activity_edges
from src.data.context_builder import TNPContextBuilder
from src.data.multiplex_loader import MultiplexPillarSampler
from src.models.tnp import BindingOnlyAffinityModel, ProteinLigandTNP
from src.protocol.prequential import build_multiplex_stream
from src.training.tnp_loss import TNPLoss


def train_binding_encoder(model, edge_index, edge_label, protein_x, drug_x, steps, batch_size, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    num_edges = edge_label.size(0)

    for step in range(steps):
        take = min(batch_size, num_edges)
        batch_idx = torch.randperm(num_edges, device=device)[:take]
        proteins = protein_x[edge_index[0, batch_idx]]
        drugs = drug_x[edge_index[1, batch_idx]]
        labels = edge_label[batch_idx]

        optimizer.zero_grad()
        mu, _ = model(proteins, drugs, global_mean_affinity=0.0)
        loss = loss_fn(mu, labels)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"binding step {step:04d} | mse={loss.item():.4f}")


def train_tnp(
    model,
    episodes,
    loader,
    builder,
    drug_features,
    global_mean_affinity,
    steps,
    query_batch_size,
    lr,
    device,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = TNPLoss().to(device)
    run_args = SimpleNamespace(model_kind="tnp", per_query_k=0)

    for step in range(steps):
        ep = episodes[step % len(episodes)]
        loader.begin_episode(step)
        pillar = loader.get_pillar_context(ep.protein_idx)
        if ep.labels.numel() > query_batch_size:
            chosen = torch.randperm(ep.labels.numel(), device=device)[:query_batch_size]
            query_drug_indices = ep.edges[1][chosen]
            labels = ep.labels[chosen]
        else:
            query_drug_indices = ep.edges[1]
            labels = ep.labels

        optimizer.zero_grad()
        mu, sigma, stats = run_episode(
            run_args,
            model,
            builder,
            drug_features,
            pillar,
            query_drug_indices,
            gnn_runtime=None,
            global_mean_affinity=global_mean_affinity,
        )
        loss = loss_fn(mu, sigma, labels.to(device))["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(loss_fn.parameters()), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(
                f"tnp step {step:04d}"
                f" | loss={loss.item():.4f}"
                f" | n_ctx={stats['n_ctx']}"
                f" | mu_std={stats['mu_std']:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Offline warm-start pretraining for BindingEncoder/TNP")
    parser.add_argument("--data", default="data/final_graph_data_not_normalized.pt")
    parser.add_argument("--priors", default="data/multiplex_priors.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binding-steps", type=int, default=1000)
    parser.add_argument("--binding-batch-size", type=int, default=1024)
    parser.add_argument("--tnp-steps", type=int, default=1000)
    parser.add_argument("--tnp-query-batch-size", type=int, default=256)
    parser.add_argument("--binding-lr", type=float, default=1e-3)
    parser.add_argument("--tnp-lr", type=float, default=1e-4)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--use-go", action="store_true")
    parser.add_argument("--output-prefix", default="models/tnp_warmstart")
    parser.add_argument("--skip-binding", action="store_true")
    parser.add_argument("--skip-tnp", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = torch.load(args.data, weights_only=False).to(device)
    data = merge_activity_edges(data, reduce="amax")
    protein_x = data["protein"].x
    drug_x = data["drug"].x
    edge_index = data["protein", "binds_activity", "drug"].edge_index
    edge_label = data["protein", "binds_activity", "drug"].edge_label
    global_mean_affinity = float(edge_label.mean())

    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    prot_dim = protein_x.size(1)
    drug_dim = drug_x.size(1)
    binding_model = BindingOnlyAffinityModel(prot_dim, drug_dim).to(device)

    if not args.skip_binding:
        print("\nStage 1: binding encoder pretraining")
        train_binding_encoder(
            binding_model,
            edge_index,
            edge_label,
            protein_x,
            drug_x,
            steps=args.binding_steps,
            batch_size=args.binding_batch_size,
            lr=args.binding_lr,
            device=device,
        )
        binding_path = f"{args.output_prefix}_binding.pt"
        torch.save(binding_model.state_dict(), binding_path)
        print(f"Saved binding warmstart to {binding_path}")

    if not args.skip_tnp:
        print("\nStage 2: episodic TNP pretraining")
        loader = MultiplexPillarSampler(
            data,
            binds_metric="binds_activity",
            priors_cache_path=args.priors,
            history_mode="full",
            temporal_decay=0.0,
        )
        episodes = build_multiplex_stream(data, binds_metric="binds_activity", min_edges=15, seed=args.seed)
        builder = TNPContextBuilder(
            drug_x,
            max_context=args.max_context,
            global_drug_mean=drug_x.mean(0),
            global_mean_affinity=global_mean_affinity,
            enable_synthetic_prior=False,
        )
        if args.use_go:
            builder.go_fingerprints = precompute_go_fingerprints(data)
        tnp_model = ProteinLigandTNP(prot_dim, drug_dim, token_dim=args.token_dim).to(device)
        if not args.skip_binding:
            tnp_model.binding_encoder.load_state_dict(binding_model.binding_encoder.state_dict(), strict=False)
        train_tnp(
            tnp_model,
            episodes,
            loader,
            builder,
            drug_x,
            global_mean_affinity=global_mean_affinity,
            steps=args.tnp_steps,
            query_batch_size=args.tnp_query_batch_size,
            lr=args.tnp_lr,
            device=device,
        )
        tnp_path = f"{args.output_prefix}_tnp.pt"
        torch.save(tnp_model.state_dict(), tnp_path)
        print(f"Saved TNP warmstart to {tnp_path}")


if __name__ == "__main__":
    main()
