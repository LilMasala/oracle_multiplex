"""
Standalone pretraining script for the Molecular Graph Encoder Prior.

Trains MolGraphPrior on protein-drug binding data using molecular graphs,
then precomputes and saves embedding tables for fast inference.

Usage (single GPU):
    python scripts/pretrain_mol_graph_prior.py \
      --hetero-data   /path/to/final_graph_data_not_normalized.pt \
      --protein-zip   /path/to/protein_graphs.zip \
      --drug-tar-dir  /path/to/drug_graphs/ \
      --drug-index    /path/to/drug_index.json \
      --output-dir    /path/to/mol_prior/

Usage (multi-GPU, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/pretrain_mol_graph_prior.py ...
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import collections

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lifelines.utils import concordance_index
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.binds_activity import merge_activity_edges
from src.models.mol_graph_encoder import MolGraphPrior, ESMGuidedMolPrior
from src.data.mol_graph_loader import (
    ProteinGraphZipLoader,
    DrugGraphTarLoader,
    MolGraphDataset,
    mol_graph_collate_fn,
)


def compute_training_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    prot_ids: torch.Tensor,
    *,
    loss_type: str,
    mse_weight: float,
    rank_weight: float,
    rank_margin: float,
) -> dict:
    mse_loss = F.mse_loss(scores, labels)
    zero = scores.new_zeros(())

    if loss_type == "mse":
        return {
            "total": mse_loss,
            "mse": mse_loss.detach(),
            "rank": zero.detach(),
            "n_pairs": 0,
        }

    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
    same_protein = prot_ids.unsqueeze(1) == prot_ids.unsqueeze(0)
    upper = torch.triu(torch.ones_like(same_protein, dtype=torch.bool), diagonal=1)
    pair_mask = same_protein & upper & (label_diff.abs() >= rank_margin)

    if pair_mask.any():
        target_sign = label_diff[pair_mask].sign()
        pred_margin = score_diff[pair_mask] * target_sign
        pair_weights = label_diff[pair_mask].abs()
        rank_terms = F.softplus(-pred_margin)
        rank_loss = (rank_terms * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
        n_pairs = int(pair_mask.sum().item())
    else:
        rank_loss = zero
        n_pairs = 0

    if loss_type == "bpr":
        total = rank_loss
    elif loss_type == "hybrid":
        total = mse_weight * mse_loss + rank_weight * rank_loss
    else:
        raise ValueError(f"Unknown loss_type={loss_type}")

    return {
        "total": total,
        "mse": mse_loss.detach(),
        "rank": rank_loss.detach(),
        "n_pairs": n_pairs,
    }


def summarize_label_split(name: str, edge_index: torch.Tensor, edge_label: torch.Tensor,
                          *, label_offset: float, label_std: float) -> str:
    if edge_label is None or edge_label.numel() == 0:
        return f"{name}: empty"

    raw = edge_label * label_std + label_offset
    proteins = torch.unique(edge_index[0]).numel()
    positives = int((raw > 6.0).sum().item())
    quantiles = torch.quantile(
        raw.float(),
        torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float),
    )
    return (
        f"{name}: edges={edge_label.numel()} proteins={proteins} "
        f"raw_mean={raw.mean().item():.3f} raw_std={raw.std().clamp_min(0.0).item():.3f} "
        f"min={raw.min().item():.3f} q05={quantiles[0].item():.3f} q25={quantiles[1].item():.3f} "
        f"median={quantiles[2].item():.3f} q75={quantiles[3].item():.3f} q95={quantiles[4].item():.3f} "
        f"max={raw.max().item():.3f} gt6={positives}/{edge_label.numel()} "
        f"({100.0 * positives / edge_label.numel():.1f}%)"
    )


def setup(args, rank: int, local_rank: int, world_size: int):
    """
    Load data, loaders, dataset, and model. Returns a state dict that can be
    passed to train() and save_embeddings() independently — useful in notebooks
    to avoid reloading the 60GB drug cache between training runs.
    """
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(f"Device: {device}  (world_size={world_size})")

    # 1. Load HeteroData
    data = torch.load(args.hetero_data, weights_only=False)
    data = merge_activity_edges(data, reduce="amax")

    ei = data["protein", "binds_activity", "drug"].edge_index
    el = data["protein", "binds_activity", "drug"].edge_label

    num_proteins = int(data["protein"].num_nodes)
    num_drugs    = int(data["drug"].num_nodes)

    label_offset = float(el.mean())
    label_std    = float(el.std().clamp(min=1e-6))
    shifted_el   = (el - label_offset) / label_std
    if rank == 0:
        print(f"Proteins: {num_proteins}, Drugs: {num_drugs}")
        print(f"Binding edges: {ei.size(1)}, label_offset: {label_offset:.4f}, label_std: {label_std:.4f}")

    # 2. Build loaders
    idx_to_uniprot = data["protein"].index_to_uniprot_id
    idx_to_chembl  = data["drug"].index_to_chembl_id
    prot_loader = ProteinGraphZipLoader(
        args.protein_zip, data["protein"].uniprot_id_to_index, cache_in_memory=True
    )
    drug_loader = DrugGraphTarLoader(
        args.drug_tar_dir, data["drug"].chembl_id_to_index, args.drug_index,
        cache_in_memory=(args.drug_packed_cache is None),
        packed_cache_path=args.drug_packed_cache,
    )

    # 3. Filter to edges where drug graphs exist
    _available = drug_loader._graph_cache if drug_loader._graph_cache else drug_loader._index
    drug_has_graph = {
        idx for idx, cid in idx_to_chembl.items() if cid in _available
    }
    valid = torch.tensor(
        [int(ei[1, i]) in drug_has_graph for i in range(ei.size(1))],
        dtype=torch.bool,
    )
    n_before = ei.size(1)
    ei, shifted_el = ei[:, valid], shifted_el[valid]
    if rank == 0:
        print(f"Training edges after drug-graph filter: {ei.size(1)}/{n_before} "
              f"({100*ei.size(1)/n_before:.1f}%)")

    # 3b. Filter to historical proteins only (prevents leakage into streaming eval set)
    val_ei, val_el = None, None
    if args.historical_protein_frac > 0.0:
        from src.protocol.prequential import build_multiplex_stream
        _all_eps = build_multiplex_stream(
            data, binds_metric="binds_activity",
            min_edges=args.stream_min_edges, seed=args.stream_seed,
        )
        _hist_count = int(len(_all_eps) * args.historical_protein_frac)
        hist_prot_set = {int(ep.protein_idx) for ep in _all_eps[:_hist_count]}
        if rank == 0:
            print(f"Historical protein filter: {len(hist_prot_set)} proteins "
                  f"({args.historical_protein_frac:.0%} of {len(_all_eps)}-episode stream)")
        hist_mask = torch.tensor(
            [int(ei[0, i]) in hist_prot_set for i in range(ei.size(1))], dtype=torch.bool
        )
        val_ei = ei[:, ~hist_mask]
        val_el = shifted_el[~hist_mask]
        ei = ei[:, hist_mask]
        shifted_el = shifted_el[hist_mask]
        if rank == 0:
            print(f"Training edges after historical filter: {ei.size(1)}")
            print(f"Validation edges (streaming proteins): {val_ei.size(1)}")
            print(summarize_label_split(
                "Train label distribution", ei, shifted_el,
                label_offset=label_offset, label_std=label_std,
            ))
            print(summarize_label_split(
                "Val label distribution", val_ei, val_el,
                label_offset=label_offset, label_std=label_std,
            ))
    elif rank == 0:
        print(summarize_label_split(
            "Train label distribution", ei, shifted_el,
            label_offset=label_offset, label_std=label_std,
        ))

    dataset = MolGraphDataset(ei, shifted_el, prot_loader, drug_loader,
                              idx_to_uniprot, idx_to_chembl)

    protein_x = data["protein"].x.cpu()
    if args.scorer == "esm_cross_attn":
        model_core = ESMGuidedMolPrior(
            esm_dim=protein_x.size(1),
            hidden=args.hidden,
            num_layers=args.num_layers,
        ).to(device)
    else:
        model_core = MolGraphPrior(
            hidden=args.hidden, num_layers=args.num_layers,
            bilinear_rank=args.bilinear_rank,
            scorer=args.scorer,
        ).to(device)
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model_core.parameters()):,}")

    model = DDP(model_core, device_ids=[local_rank], find_unused_parameters=True)

    return dict(
        device=device, rank=rank, world_size=world_size,
        model=model, model_core=model_core,
        dataset=dataset,
        prot_loader=prot_loader, drug_loader=drug_loader,
        drug_has_graph=drug_has_graph,
        num_proteins=num_proteins, num_drugs=num_drugs,
        label_offset=label_offset, label_std=label_std,
        idx_to_uniprot=idx_to_uniprot, idx_to_chembl=idx_to_chembl,
        train_ei=ei, train_el=shifted_el,
        val_ei=val_ei, val_el=val_el,
        protein_x=protein_x,
    )


def _eval_ranking_metrics(model_core, prot_loader, drug_loader, edge_index, edge_label, device,
                          embed_batch_size, label_offset=0.0, label_std=1.0,
                          protein_x=None, max_proteins=None, prefix="Val"):
    """
    Compute per-protein ranking metrics on a protein-drug edge set.
    Reports unweighted and n_pairs-weighted CI/Spearman, plus AUROC/AUPRC at pX>6.
    Labels in edge_label are standardized; raw pX = label * label_std + label_offset.
    """
    model_core.eval()
    # Binarization threshold in standardized label space
    binary_thresh = (6.0 - label_offset) / label_std

    prot_to_pairs = collections.defaultdict(list)
    for i in range(edge_index.size(1)):
        prot_to_pairs[int(edge_index[0, i])].append((int(edge_index[1, i]), float(edge_label[i])))
    eligible = {p: pairs for p, pairs in prot_to_pairs.items() if len(pairs) >= 5}
    if not eligible:
        return ""
    if max_proteins is not None and len(eligible) > max_proteins:
        limited_keys = sorted(eligible.keys())[:max_proteins]
        eligible = {k: eligible[k] for k in limited_keys}

    cis, ci_w = [], []
    rhos, rho_w = [], []
    aurocs, auprs = [], []

    def _accumulate(all_scores, all_labels, n):
        try:
            cis.append(concordance_index(all_labels, all_scores))
            ci_w.append(n)
        except ZeroDivisionError:
            pass
        rho, _ = spearmanr(all_scores, all_labels)
        if not np.isnan(rho):
            rhos.append(rho)
            rho_w.append(n)
        y_bin = [1 if l > binary_thresh else 0 for l in all_labels]
        if len(set(y_bin)) == 2:  # need both classes for ROC/PR
            aurocs.append(roc_auc_score(y_bin, all_scores))
            auprs.append(average_precision_score(y_bin, all_scores))

    if model_core.scorer_type == "esm_cross_attn":
        for p_idx, pairs in eligible.items():
            d_idxs = [d for d, _ in pairs]
            labels = [l for _, l in pairs]
            esm_1 = protein_x[p_idx].unsqueeze(0).to(device)
            all_scores, all_labels = None, []
            for start in range(0, len(d_idxs), embed_batch_size):
                chunk_d = d_idxs[start: start + embed_batch_size]
                chunk_l = labels[start: start + embed_batch_size]
                drug_graphs = [drug_loader.get_by_idx(d) for d in chunk_d]
                esm_chunk = esm_1.expand(len(chunk_d), -1)
                db = Batch.from_data_list(drug_graphs).to(device)
                with torch.no_grad():
                    chunk_scores = model_core(esm_chunk, db).cpu().numpy()
                all_scores = chunk_scores if all_scores is None else np.concatenate([all_scores, chunk_scores])
                all_labels += chunk_l
            _accumulate(all_scores, all_labels, len(all_labels))
    elif model_core.scorer_type == "node_cross_attn":
        for p_idx, pairs in eligible.items():
            d_idxs = [d for d, _ in pairs]
            labels = [l for _, l in pairs]
            prot_graph = prot_loader.get_by_idx(p_idx)
            all_scores, all_labels = None, []
            for start in range(0, len(d_idxs), embed_batch_size):
                chunk_d = d_idxs[start: start + embed_batch_size]
                chunk_l = labels[start: start + embed_batch_size]
                drug_graphs = [drug_loader.get_by_idx(d) for d in chunk_d]
                pb = Batch.from_data_list([prot_graph] * len(chunk_d)).to(device)
                db = Batch.from_data_list(drug_graphs).to(device)
                with torch.no_grad():
                    p_emb, d_emb = model_core.encode(pb, db)
                    chunk_scores = model_core.scorer(p_emb, d_emb).cpu().numpy()
                all_scores = chunk_scores if all_scores is None else np.concatenate([all_scores, chunk_scores])
                all_labels += chunk_l
            _accumulate(all_scores, all_labels, len(all_labels))
    else:
        # Precompute all protein and drug embeddings independently (bilinear / cross_attn).
        prot_indices = list(eligible.keys())
        prot_emb_map = {}
        for start in range(0, len(prot_indices), embed_batch_size):
            chunk = prot_indices[start: start + embed_batch_size]
            graphs = [prot_loader.get_by_idx(i) for i in chunk]
            pb = Batch.from_data_list(graphs).to(device)
            with torch.no_grad():
                embs = model_core.prot_enc(pb.x, pb.edge_index, pb.edge_attr, pb.batch)
            for idx, emb in zip(chunk, embs):
                prot_emb_map[idx] = emb

        drug_indices = list({d for pairs in eligible.values() for d, _ in pairs})
        drug_emb_map = {}
        for start in range(0, len(drug_indices), embed_batch_size):
            chunk = drug_indices[start: start + embed_batch_size]
            graphs = [drug_loader.get_by_idx(i) for i in chunk]
            db = Batch.from_data_list(graphs).to(device)
            with torch.no_grad():
                embs = model_core.drug_enc(db.x, db.edge_index, db.edge_attr, db.batch)
            for idx, emb in zip(chunk, embs):
                drug_emb_map[idx] = emb

        for p_idx, pairs in eligible.items():
            d_idxs = [d for d, _ in pairs]
            labels = [l for _, l in pairs]
            p_emb   = prot_emb_map[p_idx].unsqueeze(0).expand(len(d_idxs), -1)
            d_stack = torch.stack([drug_emb_map[d] for d in d_idxs])
            with torch.no_grad():
                scores = model_core.scorer(p_emb, d_stack).cpu().numpy()
            _accumulate(scores, labels, len(labels))

    if not cis:
        return ""
    ci_uw  = np.mean(cis)
    ci_w_  = np.average(cis, weights=ci_w)
    rho_uw = np.mean(rhos) if rhos else float("nan")
    rho_w_ = np.average(rhos, weights=rho_w) if rhos else float("nan")
    auroc  = np.mean(aurocs) if aurocs else float("nan")
    aupr   = np.mean(auprs)  if auprs  else float("nan")
    return (f" | {prefix}CI={ci_uw:.4f}(uw)/{ci_w_:.4f}(w)"
            f" Sp={rho_uw:.4f}(uw)/{rho_w_:.4f}(w)"
            f" AUROC={auroc:.4f} AUPR={aupr:.4f}"
            f" (n={len(cis)})")


def train(state: dict, args):
    """Run the training loop on a pre-built state (from setup())."""
    device     = state["device"]
    rank       = state["rank"]
    world_size = state["world_size"]
    model      = state["model"]       # DDP wrapper
    model_core = state["model_core"]  # unwrapped, for scorer / eval
    dataset    = state["dataset"]

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=mol_graph_collate_fn, num_workers=0)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)  # ensures different shuffle per epoch across ranks
        model.train()
        epoch_loss, n_seen = 0.0, 0
        epoch_mse, epoch_rank = 0.0, 0.0
        epoch_pairs = 0
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False, disable=(rank != 0))):
            optimizer.zero_grad()
            drug_batch = batch["drug_batch"].to(device)
            labels     = batch["labels"].to(device)
            prot_ids   = batch["prot_ids"].to(device)
            if model_core.scorer_type == "esm_cross_attn":
                esm_emb = state["protein_x"][batch["prot_ids"]].to(device)
                scores = model(esm_emb, drug_batch)
            else:
                prot_batch = batch["prot_batch"].to(device)
                scores = model(prot_batch, drug_batch)
            loss_dict = compute_training_loss(
                scores,
                labels,
                prot_ids,
                loss_type=args.loss_type,
                mse_weight=args.mse_weight,
                rank_weight=args.rank_weight,
                rank_margin=args.rank_margin,
            )
            loss = loss_dict["total"]

            loss_val = float(loss.detach())

            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"\n[NaN/Inf] epoch={epoch} step={step} loss={loss_val}")
                    print(f"  scores: min={scores.min():.3f} max={scores.max():.3f} nan={scores.isnan().any()}")
                    print(f"  labels: min={labels.min():.3f} max={labels.max():.3f} nan={labels.isnan().any()}")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if rank == 0 and step % 100 == 0:
                print(
                    f"  ep={epoch} step={step} loss={loss_val:.4f} "
                    f"mse={float(loss_dict['mse']):.4f} rank={float(loss_dict['rank']):.4f} "
                    f"pairs={loss_dict['n_pairs']}",
                    flush=True,
                )

            bs = labels.size(0)
            epoch_loss += loss_val * bs
            epoch_mse += float(loss_dict["mse"]) * bs
            epoch_rank += float(loss_dict["rank"]) * bs
            epoch_pairs += int(loss_dict["n_pairs"])
            n_seen     += bs

        # Average epoch loss across all ranks
        metric_tensor = torch.tensor(
            [
                epoch_loss / max(n_seen, 1),
                epoch_mse / max(n_seen, 1),
                epoch_rank / max(n_seen, 1),
                float(epoch_pairs),
            ],
            device=device,
        )
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = metric_tensor[0].item() / world_size
        epoch_mse = metric_tensor[1].item() / world_size
        epoch_rank = metric_tensor[2].item() / world_size
        epoch_pairs = int(metric_tensor[3].item())

        scheduler.step(epoch_loss)

        if rank == 0:
            train_metric_str = ""
            val_metric_str = ""
            train_ei = state.get("train_ei")
            train_el = state.get("train_el")
            val_ei = state.get("val_ei")
            val_el = state.get("val_el")
            if epoch % args.eval_every == 0:
                if train_ei is not None and train_ei.size(1) > 0:
                    train_metric_str = _eval_ranking_metrics(
                        model_core, state["prot_loader"], state["drug_loader"],
                        train_ei, train_el, device, args.embed_batch_size,
                        label_offset=state["label_offset"], label_std=state["label_std"],
                        protein_x=state.get("protein_x"),
                        max_proteins=args.train_eval_max_proteins,
                        prefix="Train ",
                    )
            if val_ei is not None and val_ei.size(1) > 0 and epoch % args.eval_every == 0:
                val_metric_str = _eval_ranking_metrics(
                    model_core, state["prot_loader"], state["drug_loader"],
                    val_ei, val_el, device, args.embed_batch_size,
                    label_offset=state["label_offset"], label_std=state["label_std"],
                    protein_x=state.get("protein_x"),
                    max_proteins=args.val_eval_max_proteins,
                    prefix="Val ",
                )
            print(
                f"Epoch {epoch:3d} | loss={epoch_loss:.4f} | mse={epoch_mse:.4f} "
                f"| rank={epoch_rank:.4f} | pairs={epoch_pairs} "
                f"| lr={optimizer.param_groups[0]['lr']:.2e}"
                  + train_metric_str + val_metric_str)

        # All ranks wait for rank 0 to finish evaluation before starting the next epoch.
        # Without this, ranks 1-3 enter the next DDP forward while rank 0 is still in
        # _eval_streaming_spearman, causing NCCL buffer-sync timeout.
        dist.barrier()

        model.train()


def save_embeddings(state: dict, args):
    """Precompute and save protein/drug embedding tables after training. Rank 0 only."""
    if state["rank"] != 0:
        return

    device        = state["device"]
    model_core    = state["model_core"]
    label_offset  = state["label_offset"]
    label_std     = state["label_std"]

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model_core.state_dict(), os.path.join(args.output_dir, "mol_prior_model.pt"))

    if args.scorer in ("node_cross_attn", "esm_cross_attn"):
        # No global precomputation: drug embeddings are protein-dependent.
        # GNNPrior will run the model per-episode at inference time.
        torch.save({
            "label_offset":  label_offset,
            "label_std":     label_std,
            "hidden":        args.hidden,
            "num_layers":    args.num_layers,
            "bilinear_rank": args.bilinear_rank,
            "scorer":        args.scorer,
        }, os.path.join(args.output_dir, "mol_prior_tables.pt"))
        print(f"Saved model weights (no embedding tables for {args.scorer})")
        return

    prot_loader   = state["prot_loader"]
    drug_loader   = state["drug_loader"]
    drug_has_graph = state["drug_has_graph"]
    num_proteins  = state["num_proteins"]
    num_drugs     = state["num_drugs"]

    print("Precomputing protein embeddings...")
    model_core.eval()
    prot_emb_final = torch.zeros(num_proteins, args.hidden)
    for start in tqdm(range(0, num_proteins, args.embed_batch_size)):
        chunk = range(start, min(start + args.embed_batch_size, num_proteins))
        graphs = [prot_loader.get_by_idx(i) for i in chunk]
        pb = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            emb = model_core.prot_enc(pb.x, pb.edge_index, pb.edge_attr, pb.batch)
        prot_emb_final[chunk] = emb.cpu()

    print("Precomputing drug embeddings...")
    drug_emb_final = torch.zeros(num_drugs, args.hidden)
    for start in tqdm(range(0, num_drugs, args.embed_batch_size)):
        chunk = [i for i in range(start, min(start + args.embed_batch_size, num_drugs))
                 if i in drug_has_graph]
        if not chunk:
            continue
        graphs = [drug_loader.get_by_idx(i) for i in chunk]
        db = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            emb = model_core.drug_enc(db.x, db.edge_index, db.edge_attr, db.batch)
        drug_emb_final[chunk] = emb.cpu()

    torch.save({
        "prot_emb":      prot_emb_final,
        "drug_emb":      drug_emb_final,
        "label_offset":  label_offset,
        "label_std":     label_std,
        "hidden":        args.hidden,
        "num_layers":    args.num_layers,
        "bilinear_rank": args.bilinear_rank,
        "scorer":        args.scorer,
    }, os.path.join(args.output_dir, "mol_prior_tables.pt"))
    print(f"Saved to {args.output_dir}")


def main(args):
    # Initialize process group (noop if not launched with torchrun)
    _timeout = datetime.timedelta(hours=4)
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl", timeout=_timeout)
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Single-GPU fallback
        local_rank = 0
        rank       = 0
        world_size = 1
        # Wrap in a no-op process group so the rest of the code is uniform
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo", rank=0, world_size=1)

    torch.cuda.set_device(local_rank)

    state = setup(args, rank, local_rank, world_size)
    train(state, args)
    dist.barrier()  # ensure all ranks finish training before rank 0 saves
    save_embeddings(state, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain MolGraphPrior on protein-drug binding data"
    )
    parser.add_argument("--hetero-data",        required=True)
    parser.add_argument("--protein-zip",         required=True)
    parser.add_argument("--drug-tar-dir",        required=True)
    parser.add_argument("--drug-index",          default=None)
    parser.add_argument("--drug-packed-cache",   default=None,
                        help="Path to drug_graphs_packed.pt (faster than tar loading)")
    parser.add_argument("--output-dir",          required=True)
    parser.add_argument("--hidden",              type=int, default=256)
    parser.add_argument("--num-layers",          type=int, default=4)
    parser.add_argument("--epochs",              type=int, default=50)
    parser.add_argument("--batch-size",          type=int, default=256)
    parser.add_argument("--lr",                  type=float, default=1e-3)
    parser.add_argument("--bilinear-rank",       type=int, default=128)
    parser.add_argument("--embed-batch-size",    type=int, default=512)
    parser.add_argument("--loss-type",           default="hybrid",
                        choices=["mse", "bpr", "hybrid"],
                        help="Training loss: mse, bpr, or hybrid (mse + within-protein BPR)")
    parser.add_argument("--mse-weight",          type=float, default=1.0,
                        help="Weight on the regression term when --loss-type hybrid")
    parser.add_argument("--rank-weight",         type=float, default=1.0,
                        help="Weight on the within-protein BPR term when --loss-type hybrid")
    parser.add_argument("--rank-margin",         type=float, default=0.25,
                        help="Minimum standardized label gap required to form a ranking pair")
    parser.add_argument("--train-eval-max-proteins", type=int, default=512,
                        help="Max number of train proteins to score during periodic train CI eval")
    parser.add_argument("--val-eval-max-proteins",   type=int, default=None,
                        help="Optional cap on validation proteins scored during periodic eval")
    parser.add_argument("--historical-protein-frac", type=float, default=0.0,
                        help="Fraction of stream proteins to train on (0.0=all). "
                             "Must match --historical-protein-frac in run_streaming_exp_tnp.py")
    parser.add_argument("--stream-seed",         type=int, default=42,
                        help="Protein stream shuffle seed (must match --seed in streaming exp)")
    parser.add_argument("--stream-min-edges",    type=int, default=15,
                        help="min_edges for build_multiplex_stream (must match streaming exp)")
    parser.add_argument("--scorer",              default="bilinear",
                        choices=["bilinear", "cross_attn", "node_cross_attn", "esm_cross_attn"],
                        help="Scoring head: bilinear (default), cross_attn, node_cross_attn, or esm_cross_attn")
    parser.add_argument("--eval-every",           type=int, default=5,
                        help="Run streaming val eval every N epochs (default: 5). "
                             "node_cross_attn eval is slow (~1hr for 2k proteins); set higher if needed.")
    main(parser.parse_args())
