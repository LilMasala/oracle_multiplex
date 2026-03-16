"""
Standalone pretraining script for the Molecular Graph Encoder Prior.

Trains MolGraphPrior on protein-drug binding data using molecular graphs,
then precomputes and saves embedding tables for fast inference.

Usage:
    python scripts/pretrain_mol_graph_prior.py \
      --hetero-data   /path/to/final_graph_data_not_normalized.pt \
      --protein-zip   /path/to/protein_graphs.zip \
      --drug-tar-dir  /path/to/drug_graphs/ \
      --drug-index    /path/to/drug_index.json \
      --output-dir    /path/to/mol_prior/ \
      --hidden        256 \
      --num-layers    4 \
      --epochs        50 \
      --batch-size    256 \
      --lr            1e-3 \
      --bilinear-rank 128 \
      --embed-batch-size 512
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.binds_activity import merge_activity_edges
from src.models.mol_graph_encoder import MolGraphPrior
from src.data.mol_graph_loader import (
    ProteinGraphZipLoader,
    DrugGraphTarLoader,
    MolGraphDataset,
    mol_graph_collate_fn,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load HeteroData
    data = torch.load(args.hetero_data, weights_only=False)
    data = merge_activity_edges(data, reduce="amax")

    ei    = data["protein", "binds_activity", "drug"].edge_index   # [2, N]
    el    = data["protein", "binds_activity", "drug"].edge_label   # [N]
    go_x  = data["go"].x                                           # [N_go, 200]
    pg_ei = data["protein", "relates", "go"].edge_index            # [2, E_pg]

    num_proteins = int(data["protein"].num_nodes)
    num_drugs    = int(data["drug"].num_nodes)
    n_go         = int(go_x.size(0))

    label_offset = float(el.mean())
    shifted_el   = el - label_offset
    print(f"Proteins: {num_proteins}, Drugs: {num_drugs}, GO: {n_go}")
    print(f"Binding edges: {ei.size(1)}, label_offset: {label_offset:.4f}")

    # 2. Build loaders
    idx_to_uniprot = data["protein"].index_to_uniprot_id
    idx_to_chembl  = data["drug"].index_to_chembl_id
    prot_loader = ProteinGraphZipLoader(
        args.protein_zip, data["protein"].uniprot_id_to_index, cache_in_memory=True
    )
    drug_loader = DrugGraphTarLoader(
        args.drug_tar_dir, data["drug"].chembl_id_to_index, args.drug_index
    )

    # 3. Filter to edges where both protein and drug graphs exist
    drug_has_graph = {
        idx for idx, cid in idx_to_chembl.items() if cid in drug_loader._index
    }
    valid = torch.tensor(
        [int(ei[1, i]) in drug_has_graph for i in range(ei.size(1))],
        dtype=torch.bool,
    )
    n_before = ei.size(1)
    ei, shifted_el = ei[:, valid], shifted_el[valid]
    print(f"Training edges after drug-graph filter: {ei.size(1)}/{n_before} "
          f"({100*ei.size(1)/n_before:.1f}%)")

    # Build dataset + dataloader
    dataset = MolGraphDataset(ei, shifted_el, prot_loader, drug_loader,
                              idx_to_uniprot, idx_to_chembl)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=mol_graph_collate_fn, num_workers=0)

    # 4. Model
    model = MolGraphPrior(
        n_go_terms=n_go, hidden=args.hidden, num_layers=args.num_layers,
        bilinear_rank=args.bilinear_rank,
    ).to(device)
    with torch.no_grad():
        model.go_emb.weight.copy_(go_x.to(device))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()

    pg_ei_dev = pg_ei.to(device)

    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            prot_batch = batch["prot_batch"].to(device)
            drug_batch = batch["drug_batch"].to(device)
            prot_ids   = batch["prot_ids"].to(device)
            labels     = batch["labels"].to(device)

            # forward: p is [B, h], d is [B, h]
            with autocast("cuda"):
                p, d = model(prot_batch, drug_batch, pg_ei_dev, num_proteins, prot_ids)
                scores = model.scorer(p, d)
                loss   = F.mse_loss(scores, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            epoch_loss += float(loss.detach()) * bs
            n_seen     += bs

        epoch_loss /= max(n_seen, 1)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch:3d} | MSE={epoch_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

    # 6. Precompute protein embeddings (two-pass)
    print("Precomputing protein embeddings...")
    model.eval()
    raw_prot_embs = torch.zeros(num_proteins, args.hidden)
    for start in tqdm(range(0, num_proteins, args.embed_batch_size)):
        chunk = range(start, min(start + args.embed_batch_size, num_proteins))
        graphs = [prot_loader.get_by_idx(i) for i in chunk]
        pb = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            raw = model.prot_enc(pb.x, pb.edge_index, pb.edge_attr, pb.batch)
        raw_prot_embs[chunk] = raw.cpu()

    with torch.no_grad():
        prot_emb_final = model.go_enr(
            raw_prot_embs.to(device), model.go_emb.weight, pg_ei_dev, num_proteins
        ).cpu()

    # 7. Precompute drug embeddings (chunked -- may take ~30min for 2M drugs)
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
            emb = model.drug_enc(db.x, db.edge_index, db.edge_attr, db.batch)
        drug_emb_final[chunk] = emb.cpu()

    # 8. Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "mol_prior_model.pt"))
    torch.save({
        "prot_emb":      prot_emb_final,
        "drug_emb":      drug_emb_final,
        "label_offset":  label_offset,
        "hidden":        args.hidden,
        "bilinear_rank": args.bilinear_rank,
        "n_go_terms":    n_go,
    }, os.path.join(args.output_dir, "mol_prior_tables.pt"))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pretrain MolGraphPrior on protein-drug binding data"
    )
    parser.add_argument("--hetero-data",      required=True,
                        help="Path to final_graph_data_not_normalized.pt")
    parser.add_argument("--protein-zip",      required=True,
                        help="Path to protein_graphs.zip")
    parser.add_argument("--drug-tar-dir",     required=True,
                        help="Directory containing drug graph tar files")
    parser.add_argument("--drug-index",       default=None,
                        help="Path to drug_index.json (optional)")
    parser.add_argument("--output-dir",       required=True,
                        help="Directory to write mol_prior_model.pt and mol_prior_tables.pt")
    parser.add_argument("--hidden",           type=int, default=256,
                        help="Hidden dimension for GNN layers")
    parser.add_argument("--num-layers",       type=int, default=4,
                        help="Number of GINE layers")
    parser.add_argument("--epochs",           type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size",       type=int, default=256,
                        help="Mini-batch size for training")
    parser.add_argument("--lr",               type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--bilinear-rank",    type=int, default=128,
                        help="Rank of the bilinear scoring head")
    parser.add_argument("--embed-batch-size", type=int, default=512,
                        help="Batch size for precomputing embedding tables")
    main(parser.parse_args())
