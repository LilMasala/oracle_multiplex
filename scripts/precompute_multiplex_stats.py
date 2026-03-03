import argparse
from pathlib import Path

import torch
from torch_geometric.utils import degree, scatter, to_undirected


def _build_binary_adj(num_nodes: int, edge_index: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build dense 0/1 adjacency from edge_index (undirected, no self loops)."""
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def _compute_overlap_jaccard(form_ei: torch.Tensor, role_ei: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    form_adj = _build_binary_adj(num_nodes, form_ei, device)
    role_adj = _build_binary_adj(num_nodes, role_ei, device)
    inter = (form_adj * role_adj).sum(dim=1)
    union = ((form_adj + role_adj) > 0).sum(dim=1).float()
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def _compute_participation(form_ei: torch.Tensor, role_ei: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    form_ei = to_undirected(form_ei, num_nodes=num_nodes)
    role_ei = to_undirected(role_ei, num_nodes=num_nodes)

    deg_form = degree(form_ei[0], num_nodes=num_nodes, dtype=torch.float32).to(device)
    deg_role = degree(role_ei[0], num_nodes=num_nodes, dtype=torch.float32).to(device)
    k_tot = deg_form + deg_role

    frac_sq_sum = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    mask = k_tot > 0
    frac_sq_sum[mask] = (deg_form[mask] / k_tot[mask]).pow(2) + (deg_role[mask] / k_tot[mask]).pow(2)
    # M=2 layers => P_i = 1 - sum_l (k_i^l / k_i)^2
    return 1.0 - frac_sq_sum


def _compute_neighbor_counts(form_ei: torch.Tensor, role_ei: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    form_ei = to_undirected(form_ei, num_nodes=num_nodes)
    role_ei = to_undirected(role_ei, num_nodes=num_nodes)
    edge_cat = torch.cat([form_ei, role_ei], dim=1)
    row, col = edge_cat
    pair_hash = row * num_nodes + col
    uniq_hash = torch.unique(pair_hash)
    uniq_row = torch.div(uniq_hash, num_nodes, rounding_mode="floor")
    ones = torch.ones_like(uniq_row, dtype=torch.float32, device=device)
    return scatter(ones, uniq_row, dim=0, dim_size=num_nodes, reduce="sum")


def _compute_topk_ppr(
    form_ei: torch.Tensor,
    role_ei: torch.Tensor,
    num_nodes: int,
    alpha: float,
    topk: int,
    max_iter: int,
    tol: float,
    batch_size: int,
    device: torch.device,
):
    form_ei = to_undirected(form_ei, num_nodes=num_nodes)
    role_ei = to_undirected(role_ei, num_nodes=num_nodes)
    edge_index = torch.cat([form_ei, role_ei], dim=1)

    row, col = edge_index
    w = torch.ones(row.size(0), device=device, dtype=torch.float32)

    deg = scatter(w, row, dim=0, dim_size=num_nodes, reduce="sum").clamp_min(1.0)
    trans_w = w / deg[row]

    indices = torch.stack([row, col], dim=0)
    transition = torch.sparse_coo_tensor(indices, trans_w, size=(num_nodes, num_nodes), device=device).coalesce()

    topk_idx = torch.empty((num_nodes, topk), dtype=torch.long, device=device)
    topk_score = torch.empty((num_nodes, topk), dtype=torch.float32, device=device)

    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        bsz = end - start

        seed = torch.zeros((num_nodes, bsz), device=device, dtype=torch.float32)
        cols = torch.arange(bsz, device=device)
        seed[start:end, cols] = 1.0

        p = seed.clone()
        for _ in range(max_iter):
            p_next = alpha * seed + (1.0 - alpha) * torch.sparse.mm(transition.t(), p)
            delta = (p_next - p).abs().max()
            p = p_next
            if float(delta) < tol:
                break

        p_rows = p.t()
        vals, idx = torch.topk(p_rows, k=min(topk, num_nodes), dim=1, largest=True, sorted=True)

        if idx.size(1) < topk:
            pad = topk - idx.size(1)
            idx = torch.cat([idx, torch.full((bsz, pad), -1, device=device, dtype=torch.long)], dim=1)
            vals = torch.cat([vals, torch.zeros((bsz, pad), device=device)], dim=1)

        topk_idx[start:end] = idx
        topk_score[start:end] = vals

    return topk_idx, topk_score


def main():
    parser = argparse.ArgumentParser(description="Precompute multiplex trust priors and diffusion cache.")
    parser.add_argument("--graph-path", type=str, default="data/final_graph_data_not_normalized.pt")
    parser.add_argument("--output-path", type=str, default="data/multiplex_priors.pt")
    parser.add_argument("--alpha", type=float, default=0.15, help="PPR restart probability")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    graph_path = Path(args.graph_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = torch.load(graph_path, weights_only=False)
    num_nodes = int(data["protein"].num_nodes)

    form_ei = data["protein", "similar", "protein"].edge_index.to(device)
    role_ei = data["protein", "go_shared", "protein"].edge_index.to(device)

    jaccard = _compute_overlap_jaccard(form_ei, role_ei, num_nodes, device)
    participation = _compute_participation(form_ei, role_ei, num_nodes, device)
    total_neighbors = _compute_neighbor_counts(form_ei, role_ei, num_nodes, device)

    ppr_idx, ppr_scores = _compute_topk_ppr(
        form_ei=form_ei,
        role_ei=role_ei,
        num_nodes=num_nodes,
        alpha=args.alpha,
        topk=args.topk,
        max_iter=args.max_iter,
        tol=args.tol,
        batch_size=args.batch_size,
        device=device,
    )

    cache = {
        "num_nodes": num_nodes,
        "alpha": args.alpha,
        "topk": args.topk,
        "jaccard_overlap": jaccard.cpu(),
        "participation_coeff": participation.cpu(),
        "total_neighbor_count": total_neighbors.cpu(),
        "ppr_topk_indices": ppr_idx.cpu(),
        "ppr_topk_scores": ppr_scores.cpu(),
    }
    torch.save(cache, output_path)
    print(f"Saved multiplex priors to {output_path}")


if __name__ == "__main__":
    main()
