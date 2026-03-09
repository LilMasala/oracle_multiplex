"""
Offline DPMM pre-initialization for BayesianMultiplexRouter.

Runs Pyro SVI on static protein features [z_t, ppr_centroid, static_trust_4]
to initialize DPMM component centroids and Beta variational params before streaming.

Usage:
    micromamba activate oracle
    python scripts/pretrain_dpmm.py \\
        --data data/final_graph_data_not_normalized.pt \\
        --priors data/multiplex_priors.pt \\
        --max-experts 16 \\
        --n-steps 2000 \\
        --output data/dpmm_init.pt
"""

import argparse
import os
import torch
import torch.distributions as td
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from src.models.routing import BayesianMultiplexRouter
from src.data.multiplex_loader import MultiplexPillarSampler


def build_static_obs(data, loader, device):
    """Collect [protein_emb, ppr_centroid, static_trust_4] for all proteins."""
    n_proteins = data["protein"].x.size(0)
    protein_x = data["protein"].x  # [N, protein_dim]

    z_list, ppr_list, trust_list = [], [], []
    for idx in range(n_proteins):
        pillar = loader.get_pillar_context(idx)
        z_list.append(pillar["target_features"])
        prot_dim = protein_x.size(1)
        ppr_centroid = pillar.get("ppr_centroid", torch.zeros(prot_dim, device=device))
        trust_vec = pillar.get("trust_vector", torch.zeros(5, device=device)).float()
        ppr_list.append(ppr_centroid)
        trust_list.append(trust_vec[:4])  # static_trust_4: drop binding_density (index 4)

    z_t = torch.stack(z_list, dim=0).to(device)           # [N, protein_dim]
    ppr_centroid = torch.stack(ppr_list, dim=0).to(device) # [N, protein_dim]
    static_trust_4 = torch.stack(trust_list, dim=0).to(device)  # [N, 4]
    return z_t, ppr_centroid, static_trust_4


def compute_pca(z_t, ppr_centroid, static_trust_4, pca_dim):
    """Fit PCA on raw static obs [N, 2*protein_dim+4] → keep top pca_dim components."""
    raw = torch.cat([z_t, ppr_centroid, static_trust_4], dim=-1).float()  # [N, D]
    mean = raw.mean(dim=0)
    centered = raw - mean
    _, _, V = torch.pca_lowrank(centered, q=pca_dim, niter=4)  # V: [D, pca_dim]
    components = V.T.contiguous()  # [pca_dim, D]
    return mean.cpu(), components.cpu()


def pretrain_dpmm(z_t, ppr_centroid, static_trust_4, max_experts=16, n_steps=2000, lr=1e-3, device="cpu", pca_dim=256):
    protein_dim = z_t.size(1)

    print(f"  Computing PCA: {2 * protein_dim + 4}D → {pca_dim}D ...")
    pca_mean, pca_components = compute_pca(z_t, ppr_centroid, static_trust_4, pca_dim)

    pyro.clear_param_store()
    # drug_dim=1 is a placeholder; it's only used by router_net / forward(), not model/guide obs.
    router = BayesianMultiplexRouter(
        protein_dim=protein_dim,
        drug_dim=1,
        max_experts=max_experts,
        dpmm_init={"pca_mean": pca_mean, "pca_components": pca_components},
    ).to(device)

    svi = SVI(
        router.model,
        router.guide,
        ClippedAdam({"lr": lr, "clip_norm": 5.0}),
        loss=Trace_ELBO(num_particles=4),
    )

    # Dummy full-context tensors for router_net in guide.
    # Must match batch dim N since _router_input uses torch.cat (no broadcast).
    N = z_t.size(0)
    v_dummy = torch.zeros(N, 1, device=device)
    delta_dummy = torch.zeros(N, protein_dim, device=device)
    trust_dummy = torch.zeros(N, 5, device=device)

    print(f"  static_obs_dim = {router.static_obs_dim}  |  N = {z_t.size(0)}")
    for step in range(n_steps):
        loss = svi.step(z_t, ppr_centroid, static_trust_4, v_dummy, delta_dummy, trust_dummy)
        if step % 200 == 0:
            print(f"  step {step:4d}  ELBO loss: {loss:.4f}")

    centroids = router.component_loc.detach().cpu()  # [K, static_obs_dim]
    q_beta_a = router.q_beta_a.detach().cpu()        # [K-1]
    q_beta_b = router.q_beta_b.detach().cpu()        # [K-1]

    with torch.no_grad():
        weights = router.expected_stick_weights().cpu()  # [K]

    # Compute soft assignments for inspection.
    static_obs = router._static_obs(z_t, ppr_centroid, static_trust_4)  # [N, static_obs_dim]
    with torch.no_grad():
        log_probs = dist_log_prob(static_obs, centroids.to(device), router.component_scale.detach())

        log_weights = torch.log(weights.to(device) + 1e-12)

        log_post = log_probs + log_weights
        assignments = log_post.argmax(dim=-1)

    return {
        "centroids": centroids,
        "weights": weights,
        "q_beta_a": q_beta_a,
        "q_beta_b": q_beta_b,
        "assignments": assignments,
        "pca_mean": pca_mean,
        "pca_components": pca_components,
    }


def dist_log_prob(obs, locs, scales):
    """Compute log p(obs | component k) for each k. Returns [N, K]."""
    N = obs.size(0)
    K = locs.size(0)
    log_p = torch.zeros(N, K, device=obs.device)
    for k in range(K):
        log_p[:, k] = td.Normal(locs[k], scales[k]).log_prob(obs).sum(-1)
    return log_p


def main():
    parser = argparse.ArgumentParser(description="Offline DPMM pre-initialization")
    parser.add_argument("--data", default="data/final_graph_data_not_normalized.pt")
    parser.add_argument("--priors", default="data/multiplex_priors.pt")
    parser.add_argument("--max-experts", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pca-dim", type=int, default=256, help="PCA dimensionality for static obs before DPMM")
    parser.add_argument("--output", default="data/dpmm_init.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading graph data from {args.data}...")
    data = torch.load(args.data, weights_only=False).to(device)

    print("Merging binding edge types (pic50, pki, pkd) → binds_activity (max per protein-drug pair)...")
    metrics = ["binds_pic50", "binds_pki", "binds_pkd"]
    all_ei, all_y = [], []
    for m in metrics:
        store = data["protein", m, "drug"]
        all_ei.append(store.edge_index)
        all_y.append(store.edge_label)
    ei_cat = torch.cat(all_ei, dim=1)   # [2, E_total]
    y_cat = torch.cat(all_y, dim=0)     # [E_total]
    num_drugs = data["drug"].x.size(0)
    pair_key = ei_cat[0] * num_drugs + ei_cat[1]
    uniq_keys, inv = torch.unique(pair_key, return_inverse=True)
    y_max = torch.full((uniq_keys.size(0),), float("-inf"), device=y_cat.device)
    y_max.scatter_reduce_(0, inv, y_cat.float(), reduce="amax", include_self=True)
    ei_merged = torch.stack([uniq_keys // num_drugs, uniq_keys % num_drugs], dim=0)
    data["protein", "binds_activity", "drug"].edge_index = ei_merged
    data["protein", "binds_activity", "drug"].edge_label = y_max

    print(f"Building pillar sampler (priors: {args.priors})...")
    loader = MultiplexPillarSampler(data, binds_metric="binds_activity", priors_cache_path=args.priors)

    print("Collecting static protein observations...")
    z_t, ppr_centroid, static_trust_4 = build_static_obs(data, loader, device)
    print(f"  z_t: {z_t.shape}  ppr_centroid: {ppr_centroid.shape}  static_trust_4: {static_trust_4.shape}")

    print(f"\nRunning offline DPMM SVI ({args.n_steps} steps, max_experts={args.max_experts}, pca_dim={args.pca_dim})...")
    result = pretrain_dpmm(
        z_t, ppr_centroid, static_trust_4,
        max_experts=args.max_experts,
        n_steps=args.n_steps,
        lr=args.lr,
        device=str(device),
        pca_dim=args.pca_dim,
    )

    print(f"\nFitted cluster weights (top-5): {result['weights'].topk(5).values.tolist()}")
    print(f"Assignments shape: {result['assignments'].shape}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save({k: v.cpu() for k, v in result.items()}, args.output)
    print(f"\nSaved DPMM init to {args.output}")
    print(f"  centroids:      {result['centroids'].shape}")
    print(f"  weights:        {result['weights'].shape}")
    print(f"  q_beta_a:       {result['q_beta_a'].shape}")
    print(f"  q_beta_b:       {result['q_beta_b'].shape}")
    print(f"  pca_mean:       {result['pca_mean'].shape}")
    print(f"  pca_components: {result['pca_components'].shape}")


if __name__ == "__main__":
    main()
