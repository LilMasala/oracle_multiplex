"""
Tiny overfit sanity check for the TNP attention path.

This is intentionally small and deterministic: if the model cannot drive the
toy 5-drug ranking task close to CI=1.0, the issue is structural or
implementation-level rather than dataset difficulty.
"""

from __future__ import annotations

import torch

from src.models.tnp import ProteinLigandTNP
from src.training.metrics import calculate_ci
from src.training.tnp_loss import TNPLoss


PROT_DIM = 8
DRUG_DIM = 4
TOKEN_DIM = 16
NHEAD = 2
NLAYERS = 1
N_CTX = 5
N_QRY = 5
N_STEPS = 300

CTX_AFF = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0])
QRY_AFF = torch.tensor([5.5, 6.5, 7.5, 8.5, 9.5])


def build_toy_batch(seed: int = 0):
    torch.manual_seed(seed)
    prot_0 = torch.randn(PROT_DIM)
    prot_1 = torch.randn(PROT_DIM)
    drugs = torch.randn(N_CTX, DRUG_DIM)

    ctx_protein = prot_0.unsqueeze(0).expand(N_CTX, -1)
    ctx_drug = drugs
    ctx_affinity = CTX_AFF.unsqueeze(1)
    ctx_ppr = torch.tensor([0.40, 0.30, 0.20, 0.10, 0.05])
    ctx_trust = torch.ones(N_CTX)
    qry_protein = prot_1.unsqueeze(0).expand(N_QRY, -1)
    qry_drug = drugs
    labels = QRY_AFF
    return ctx_protein, ctx_drug, ctx_affinity, ctx_ppr, ctx_trust, qry_protein, qry_drug, labels


def run_diagnostic(n_steps: int = N_STEPS, lr: float = 3e-3, seed: int = 0, verbose: bool = True):
    torch.manual_seed(seed)
    (
        ctx_protein,
        ctx_drug,
        ctx_affinity,
        ctx_ppr,
        ctx_trust,
        qry_protein,
        qry_drug,
        labels,
    ) = build_toy_batch(seed=seed)

    model = ProteinLigandTNP(
        protein_dim=PROT_DIM,
        drug_dim=DRUG_DIM,
        token_dim=TOKEN_DIM,
        nhead=NHEAD,
        num_layers=NLAYERS,
        dropout=0.0,
    )
    loss_fn = TNPLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=lr)

    checkpoints = {0, 1, 2, 10, 50, 100, n_steps - 1}
    for step in range(n_steps):
        optimizer.zero_grad()
        mu, sigma = model(
            ctx_protein,
            ctx_drug,
            ctx_affinity,
            qry_protein,
            qry_drug,
            ctx_ppr,
            ctx_trust,
        )
        loss = loss_fn(mu, sigma, labels)["total_loss"]
        loss.backward()
        optimizer.step()

        if verbose and step in checkpoints:
            ci = calculate_ci(labels, mu.detach())
            print(
                f"step {step:03d}"
                f" | loss={loss.item():.4f}"
                f" | ci={ci:.4f}"
                f" | mu_std={model.last_forward_stats.get('mu_std', 0.0):.4f}"
                f" | binding_prior_std={model.last_forward_stats.get('binding_prior_std', 0.0):.4f}"
                f" | log_ppr_alpha={model.last_forward_stats.get('log_ppr_alpha', 0.0):+.4f}"
            )

    model.eval()
    with torch.no_grad():
        final_mu, _ = model(
            ctx_protein,
            ctx_drug,
            ctx_affinity,
            qry_protein,
            qry_drug,
            ctx_ppr,
            ctx_trust,
        )
    final_ci = calculate_ci(labels, final_mu.detach())
    summary = {
        "final_ci": float(final_ci),
        "final_mu": final_mu.detach().cpu(),
        "log_ppr_alpha": model.last_forward_stats.get("log_ppr_alpha", 0.0),
        "binding_prior_std": model.last_forward_stats.get("binding_prior_std", 0.0),
    }
    if verbose:
        print("\nFinal overfit sanity check")
        print(f"  final_ci={summary['final_ci']:.4f}")
        print(f"  final_mu={summary['final_mu'].tolist()}")
        print(f"  log_ppr_alpha={summary['log_ppr_alpha']:+.4f}")
        print("  verdict: PASS" if final_ci >= 0.95 else "  verdict: FAIL")
    return summary


if __name__ == "__main__":
    run_diagnostic()
