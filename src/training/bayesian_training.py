import pyro
from pyro.infer import TraceMeanField_ELBO


def build_router_elbo(num_particles=1):
    """Create ELBO object used in unified differentiable optimization."""
    pyro.clear_param_store()
    return TraceMeanField_ELBO(num_particles=num_particles)


def train_stream_step_with_svi(
    router,
    elbo,
    optimizer,
    ebl_loss_fn,
    z_refined,
    protein_raw_features,
    v_prior,
    query_drug_features,
    trust_vector,
    labels,
    elbo_weight=1.0,
):
    """
    Unified Option-B update with one optimizer:
      total_loss = supervised_ebl_loss + elbo_weight * mean_elbo_loss

    The ELBO is normalized by the number of protein-level contexts in the step
    so its scale does not grow with drug query batch size.
    """
    optimizer.zero_grad(set_to_none=True)

    elbo_loss = elbo.differentiable_loss(
        router.model,
        router.guide,
        protein_raw_features,
        v_prior,
        trust_vector,
    )
    num_context = protein_raw_features.shape[0] if protein_raw_features.dim() > 1 else 1
    mean_elbo_loss = elbo_loss / max(num_context, 1)

    final_scores, gate_probs, expert_tensor = router(
        z_refined=z_refined,
        protein_raw_features=protein_raw_features,
        v_prior=v_prior,
        query_drug_features=query_drug_features,
        trust_vector=trust_vector,
    )
    loss_dict = ebl_loss_fn(
        final_preds=final_scores,
        true_labels=labels,
        gate_probs=gate_probs,
        expert_tensor=expert_tensor,
        protein_level_gate=True,
    )

    total_loss = loss_dict["total_loss"] + (elbo_weight * mean_elbo_loss)
    total_loss.backward()
    optimizer.step()

    return {
        "total": float(total_loss.item()),
        "elbo": float(mean_elbo_loss.item()),
        "supervised_total": float(loss_dict["total_loss"].item()),
        "expert_loss": float(loss_dict["expert_loss"].item()),
        "gate_loss": float(loss_dict["gate_loss"].item()),
        "rank_loss": float(loss_dict["rank_loss"].item()),
    }
