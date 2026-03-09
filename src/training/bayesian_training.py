import pyro
from pyro.infer import Trace_ELBO


def build_router_elbo(num_particles=4):
    """Create ELBO object used in unified differentiable optimization.

    Uses Trace_ELBO (vs TraceMeanField_ELBO) because the guide contains
    dist.Categorical for the cluster assignment z, which is discrete and
    non-reparameterizable. Trace_ELBO handles discrete variables via the
    score function (REINFORCE) estimator. num_particles > 1 reduces variance.

    Caller is responsible for clearing the param store at initialization time if needed;
    doing so here would silently reset learned variational parameters (q_beta_a, q_beta_b,
    component_loc, component_scale) if called after training has started.
    """
    return Trace_ELBO(num_particles=num_particles)


def train_stream_step_with_svi(
    router,
    elbo,
    optimizer,
    ebl_loss_fn,
    z_refined,
    protein_raw_features,
    ppr_centroid,
    static_trust_4,
    v_prior,
    delta_mean,
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

    Args:
        ppr_centroid: PPR-weighted protein centroid [B, protein_dim] (static obs space).
        static_trust_4: First 4 trust features [B, 4], drug-data-independent (static obs space).
    """
    optimizer.zero_grad(set_to_none=True)

    elbo_loss = elbo.differentiable_loss(
        router.model,
        router.guide,
        protein_raw_features,
        ppr_centroid,
        static_trust_4,
        v_prior,
        delta_mean,
        trust_vector,
    )
    num_context = protein_raw_features.shape[0] if protein_raw_features.dim() > 1 else 1
    mean_elbo_loss = elbo_loss / max(num_context, 1)

    final_scores, gate_probs, expert_tensor = router(
        z_refined=z_refined,
        protein_raw_features=protein_raw_features,
        v_prior=v_prior,
        delta_mean=delta_mean,
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
