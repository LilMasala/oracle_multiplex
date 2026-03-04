import torch
import torch.nn.functional as F
import numpy as np



def _sample_replay_batches(loader, current_protein_idx, replay_edges_per_protein=256, max_replay_proteins=1):
    if loader.binds_ei.size(1) == 0:
        return []

    proteins = torch.unique(loader.binds_ei[0])
    proteins = proteins[proteins != int(current_protein_idx)]
    if proteins.numel() == 0:
        return []

    selected = proteins[torch.randperm(proteins.numel(), device=proteins.device)[: min(max_replay_proteins, proteins.numel())]]
    replay = []
    for pidx in selected.tolist():
        eids = torch.nonzero(loader.binds_ei[0] == pidx, as_tuple=False).squeeze(-1)
        if eids.numel() == 0:
            continue
        take = min(replay_edges_per_protein, eids.numel())
        chosen = eids[torch.randperm(eids.numel(), device=eids.device)[:take]]
        replay.append((loader.binds_ei[:, chosen], loader.binds_y[chosen], loader.get_pillar_context(int(pidx))))
    return replay


def run_prequential_stream(model, episodes, loader, drug_features, optimizer, config, loss_fn=None):
    """
    Executes the linear timeline.
    model: MultiplexMoE wrapper.
    episodes: list of ProteinEpisode objects.
    loader: MultiplexPillarSampler.
    """
    model.train()
    stream_results = {"ci": [], "mse": []}
    replay_weight = config.get("replay_weight", 0.25)

    for ep_num, ep in enumerate(episodes):
        optimizer.zero_grad()
        pillar = loader.get_pillar_context(ep.protein_idx)

        with torch.no_grad():
            query_drug_indices = ep.query_edges[1]
            query_preds, _, _, q_stats = model(pillar, drug_features, query_drug_indices)
            mse_val = F.mse_loss(query_preds, ep.query_labels).item()
            stream_results["mse"].append(mse_val)
            print(f"Episode {ep_num} | Prot {ep.protein_idx} | Query MSE: {mse_val:.4f} | ||delta||={q_stats['delta_norm'].item():.4f}")

        support_drug_indices = ep.support_edges[1]
        support_preds, gate_probs, expert_tensor, _ = model(pillar, drug_features, support_drug_indices)

        base_loss = F.mse_loss(support_preds, ep.support_labels)
        with torch.no_grad():
            per_expert_err = (expert_tensor - ep.support_labels.unsqueeze(-1)) ** 2
            target_gate_probs = F.softmax(-per_expert_err, dim=-1)
        gate_loss = -torch.sum(target_gate_probs * torch.log(gate_probs + 1e-12), dim=-1).mean()
        current_loss = base_loss + (config.get("ebl_alpha", 0.1) * gate_loss)

        replay_loss = torch.tensor(0.0, device=support_preds.device)
        replay_batches = _sample_replay_batches(loader, ep.protein_idx, replay_edges_per_protein=config.get("replay_edges", 256))
        for replay_edges, replay_labels, replay_pillar in replay_batches:
            rp, rg, re, _ = model(replay_pillar, drug_features, replay_edges[1])
            if loss_fn is not None:
                replay_loss = replay_loss + loss_fn(rp, replay_labels, rg, re)["total_loss"]
            else:
                replay_loss = replay_loss + F.mse_loss(rp, replay_labels)

        if replay_batches:
            replay_loss = replay_loss / len(replay_batches)

        total_loss = current_loss + replay_weight * replay_loss
        total_loss.backward()
        optimizer.step()
        print(
            f"Episode {ep_num} | current_loss={current_loss.item():.4f} | replay_loss={replay_loss.item():.4f}"
        )

        loader.add_revealed_edges(ep.support_edges, ep.support_labels)

    return stream_results
