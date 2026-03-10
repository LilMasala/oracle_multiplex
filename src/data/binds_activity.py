"""Helpers for building a consistent merged protein-drug activity edge type."""

from __future__ import annotations

from typing import Iterable

import torch


DEFAULT_ACTIVITY_METRICS = ("binds_pic50", "binds_pki", "binds_pkd")


def merge_activity_edges(
    data,
    metrics: Iterable[str] = DEFAULT_ACTIVITY_METRICS,
    reduce: str = "amax",
    output_edge_type: tuple[str, str, str] = ("protein", "binds_activity", "drug"),
):
    """
    Merge multiple binding edge types into a single activity edge type.

    Duplicate protein-drug pairs are reduced explicitly so every consumer trains
    against the same target definition.
    """
    edge_index_parts = []
    edge_label_parts = []
    for metric in metrics:
        edge_type = ("protein", metric, "drug")
        if edge_type not in data.edge_types:
            continue
        store = data[edge_type]
        edge_index_parts.append(store.edge_index)
        edge_label_parts.append(store.edge_label.float())

    if not edge_index_parts:
        return data

    edge_index = torch.cat(edge_index_parts, dim=1)
    edge_label = torch.cat(edge_label_parts, dim=0)
    num_drugs = int(data["drug"].num_nodes)

    pair_key = edge_index[0] * num_drugs + edge_index[1]
    unique_keys, inverse = torch.unique(pair_key, return_inverse=True)
    merged_label = torch.full(
        (unique_keys.size(0),),
        float("-inf") if reduce == "amax" else 0.0,
        dtype=edge_label.dtype,
        device=edge_label.device,
    )

    if reduce == "amax":
        merged_label.scatter_reduce_(0, inverse, edge_label, reduce="amax", include_self=True)
    elif reduce == "mean":
        merged_label.zero_()
        merged_label.scatter_add_(0, inverse, edge_label)
        counts = torch.bincount(inverse, minlength=unique_keys.size(0)).to(edge_label.device)
        merged_label = merged_label / counts.clamp(min=1)
    else:
        raise ValueError(f"Unsupported activity merge reduce='{reduce}'")

    merged_edge_index = torch.stack([unique_keys // num_drugs, unique_keys % num_drugs], dim=0)
    data[output_edge_type].edge_index = merged_edge_index
    data[output_edge_type].edge_label = merged_label
    return data
