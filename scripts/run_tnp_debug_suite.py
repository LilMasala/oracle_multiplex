"""
Run the strict prequential debug/ablation matrix for the TNP harness.

This script shells out to `run_streaming_exp_tnp.py` so each configuration gets
its own clean process, output files, and model initialization.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_run_summary(csv_path: str):
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}

    def f(row, key, default=0.0):
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    last = rows[-1]
    regime_counts = {"cold": 0, "sparse": 0, "warm": 0}
    for row in rows:
        regime = row.get("regime")
        if regime in regime_counts:
            regime_counts[regime] += 1

    return {
        "episodes": len(rows),
        "mean_ci": sum(f(r, "ci") for r in rows) / len(rows),
        "last_ci_roll100": f(last, "ci_roll100"),
        "mean_ef10": sum(f(r, "ef10") for r in rows) / len(rows),
        "last_revealed_edge_count": int(f(last, "revealed_edge_count_after")),
        "cold_count": regime_counts["cold"],
        "sparse_count": regime_counts["sparse"],
        "warm_count": regime_counts["warm"],
    }


def main():
    parser = argparse.ArgumentParser(description="Run the strict TNP debug suite")
    parser.add_argument("--data", default="data/final_graph_data_not_normalized.pt")
    parser.add_argument("--priors", default="data/multiplex_priors.pt")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default="results/tnp_debug_suite.csv")
    parser.add_argument(
        "--include-incremental",
        action="store_true",
        help="Also run the incremental TNP variants: synthetic prior, GO, per-query, GNN, drug analogs",
    )
    args = parser.parse_args()

    output_path = args.output_csv
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    configs = [
        ("global_mean_floor", ["--model-kind", "global-mean", "--strict-baseline"]),
        ("binding_only", ["--model-kind", "binding-only", "--strict-baseline"]),
        ("tnp_strict", ["--model-kind", "tnp", "--strict-baseline"]),
    ]
    if args.include_incremental:
        configs.extend(
            [
                ("tnp_syn", ["--model-kind", "tnp", "--strict-baseline", "--enable-synthetic-prior"]),
                ("tnp_go", ["--model-kind", "tnp", "--strict-baseline", "--enable-synthetic-prior", "--use-go"]),
                ("tnp_per_query", ["--model-kind", "tnp", "--strict-baseline", "--enable-synthetic-prior", "--use-go", "--per-query-k", "32"]),
                ("tnp_gnn", ["--model-kind", "tnp", "--strict-baseline", "--enable-synthetic-prior", "--use-go", "--per-query-k", "32", "--gnn-mode", "frozen"]),
                ("tnp_analogs", ["--model-kind", "tnp", "--strict-baseline", "--enable-synthetic-prior", "--use-go", "--per-query-k", "32", "--gnn-mode", "frozen", "--drug-analogs"]),
            ]
        )

    suite_rows = []
    for name, extra_args in configs:
        run_name = f"suite_{name}"
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "run_streaming_exp_tnp.py"),
            "--data",
            args.data,
            "--priors",
            args.priors,
            "--n-episodes",
            str(args.n_episodes),
            "--seed",
            str(args.seed),
            "--run-name",
            run_name,
        ] + extra_args
        print(f"\n=== Running {name} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

        csv_path = os.path.join(PROJECT_ROOT, "results", f"{run_name}.csv")
        summary = read_run_summary(csv_path)
        summary["config"] = name
        summary["results_path"] = csv_path
        suite_rows.append(summary)

    if suite_rows:
        fieldnames = list(suite_rows[0].keys())
        with open(output_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(suite_rows)
        print(f"\nSaved suite summary to {output_path}")


if __name__ == "__main__":
    main()
