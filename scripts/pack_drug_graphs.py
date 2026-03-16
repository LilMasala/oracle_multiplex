"""
One-time preprocessing: pack all drug graphs from tar archives into a single .pt file.

Loading one large file is orders of magnitude faster than seeking across 2410 tar archives.

Usage:
    python scripts/pack_drug_graphs.py \
      --tar-dir     /path/to/drug_graphs \
      --index       /path/to/drug_index.json \
      --out         /path/to/drug_graphs_packed.pt

Optional: filter to only drugs that are nodes in the HeteroData graph (~800k vs 2.3M):
    python scripts/pack_drug_graphs.py \
      --tar-dir     /path/to/drug_graphs \
      --index       /path/to/drug_index.json \
      --hetero-data /path/to/final_graph_data_not_normalized.pt \
      --out         /path/to/drug_graphs_packed.pt

Output: a dict {chembl_id: torch_geometric.data.Data} saved with torch.save.
Expected size: ~4 GB for 820k drugs. Load time: ~1-2 min vs 15 hours from Drive.
"""

import argparse
import collections
import io
import json
import os
import sys
import tarfile

import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main(args):
    print(f"Loading index from {args.index}...")
    with open(args.index) as f:
        index = json.load(f)
    print(f"Index has {len(index)} entries")

    # Optional: filter to only drugs that are actual nodes in the HeteroData graph.
    # The full drug_index.json covers all of ChEMBL (~2.3M), but only ~800k are
    # used as drug nodes in binding edges. This cuts RAM from ~160 GB to ~4 GB.
    if args.hetero_data:
        print(f"Loading HeteroData to filter used drugs from {args.hetero_data}...")
        data = torch.load(args.hetero_data, weights_only=False)
        used_chembl = set(data["drug"].index_to_chembl_id.values())
        before = len(index)
        index = {k: v for k, v in index.items() if k in used_chembl}
        print(f"Filtered index: {before} → {len(index)} entries (only HeteroData drug nodes)")
        del data, used_chembl

    # Group by archive to open each tar only once
    by_archive: dict[str, list] = collections.defaultdict(list)
    for cid, (archive_name, member_name) in index.items():
        by_archive[archive_name].append((cid, member_name))

    packed = {}
    n_skipped = 0
    for archive_name, members in tqdm(by_archive.items(), desc="packing archives"):
        archive_path = os.path.join(args.tar_dir, archive_name)
        with tarfile.open(archive_path) as tf:
            for cid, member_name in members:
                try:
                    f = tf.extractfile(member_name)
                    packed[cid] = torch.load(io.BytesIO(f.read()), weights_only=False)
                except Exception:
                    n_skipped += 1

    print(f"Packed {len(packed)} drug graphs ({n_skipped} skipped/corrupt). Saving to {args.out}...")
    torch.save(packed, args.out)
    size_gb = os.path.getsize(args.out) / 1e9
    print(f"Done. File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack drug graphs into a single .pt file")
    parser.add_argument("--tar-dir",     required=True, help="Directory containing drug graph tar.gz files")
    parser.add_argument("--index",       required=True, help="Path to drug_index.json")
    parser.add_argument("--out",         required=True, help="Output path for packed .pt file")
    parser.add_argument("--hetero-data", default=None,
                        help="Path to HeteroData .pt file; if given, only pack drugs that are "
                             "nodes in the graph (~800k) instead of all ChEMBL (~2.3M).")
    main(parser.parse_args())
