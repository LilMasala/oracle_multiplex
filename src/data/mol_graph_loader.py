import collections
import io
import json
import os
import tarfile
import zipfile
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import Data
from tqdm import tqdm


class ProteinGraphZipLoader:
    def __init__(self, zip_path: str, uniprot_to_idx: dict, cache_in_memory: bool = True):
        self._zip_path = zip_path
        self.idx_to_uniprot = {v: k for k, v in uniprot_to_idx.items()}

        with zipfile.ZipFile(zip_path, "r") as zf:
            self._member_map = {}
            for name in zf.namelist():
                if not name.endswith(".pt"):
                    continue
                stem = os.path.basename(name).replace("_protein.pt", "")
                self._member_map[stem] = name

            self._cache = {}
            if cache_in_memory:
                for uniprot_id, member_name in self._member_map.items():
                    if uniprot_id not in uniprot_to_idx:
                        continue
                    self._cache[uniprot_id] = torch.load(
                        io.BytesIO(zf.read(member_name)), weights_only=False
                    )

    def get(self, uniprot_id: str) -> Data:
        if uniprot_id in self._cache:
            return self._cache[uniprot_id]
        member_name = self._member_map[uniprot_id]
        with zipfile.ZipFile(self._zip_path, "r") as zf:
            return torch.load(io.BytesIO(zf.read(member_name)), weights_only=False)

    def get_by_idx(self, global_prot_idx: int) -> Data:
        return self.get(self.idx_to_uniprot[global_prot_idx])


# NOTE: tarfile.TarFile handles are NOT picklable.
# Always use num_workers=0 when constructing a DataLoader with MolGraphDataset.
class DrugGraphTarLoader:
    def __init__(self, tar_dir: str, chembl_to_idx: dict, index_path: Optional[str] = None,
                 cache_in_memory: bool = False):
        if index_path is not None and os.path.isfile(index_path):
            with open(index_path) as f:
                self._index = json.load(f)
        else:
            self._index = self._build_index(tar_dir)
            if index_path is not None:
                with open(index_path, "w") as f:
                    json.dump(self._index, f)

        self.idx_to_chembl = {v: k for k, v in chembl_to_idx.items()}
        self._tar_dir = tar_dir
        self._open_archives = collections.OrderedDict()
        self._lru_maxsize = 8
        self._graph_cache: dict = {}

        if cache_in_memory:
            self._load_all_into_cache(chembl_to_idx)

    def _load_all_into_cache(self, chembl_to_idx: dict):
        """Load all drug graphs present in both the index and chembl_to_idx into memory."""
        wanted = {cid for cid in chembl_to_idx if cid in self._index}
        by_archive: dict[str, list] = collections.defaultdict(list)
        for cid in wanted:
            archive_name, member_name = self._index[cid]
            by_archive[archive_name].append((cid, member_name))
        print(f"Caching {len(wanted)} drug graphs from {len(by_archive)} archives...")
        for archive_name, members in tqdm(by_archive.items(), desc="loading drug graphs"):
            with tarfile.open(os.path.join(self._tar_dir, archive_name)) as tf:
                for cid, member_name in members:
                    f = tf.extractfile(member_name)
                    self._graph_cache[cid] = torch.load(
                        io.BytesIO(f.read()), weights_only=False
                    )
        print(f"Drug graph cache ready: {len(self._graph_cache)} entries")

    def _build_index(self, tar_dir: str) -> dict:
        index = {}
        archives = [f for f in os.listdir(tar_dir) if f.endswith(".tar.gz")]
        for i, archive_name in enumerate(archives):
            if i > 0 and i % 100 == 0:
                print(f"  indexing archives: {i}/{len(archives)}")
            archive_path = os.path.join(tar_dir, archive_name)
            with tarfile.open(archive_path) as tf:
                for m in tf.getmembers():
                    if m.isfile() and m.name.endswith(".pt"):
                        chembl_id = os.path.basename(m.name).replace(".pt", "")
                        index[chembl_id] = (archive_name, m.name)
        return index

    def _get_tar(self, archive_filename: str) -> tarfile.TarFile:
        if archive_filename in self._open_archives:
            self._open_archives.move_to_end(archive_filename)
            return self._open_archives[archive_filename]
        handle = tarfile.open(os.path.join(self._tar_dir, archive_filename))
        if len(self._open_archives) >= self._lru_maxsize:
            _, oldest = self._open_archives.popitem(last=False)
            oldest.close()
        self._open_archives[archive_filename] = handle
        return handle

    def get(self, chembl_id: str) -> Data:
        if chembl_id in self._graph_cache:
            return self._graph_cache[chembl_id]
        archive_filename, member_name = self._index[chembl_id]
        tf = self._get_tar(archive_filename)
        f = tf.extractfile(member_name)
        return torch.load(io.BytesIO(f.read()), weights_only=False)

    def get_by_idx(self, global_drug_idx: int) -> Data:
        return self.get(self.idx_to_chembl[global_drug_idx])

    def close(self):
        for tar in self._open_archives.values():
            tar.close()
        self._open_archives.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MolGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        edge_index: Tensor,
        edge_label: Tensor,
        prot_loader: ProteinGraphZipLoader,
        drug_loader: DrugGraphTarLoader,
        idx_to_uniprot: dict,
        idx_to_chembl: dict,
    ):
        self.edge_index = edge_index
        self.edge_label = edge_label
        self.prot_loader = prot_loader
        self.drug_loader = drug_loader
        self.idx_to_uniprot = idx_to_uniprot
        self.idx_to_chembl = idx_to_chembl

    def __len__(self):
        return self.edge_index.size(1)

    def __getitem__(self, i) -> dict:
        prot_idx = int(self.edge_index[0, i])
        drug_idx = int(self.edge_index[1, i])
        return {
            "prot_graph": self.prot_loader.get_by_idx(prot_idx),
            "drug_graph": self.drug_loader.get_by_idx(drug_idx),
            "label": float(self.edge_label[i]),
            "prot_idx": prot_idx,
            "drug_idx": drug_idx,
        }


def mol_graph_collate_fn(batch: list) -> dict:
    from torch_geometric.data import Batch as PyGBatch

    return {
        "prot_batch": PyGBatch.from_data_list([b["prot_graph"] for b in batch]),
        "drug_batch": PyGBatch.from_data_list([b["drug_graph"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.float),
        "prot_ids": torch.tensor([b["prot_idx"] for b in batch], dtype=torch.long),
        "drug_ids": torch.tensor([b["drug_idx"] for b in batch], dtype=torch.long),
    }


def _smoke_test():
    import tempfile
    import zipfile as zf
    from torch_geometric.data import Data

    g = Data(
        x=torch.randn(5, 11),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, 13),
    )
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    with zf.ZipFile(tmp_path, "w") as z:
        buf = io.BytesIO()
        torch.save(g, buf)
        buf.seek(0)
        z.writestr("protein_graphs/P12345_protein.pt", buf.read())

    loader = ProteinGraphZipLoader(tmp_path, {"P12345": 0})
    g2 = loader.get_by_idx(0)
    assert g2.x.shape == (5, 11), f"Expected (5,11), got {g2.x.shape}"
    print("Unit B smoke test PASSED")
    os.unlink(tmp_path)


if __name__ == "__main__":
    import sys

    _smoke_test()

    if len(sys.argv) > 1:
        import argparse

        parser = argparse.ArgumentParser(description="Build drug graph tar index")
        parser.add_argument("--tar-dir", required=True)
        parser.add_argument("--out-index", required=True)
        args = parser.parse_args()
        loader = DrugGraphTarLoader.__new__(DrugGraphTarLoader)
        loader._tar_dir = args.tar_dir
        loader._open_archives = {}
        loader._lru_maxsize = 1
        index = loader._build_index(args.tar_dir)
        with open(args.out_index, "w") as f:
            json.dump(index, f)
        print(f"Saved index with {len(index)} entries to {args.out_index}")
