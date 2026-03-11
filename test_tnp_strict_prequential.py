import unittest

import torch
from torch_geometric.data import HeteroData

from diagnostic_attention import run_diagnostic
from run_streaming_exp_tnp import split_stream_episodes
from src.data.binds_activity import merge_activity_edges
from src.data.context_builder import TNPContextBuilder
from src.data.multiplex_loader import MultiplexPillarSampler
from src.models.neighbor_transfer import NeighborTransferModel
from src.protocol.prequential import ProteinEpisode
from src.training.cold_start_metrics import classify_regime


def make_test_graph():
    data = HeteroData()
    data["protein"].x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    data["drug"].x = torch.arange(36, dtype=torch.float32).view(12, 3) / 10.0
    data["protein", "similar", "protein"].edge_index = torch.tensor([[1, 2, 2], [0, 0, 1]])
    data["protein", "go_shared", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data["protein", "binds_pic50", "drug"].edge_index = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ]
    )
    data["protein", "binds_pic50", "drug"].edge_label = torch.tensor(
        [5.0, 6.0, 7.0, 8.0, 5.5, 6.5, 7.5, 8.5, 9.5, 6.2, 7.2]
    )
    return data


class StrictPrequentialTests(unittest.TestCase):
    def test_loader_empty_history_starts_cold(self):
        data = make_test_graph()
        loader = MultiplexPillarSampler(data, history_mode="empty")

        stats = loader.history_stats()
        self.assertEqual(stats["revealed_edge_count"], 0)
        self.assertEqual(stats["unique_revealed_edge_count"], 0)

        pillar = loader.get_pillar_context(1)
        self.assertEqual(pillar["form_binds_ei"].size(1), 0)
        self.assertEqual(pillar["role_binds_ei"].size(1), 0)

    def test_stream_split_partitions_historical_prefix(self):
        episodes = [ProteinEpisode(t=i, protein_idx=i, edges=torch.zeros((2, 0), dtype=torch.long), labels=torch.zeros(0)) for i in range(10)]
        historical, stream = split_stream_episodes(episodes, 0.5)

        self.assertEqual(len(historical), 5)
        self.assertEqual(len(stream), 5)
        self.assertEqual([ep.protein_idx for ep in historical], [0, 1, 2, 3, 4])
        self.assertEqual([ep.protein_idx for ep in stream], [5, 6, 7, 8, 9])

    def test_add_revealed_edges_deduplicates_pairs(self):
        data = make_test_graph()
        loader = MultiplexPillarSampler(data, history_mode="empty")
        edge_index = data["protein", "binds_pic50", "drug"].edge_index
        edge_label = data["protein", "binds_pic50", "drug"].edge_label

        loader.add_revealed_edges(edge_index[:, :3], edge_label[:3])
        loader.add_revealed_edges(edge_index[:, 1:4], edge_label[1:4])

        stats = loader.history_stats()
        self.assertEqual(stats["revealed_edge_count"], 4)
        self.assertEqual(stats["unique_revealed_edge_count"], 4)
        self.assertEqual(stats["duplicate_revealed_edges"], 2)

    def test_historical_seed_populates_stream_context_without_target_leakage(self):
        data = make_test_graph()
        loader = MultiplexPillarSampler(data, history_mode="empty")
        edge_index = data["protein", "binds_pic50", "drug"].edge_index
        edge_label = data["protein", "binds_pic50", "drug"].edge_label

        hist_mask = edge_index[0] == 0
        loader.begin_episode(0)
        loader.add_revealed_edges(edge_index[:, hist_mask], edge_label[hist_mask])

        stats = loader.history_stats()
        self.assertEqual(stats["revealed_edge_count"], int(hist_mask.sum().item()))

        loader.begin_episode(1)
        pillar = loader.get_pillar_context(1)
        self.assertGreater(pillar["form_binds_ei"].size(1), 0)
        self.assertTrue(torch.all(pillar["form_binds_ei"][0] == 0))

    def test_merge_activity_edges_uses_amax(self):
        data = HeteroData()
        data["protein"].x = torch.randn(2, 3)
        data["drug"].x = torch.randn(2, 2)
        data["protein", "similar", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["protein", "go_shared", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data["protein", "binds_pic50", "drug"].edge_index = torch.tensor([[0, 1], [0, 1]])
        data["protein", "binds_pic50", "drug"].edge_label = torch.tensor([6.0, 5.0])
        data["protein", "binds_pki", "drug"].edge_index = torch.tensor([[0], [0]])
        data["protein", "binds_pki", "drug"].edge_label = torch.tensor([8.0])

        merged = merge_activity_edges(data, reduce="amax")
        edge_index = merged["protein", "binds_activity", "drug"].edge_index
        edge_label = merged["protein", "binds_activity", "drug"].edge_label
        pair_to_label = {
            (int(edge_index[0, i]), int(edge_index[1, i])): float(edge_label[i])
            for i in range(edge_index.size(1))
        }
        self.assertEqual(pair_to_label[(0, 0)], 8.0)
        self.assertEqual(pair_to_label[(1, 1)], 5.0)

    def test_per_query_cold_start_returns_full_tuple(self):
        drug_features = torch.randn(5, 4)
        builder = TNPContextBuilder(drug_features, global_mean_affinity=6.5)
        pillar = {
            "target_features": torch.randn(3),
            "form_neighbors": torch.zeros(0, dtype=torch.long),
            "form_features": torch.zeros(0, 3),
            "form_diff_w": torch.zeros(0),
            "form_binds_ei": torch.zeros((2, 0), dtype=torch.long),
            "form_binds_y": torch.zeros(0),
            "form_binds_w": torch.zeros(0),
            "role_neighbors": torch.zeros(0, dtype=torch.long),
            "role_features": torch.zeros(0, 3),
            "role_diff_w": torch.zeros(0),
            "role_binds_ei": torch.zeros((2, 0), dtype=torch.long),
            "role_binds_y": torch.zeros(0),
            "role_binds_w": torch.zeros(0),
            "ppr_centroid": torch.zeros(3),
        }

        out = builder.build_per_query_context(pillar, torch.tensor([0, 1, 2]), per_query_k=4)
        self.assertEqual(len(out), 8)
        pq_protein, pq_drug, pq_affinity, pq_ppr, pq_trust, pq_gnn, pq_aff_mean, pq_go_fp = out
        self.assertEqual(pq_protein.shape, (3, 0, 3))
        self.assertEqual(pq_drug.shape, (3, 0, 4))
        self.assertEqual(pq_affinity.shape, (3, 0, 1))
        self.assertEqual(pq_ppr.shape, (3, 0))
        self.assertEqual(pq_trust.shape, (3, 0))
        self.assertIsNone(pq_gnn)
        self.assertIsNone(pq_go_fp)
        self.assertTrue(torch.allclose(pq_aff_mean, torch.full((3,), 6.5)))

    def test_strict_history_regimes_progress_cold_sparse_warm(self):
        data = make_test_graph()
        loader = MultiplexPillarSampler(data, history_mode="empty")
        edge_index = data["protein", "binds_pic50", "drug"].edge_index
        edge_label = data["protein", "binds_pic50", "drug"].edge_label

        regimes = []

        loader.begin_episode(0)
        regimes.append(classify_regime(loader.get_pillar_context(0)["form_binds_ei"].size(1)))
        mask0 = edge_index[0] == 0
        loader.add_revealed_edges(edge_index[:, mask0], edge_label[mask0])

        loader.begin_episode(1)
        regimes.append(classify_regime(loader.get_pillar_context(1)["form_binds_ei"].size(1)))
        mask1 = edge_index[0] == 1
        loader.add_revealed_edges(edge_index[:, mask1], edge_label[mask1])

        loader.begin_episode(2)
        regimes.append(classify_regime(loader.get_pillar_context(2)["form_binds_ei"].size(1)))

        self.assertEqual(regimes, ["cold", "sparse", "warm"])

    def test_attention_overfit_sanity_check(self):
        summary = run_diagnostic(n_steps=500, verbose=False)
        self.assertGreaterEqual(summary["final_ci"], 0.95)

    def test_neighbor_transfer_context_uses_similar_neighbor_drugs(self):
        data = make_test_graph()
        loader = MultiplexPillarSampler(data, history_mode="empty")
        edge_index = data["protein", "binds_pic50", "drug"].edge_index
        edge_label = data["protein", "binds_pic50", "drug"].edge_label

        mask0 = edge_index[0] == 0
        mask1 = edge_index[0] == 1
        loader.add_revealed_edges(edge_index[:, mask0], edge_label[mask0])
        loader.add_revealed_edges(edge_index[:, mask1], edge_label[mask1])

        builder = TNPContextBuilder(data["drug"].x)
        pillar = loader.get_pillar_context(2)
        (
            neighbor_protein,
            neighbor_drug,
            neighbor_affinity,
            neighbor_ppr,
            neighbor_trust,
            neighbor_mask,
            matched_counts,
            neighbor_go_fp,
        ) = builder.build_neighbor_transfer_context(pillar, torch.tensor([4, 11]), top_k=2)

        self.assertEqual(neighbor_protein.shape, (2, 2, 3))
        self.assertEqual(neighbor_drug.shape, (2, 2, 3))
        self.assertTrue(neighbor_mask[0, 0].item())
        self.assertEqual(int(matched_counts[0]), 2)
        self.assertEqual(int(matched_counts[1]), 2)
        self.assertTrue(neighbor_mask[1].all().item())
        self.assertIsNone(neighbor_go_fp)
        self.assertEqual(float(neighbor_ppr[0, 0]), 1.0)
        self.assertEqual(float(neighbor_trust[0, 0]), 1.0)
        self.assertTrue(torch.all(neighbor_affinity[0] >= 0))

    def test_neighbor_transfer_model_falls_back_without_neighbors(self):
        model = NeighborTransferModel(protein_dim=3, drug_dim=2, go_fp_dim=0, hidden_dim=32)
        mu, sigma = model(
            neighbor_protein=torch.zeros(2, 2, 3),
            neighbor_drug=torch.zeros(2, 2, 2),
            neighbor_affinity=torch.zeros(2, 2),
            neighbor_ppr=torch.zeros(2, 2),
            neighbor_trust=torch.zeros(2, 2),
            neighbor_mask=torch.zeros(2, 2, dtype=torch.bool),
            qry_protein=torch.randn(2, 3),
            qry_drug=torch.randn(2, 2),
            global_mean_affinity=6.5,
        )
        self.assertFalse(torch.allclose(mu, torch.full((2,), 6.5)))
        self.assertTrue(torch.all(sigma > 0))


if __name__ == "__main__":
    unittest.main()
