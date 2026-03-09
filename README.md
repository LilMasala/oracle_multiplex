# Oracle Multiplex

A drug discovery system for cold-start protein-ligand binding affinity prediction. Given a new protein with zero known binding interactions, Oracle Multiplex leverages a multiplex protein graph—encoding structural and functional similarity—to transfer knowledge from related proteins and rank candidate drug compounds accurately.

## Overview

The core challenge in virtual screening is the **cold-start problem**: new or under-characterized proteins lack binding data, making standard supervised models unreliable. Oracle Multiplex addresses this with:

- **Inductive Multiplex Mixture-of-Experts (MoE)**: routes each protein to specialized expert scorers based on its multiplex neighborhood context
- **Chemical Footprinting**: aggregates binding preference signals from structurally/functionally similar proteins via cross-attention
- **Prequential Protocol**: evaluates the model in a test-then-train streaming loop that mirrors real laboratory conditions—each protein episode is encountered once, ranked blindly, then used for learning
- **Bayesian Expert Routing**: a Dirichlet Process Mixture Model (DPMM) provides a principled probabilistic prior over expert assignments, with variational inference via Pyro

## Architecture

```
MultiplexPillarSampler          # Fetches structural/functional neighbors + trust vectors
        │
        ▼
MultiplexInductiveSmoother      # Chemical footprinting: cross-attention over neighbor binding histories
        │
        ▼
BayesianMultiplexRouter         # DPMM-based probabilistic routing to K expert scorers
        │
        ▼
ExpertScorer × K                # Bilinear affinity prediction heads
        │
        ▼
EBLLoss                         # Explanation-based learning: oracle routing supervision + ranking losses
```

### Key Components

| Module | File | Description |
|---|---|---|
| `MultiplexPillarSampler` | `src/data/multiplex_loader.py` | Builds per-protein context: neighbors, trust vectors, PPR centroids |
| `MultiplexInductiveSmoother` | `src/models/smoothing.py` | Cross-attention aggregation of neighbor binding preferences |
| `BayesianMultiplexRouter` | `src/models/routing.py` | Stick-breaking DPMM with variational Beta/Gamma posteriors |
| `ExpertScorer` | `src/models/routing.py` | Per-expert bilinear affinity scoring heads |
| `EBLLoss` | `src/training/ebl_loss.py` | Oracle gate supervision + ListNet/Lambda-MART ranking losses |
| `build_multiplex_stream` | `src/protocol/prequential.py` | Constructs sequential protein episodes for streaming evaluation |

## Installation

```bash
# Recommended: create a dedicated environment
micromamba create -n oracle python=3.10
micromamba activate oracle

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** `torch>=2.0`, `torch-geometric`, `pyro-ppl>=1.8`, `numpy`, `pandas`

## Data

The following files are expected under `data/`:

| File | Description |
|---|---|
| `data/final_graph_data_not_normalized.pt` | Heterogeneous PyG graph with protein, drug nodes and binding edges |
| `data/multiplex_priors.pt` | Pre-computed PPR scores and trust metrics |
| `data/dpmm_init.pt` | (Optional) Offline DPMM initialization checkpoint |

The graph contains three edge types:
- `similar` — structural similarity between proteins
- `go_shared` — shared functional annotations
- `binds_activity` — merged binding affinity labels (pIC50, pKi, pKd)

## Usage

### 1. (Optional) Pre-initialize the DPMM

Runs offline clustering to warm-start the Bayesian router before streaming:

```bash
python scripts/pretrain_dpmm.py \
    --data data/final_graph_data_not_normalized.pt \
    --priors data/multiplex_priors.pt \
    --max-experts 16 \
    --n-steps 2000 \
    --output data/dpmm_init.pt
```

### 2. Run the Prequential Streaming Experiment

```bash
python run_streaming_exp.py
```

For each protein episode:
1. **Evaluate**: rank all candidate drugs cold-start (no binding labels seen)
2. **Train**: optimize on revealed labels + replayed historical data
3. **Update**: add new edges to the growing information base

Outputs:
- `results/stream_analysis_v1.csv` — per-episode CI, EF@10, and loss breakdown
- `models/oracle_multiplex_v1.pt` — final model weights
- `models/checkpoint_ep*.pt` — checkpoints every 50 episodes

### 3. Interactive Notebook

```bash
jupyter notebook run_streaming_exp.ipynb
```

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `lr` | `5e-4` | Adam learning rate |
| `max_experts` | `16` | Number of expert scorers |
| `elbo_weight` | `0.1` | DPMM ELBO loss scaling |
| `ebl_alpha` | `0.3` | Gate supervision loss weight |
| `replay_weight` | `0.25` | Experience replay loss scaling |
| `support_batch_size` | `128` | Mini-batch size during training |
| `temperature` | `0.1` | EBL oracle distribution temperature |

## Evaluation Metrics

- **CI (Concordance Index)**: probability the model correctly ranks a randomly drawn drug pair; 0.5 = random, 1.0 = perfect
- **EF@10 (Enrichment Factor)**: ratio of active compounds in the top 10% of predictions vs. random expectation

## Technical Details

### Chemical Footprinting

For each neighbor protein `P_n`, a binding preference vector is constructed by projecting its known drug interactions into chemical space, weighted by temporal decay and attention scores over the target-neighbor similarity. These are aggregated via cross-attention to produce `v_prior`.

### Trust-Aware Routing

The router receives a 5-dimensional trust vector per neighbor:

```
[participation_coefficient, jaccard_overlap, neighbor_count, mean_ppr, binding_density]
```

The full gate input is `[z_refined, v_prior, delta_mean, trust_vector]`, producing a sparse top-k routing distribution over K experts.

### Explanation-Based Learning (EBL)

After observing labels, the oracle identifies which expert would have performed best per protein and constructs a soft target distribution:

```
oracle_dist = softmax(-MSE_per_expert / temperature)
target      = (1 - ε) * oracle_dist + ε * uniform
loss_gate   = CrossEntropy(predicted_gate, target)
```

Combined with ListNet and Lambda-MART ranking surrogates to directly optimize CI.

## Project Structure

```
oracle_multiplex/
├── src/
│   ├── data/
│   │   ├── multiplex_loader.py   # MultiplexPillarSampler
│   │   └── make_graph.py
│   ├── models/
│   │   ├── multiplex_moe.py      # Top-level MultiplexMoE orchestrator
│   │   ├── routing.py            # BayesianMultiplexRouter + ExpertScorer
│   │   └── smoothing.py          # MultiplexInductiveSmoother
│   ├── protocol/
│   │   └── prequential.py        # build_multiplex_stream
│   └── training/
│       ├── ebl_loss.py           # EBLLoss
│       ├── bayesian_training.py  # ELBO construction
│       ├── metrics.py            # CI, EF@k
│       └── runner.py
├── scripts/
│   ├── pretrain_dpmm.py
│   ├── precompute_multiplex_stats.py
│   └── build_base_graph.py
├── config/
│   └── default_config.yaml
├── run_streaming_exp.py          # Main entry point
├── run_streaming_exp.ipynb
└── requirements.txt
```
