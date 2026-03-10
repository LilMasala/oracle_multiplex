# Oracle Multiplex

A drug discovery system for cold-start protein-ligand binding affinity prediction. Given a new protein with zero known binding interactions, Oracle Multiplex leverages a multiplex protein graph—encoding structural and functional similarity—to transfer knowledge from related proteins and rank candidate drug compounds accurately.

## Overview

The core challenge in virtual screening is the **cold-start problem**: new or under-characterized proteins lack binding data, making standard supervised models unreliable. Oracle Multiplex addresses this with:

- **Transformer Neural Process (TNP)**: a probabilistic model that treats known neighbor-binding tuples as a context set and outputs a predictive distribution over affinity for each query drug — explicitly representing uncertainty
- **Graph-Biased Attention**: attention weights incorporate PPR scores (diffusion proximity), structural delta (target–neighbor difference), and temporal decay — directly encoding graph structure into the information-gathering mechanism
- **Prequential Protocol**: evaluates the model in a test-then-train streaming loop that mirrors real laboratory conditions — each protein episode is encountered once, ranked blindly, then used for learning
- **Experience Replay**: historical protein episodes are sampled and replayed during training to mitigate forgetting in the sequential setting

## Architecture

```
MultiplexPillarSampler          # Fetches structural/functional neighbors, PPR scores, trust weights
        │
        ▼
TNPContextBuilder               # Assembles context set: (protein, drug, affinity, ppr, delta, trust)
        │
        ▼
ProteinLigandTNP                # Graph-biased Transformer Neural Process
  ├── Context encoder           # (protein, drug, affinity) → token
  ├── Query encoder             # (protein, drug) → token
  ├── GraphBiasedTransformer    # PPR + delta logit biases; trust V-gate
  └── Output head               # token → (mu, sigma)
        │
        ▼
TNPLoss                         # Gaussian NLL + ListNet + Lambda-MART ranking losses
```

### Key Components

| Module | File | Description |
|---|---|---|
| `MultiplexPillarSampler` | `src/data/multiplex_loader.py` | Builds per-protein context: neighbors, PPR scores, trust vectors, temporal decay weights |
| `TNPContextBuilder` | `src/data/context_builder.py` | Converts pillar dicts to TNP context tensors including ppr, delta, trust per edge |
| `ProteinLigandTNP` | `src/models/tnp.py` | Graph-biased TNP with learnable PPR/delta/trust attention parameters |
| `GraphBiasedMHA` | `src/models/tnp.py` | Multi-head attention with additive logit biases and multiplicative V gate |
| `TNPLoss` | `src/training/tnp_loss.py` | NLL + ListNet + Lambda-MART composite ranking loss |
| `build_multiplex_stream` | `src/protocol/prequential.py` | Constructs sequential protein episodes for streaming evaluation |
| `BayesianMultiplexRouter` | `src/models/routing.py` | (Legacy v1) Stick-breaking DPMM with variational Beta/Gamma posteriors |

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
| `data/multiplex_priors.pt` | Pre-computed PPR scores, trust metrics, PPR centroids |
| `data/dpmm_init.pt` | (Optional) Offline DPMM initialization checkpoint for legacy v1 |

The graph contains three edge types:
- `similar` — structural similarity between proteins
- `go_shared` — shared functional annotations
- `binds_activity` — merged binding affinity labels (pIC50, pKi, pKd)

## Usage

### 1. Run the TNP Prequential Streaming Experiment

```bash
python run_streaming_exp_tnp.py \
    --data data/final_graph_data_not_normalized.pt \
    --priors data/multiplex_priors.pt
```

For each protein episode:
1. **Evaluate**: rank all candidate drugs cold-start (no binding labels seen yet)
2. **Train**: optimize on revealed labels + replayed historical data
3. **Update**: add new edges to the growing information base

Outputs:
- `results/stream_tnp_v1.csv` — per-episode CI, EF@10, NLL and loss breakdown
- `models/oracle_tnp_v1.pt` — final model weights
- `models/tnp_checkpoint_ep*.pt` — checkpoints every 50 episodes

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--lr` | `5e-4` | Adam learning rate |
| `--token-dim` | `256` | Transformer token dimensionality |
| `--max-context` | `256` | Max neighbor binding tuples in context set |
| `--replay-weight` | `0.25` | Experience replay loss scaling |
| `--n-episodes` | all | Limit episodes for quick testing |

### 2. (Optional) Pre-initialize the DPMM (Legacy v1)

Pre-trains the Bayesian router with offline protein clustering before streaming:

```bash
python scripts/pretrain_dpmm.py \
    --data data/final_graph_data_not_normalized.pt \
    --priors data/multiplex_priors.pt \
    --max-experts 16 \
    --n-steps 2000 \
    --output data/dpmm_init.pt
```

### 3. Interactive Notebook

```bash
jupyter notebook run_streaming_exp.ipynb
```

## Evaluation Metrics

- **CI (Concordance Index)**: probability the model correctly ranks a randomly drawn drug pair; 0.5 = random, 1.0 = perfect
- **EF@10 (Enrichment Factor)**: ratio of active compounds in the top 10% of predictions vs. random expectation

## Technical Details

### Transformer Neural Process

The TNP frames binding prediction as in-context learning. For a new target protein with no binding data, the model:

1. Assembles a **context set** of `(neighbor_protein, drug, affinity)` triples from structurally/functionally related proteins
2. Encodes each triple as a token; encodes each `(target_protein, drug)` query pair as a token
3. Runs cross-attention (context → queries) so each query reads from the most relevant context items
4. Predicts a **Gaussian distribution** `(mu, sigma)` over affinity per query drug

The explicit `sigma` output allows the model to express uncertainty — useful for active learning and multi-round screening.

### Graph-Biased Attention

Rather than treating all neighbor-binding tuples as equally informative, three graph signals are injected into the attention mechanism:

**1. PPR logit bias** — Personalized PageRank score (diffusion proximity) of the context neighbor to the target protein:
```
attn_logit(i → ctx_k) += α · log(ppr_k + ε)
```
High-PPR neighbors receive higher attention, so the model naturally up-weights close relatives in the protein graph.

**2. Structural delta bias** — Projected difference between target and neighbor protein embeddings:
```
attn_logit(i → ctx_k) += w_delta · (target_emb - neighbor_emb_k)
```
This encourages attention to neighbors that are structurally close (small delta → small bias correction), and lets the model learn which structural dimensions matter for binding transfer.

**3. Temporal trust gate** — Multiplicative gate on context token values using edge recency:
```
v_ctx_k ← sigmoid(s · decay_weight_k) · v_ctx_k
```
Old, low-confidence binding edges are downweighted in the value aggregation; fresh or high-confidence edges pass through at full strength.

All three parameters (`α`, `w_delta`, `s`) are learnable and initialized at zero, so the model starts as a standard TNP and gradually learns to exploit graph structure.

### Prequential Protocol

Each protein appears exactly once in streaming order. This simulates a real screening campaign:
- No look-ahead: ranking happens before labels are revealed
- Sequential online learning: weights updated after each protein
- Experience replay: subsample of past proteins replayed to prevent forgetting

### Context Assembly

`TNPContextBuilder` populates the context set from the pillar dict produced by `MultiplexPillarSampler`:
- Gathers binding edges from `form` (structural similarity) and `role` (GO-term) neighbor layers
- Filters low-confidence edges (`min_affinity_weight` threshold)
- Per edge: extracts `ppr` (from `form_diff_w`/`role_diff_w`), `delta` (target − neighbor features), `trust` (decay-weighted edge weight)
- Subsamples to `max_context` if over budget

## Project Structure

```
oracle_multiplex/
├── src/
│   ├── data/
│   │   ├── multiplex_loader.py    # MultiplexPillarSampler
│   │   ├── context_builder.py     # TNPContextBuilder
│   │   └── make_graph.py
│   ├── models/
│   │   ├── tnp.py                 # ProteinLigandTNP (graph-biased)
│   │   ├── multiplex_moe.py       # (Legacy v1) MultiplexMoE orchestrator
│   │   ├── routing.py             # BayesianMultiplexRouter + ExpertScorer
│   │   └── smoothing.py           # MultiplexInductiveSmoother
│   ├── protocol/
│   │   └── prequential.py         # build_multiplex_stream
│   └── training/
│       ├── tnp_loss.py            # TNPLoss (NLL + ListNet + Lambda-MART)
│       ├── ebl_loss.py            # (Legacy v1) EBLLoss
│       ├── bayesian_training.py   # ELBO construction
│       ├── metrics.py             # CI, EF@k
│       └── runner.py
├── scripts/
│   ├── pretrain_dpmm.py           # Offline DPMM pre-initialization
│   ├── precompute_multiplex_stats.py
│   └── build_base_graph.py
├── run_streaming_exp_tnp.py       # Main entry point (TNP)
├── run_streaming_exp.py           # (Legacy v1) MoE entry point
├── run_streaming_exp.ipynb
└── requirements.txt
```
