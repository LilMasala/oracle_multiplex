#!/bin/bash
#SBATCH --job-name=mol_pretrain
#SBATCH --partition=exx512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:4
#SBATCH --output=/zfshomes/aparikh02/ORACLE/logs/%x_%j.log
#SBATCH --error=/zfshomes/aparikh02/ORACLE/logs/%x_%j.err

set -e
mkdir -p /zfshomes/aparikh02/ORACLE/logs
mkdir -p /zfshomes/aparikh02/ORACLE/mol_prior

# ── Environment ───────────────────────────────────────────────────────────────
source /zfshomes/aparikh02/miniconda3/etc/profile.d/conda.sh
conda activate oracle
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# ── Paths ─────────────────────────────────────────────────────────────────────
ORACLE_DIR=/zfshomes/aparikh02/ORACLE
REPO=$ORACLE_DIR/oracle_multiplex

HETERO_DATA=$ORACLE_DIR/final_graph_data_not_normalized.pt
PROTEIN_ZIP=$ORACLE_DIR/protein_graphs.zip
DRUG_TAR_DIR=$ORACLE_DIR/drug_graphs
DRUG_INDEX=$ORACLE_DIR/drug_index.json
DRUG_PACKED=/sanscratch/aparikh02/oracle_cache/drug_graphs_packed_reassembled.pt
MOL_PRIOR_DIR=$ORACLE_DIR/mol_prior

cd $REPO

# ── VRAM monitor (background) ─────────────────────────────────────────────────
nvidia-smi dmon -s mu -d 10 -o DT \
    > $ORACLE_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_vram.log &
VRAM_PID=$!

echo "=============================="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=============================="

# ── Pretrain MolGraphPrior ────────────────────────────────────────────────────
echo ""
echo ">>> Pretraining MolGraphPrior"
torchrun --nproc_per_node=4 scripts/pretrain_mol_graph_prior.py \
    --hetero-data             $HETERO_DATA \
    --protein-zip             $PROTEIN_ZIP \
    --drug-tar-dir            $DRUG_TAR_DIR \
    --drug-index              $DRUG_INDEX \
    --drug-packed-cache       $DRUG_PACKED \
    --output-dir              $MOL_PRIOR_DIR \
    --hidden                  256 \
    --num-layers              4 \
    --epochs                  50 \
    --batch-size              256 \
    --lr                      3e-4 \
    --bilinear-rank           128 \
    --embed-batch-size        512 \
    --historical-protein-frac 0.5

# ── Wrap up ───────────────────────────────────────────────────────────────────
kill $VRAM_PID 2>/dev/null || true

echo ""
echo ">>> Done. mol_prior_tables.pt saved to $MOL_PRIOR_DIR"
