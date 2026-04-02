#!/bin/bash
#SBATCH --job-name=oracle_gp
#SBATCH --partition=mwgpu256
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --output=/zfshomes/aparikh02/ORACLE/logs/%x_%j.log
#SBATCH --error=/zfshomes/aparikh02/ORACLE/logs/%x_%j.err

set -e
mkdir -p /zfshomes/aparikh02/ORACLE/logs
mkdir -p /zfshomes/aparikh02/ORACLE/mol_prior_node

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda/12.4
source /share/apps/CENTOS8/ohpc/software/miniconda3/py311/etc/profile.d/conda.sh
conda activate oracle

# ── Paths ─────────────────────────────────────────────────────────────────────
ORACLE_DIR=/zfshomes/aparikh02/ORACLE
REPO=$ORACLE_DIR/oracle_multiplex

HETERO_DATA=$ORACLE_DIR/final_graph_data_not_normalized.pt
PRIORS=$ORACLE_DIR/multiplex_priors.pt
PROTEIN_ZIP=$ORACLE_DIR/protein_graphs.zip
DRUG_TAR_DIR=$ORACLE_DIR/drug_graphs
DRUG_INDEX=$ORACLE_DIR/drug_index.json
DRUG_PACKED_CACHE=/sanscratch/aparikh02/oracle_cache/drug_graphs_packed_reassembled.pt
MOL_PRIOR_DIR=$ORACLE_DIR/mol_prior_node
RUN_NAME=oracle_gp_mol_${SLURM_JOB_ID}

cd $REPO

# ── VRAM monitor (background) ─────────────────────────────────────────────────
nvidia-smi dmon -s mu -d 10 -o DT \
    > $ORACLE_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_vram.log &
VRAM_PID=$!

echo "=============================="
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Run:     $RUN_NAME"
echo "=============================="

# ── Stage 1: Pretrain MolGraphPrior (skip if already trained) ─────────────────
# echo ""
# echo ">>> Stage 1: Pretrain MolGraphPrior"
# torchrun --nproc_per_node=4 scripts/pretrain_mol_graph_prior.py \
#     --hetero-data      $HETERO_DATA \
#     --protein-zip      $PROTEIN_ZIP \
#     --drug-tar-dir     $DRUG_TAR_DIR \
#     --drug-index       $DRUG_INDEX \
#     --drug-packed-cache $DRUG_PACKED_CACHE \
#     --output-dir       $MOL_PRIOR_DIR \
#     --hidden           256 \
#     --num-layers       4 \
#     --epochs           50 \
#     --batch-size       32 \
#     --lr               5e-4 \
#     --bilinear-rank    128 \
#     --embed-batch-size 512 \
#     --historical-protein-frac 0.5 \
#     --scorer           esm_cross_attn \
#     --eval-every       5
# echo ">>> Stage 1 complete — mol_prior_tables.pt saved to $MOL_PRIOR_DIR"

# ── Stage 2: Prequential streaming experiment ─────────────────────────────────
echo ""
echo ">>> Stage 2: Prequential GP streaming (esm_cross_attn prior)"
python run_streaming_exp_tnp.py \
    --data                    $HETERO_DATA \
    --priors                  $PRIORS \
    --run-name                $RUN_NAME \
    --model-kind              gp \
    --gnn-prior               mol \
    --mol-prior-dir           $MOL_PRIOR_DIR \
    --mol-protein-zip         $PROTEIN_ZIP \
    --mol-drug-tar-dir        $DRUG_TAR_DIR \
    --mol-drug-packed-cache   $DRUG_PACKED_CACHE \
    --historical-protein-frac 0.5 \
    --history-mode            full \
    --max-context             32 \
    --lr                      1e-4 \
    --nb-role-boost           1.0 \
    --checkpoint-every        100 \
    --mol-runtime-batch-size  16 \
    --seed                    42

# ── Wrap up ───────────────────────────────────────────────────────────────────
kill $VRAM_PID 2>/dev/null || true

echo ""
echo ">>> Done."
echo "    Results:   $REPO/results/${RUN_NAME}.csv"
echo "    VRAM log:  $ORACLE_DIR/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_vram.log"
