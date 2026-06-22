#!/bin/bash
# ============================================================================
#  SLURM job: train the NS pressure warm-start MLP on the HPC.
#
#  This is a JOB ARRAY: each array task trains ONE model configuration
#  (a different seed or physics-loss weight) in parallel. See CONFIGS below.
#
#  Submit with:   sbatch --array=0-5 hpc/train.slurm
#  (0-5 = the six configurations defined in the CONFIGS array below.)
# ============================================================================

#SBATCH --job-name=swiftcfd_train
#SBATCH --output=hpc/logs/train_%A_%a.out   # %A=job id, %a=array index
#SBATCH --error=hpc/logs/train_%A_%a.err
#SBATCH --time=04:00:00                      # walltime per array task
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                         # 1 GPU; drop this line for CPU-only
# #SBATCH --partition=gpu                    # <-- set to your cluster's GPU partition
# #SBATCH --account=YOUR_ACCOUNT             # <-- set to your allocation/account

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Environment — plain venv, torch only (training needs NO PETSc).
# ---------------------------------------------------------------------------
module load python/3.10 2>/dev/null || true
mkdir -p hpc/logs
if [ ! -d venv ]; then
    python3 -m venv venv && source venv/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -r hpc/requirements-train.txt
else
    source venv/bin/activate
fi
export PYTHONPATH=$PWD:$PYTHONPATH

# ---------------------------------------------------------------------------
# 2. The configurations this array trains (one per array index).
#    Format:  LABEL|EXTRA_ARGS
# ---------------------------------------------------------------------------
CONFIGS=(
  "run_dataonly_s0|--seed 0 --weight-pde 0.0 --weight-data 1.0"   # noise-floor seeds
  "run_dataonly_s1|--seed 1 --weight-pde 0.0 --weight-data 1.0"
  "run_dataonly_s2|--seed 2 --weight-pde 0.0 --weight-data 1.0"
  "run_w1e-7|--seed 0 --weight-pde 1e-7 --weight-data 1.0"        # physics-weight sweep
  "run_w1e-6|--seed 0 --weight-pde 1e-6 --weight-data 1.0"
  "run_w1e-5|--seed 0 --weight-pde 1e-5 --weight-data 1.0"
)

ENTRY="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${ENTRY%%|*}"
ARGS="${ENTRY#*|}"

echo "=== Array task $SLURM_ARRAY_TASK_ID : $LABEL ==="
echo "    args: $ARGS"
nvidia-smi 2>/dev/null | head -15 || echo "(no GPU visible — running on CPU)"

# ---------------------------------------------------------------------------
# 3. Train.
# ---------------------------------------------------------------------------
python3 swiftcfd.py --train --equation-type ns \
    --model mlp --input-variables u,v,p --output-variables p \
    --epochs 200 \
    --output-dir output/$LABEL \
    $ARGS

echo "=== done: output/$LABEL ==="
