#!/bin/bash
# ============================================================================
#  Phase 0.3 — NOISE FLOOR.
#  Trains the SAME data-only config with 3 different seeds, then evaluates each
#  with the standard warm-start protocol, all in one job on the HPC.
#  Training AND evaluation need only torch/numpy/pandas (no PETSc).
#
#  Needs uploaded:  code (swiftcfd/, swiftcfd.py), hpc/, the training data
#                   (output/training_*/...), AND input/generated/val_01.toml
#
#  Submit:   sbatch hpc/noise_floor.slurm
# ============================================================================
#SBATCH --job-name=nf_mlp
#SBATCH --output=hpc/logs/nf_%j.out
#SBATCH --error=hpc/logs/nf_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# #SBATCH --partition=YOUR_GPU_PARTITION   # <-- set to your cluster's GPU partition
# #SBATCH --account=YOUR_ACCOUNT           # <-- set to your allocation code

set -euo pipefail
mkdir -p hpc/logs output

# --- 1. Modules (your cluster) ---------------------------------------------
module purge
module use /apps/modules/all
module load Python/3.11.5-GCCcore-13.2.0

# --- 2. Virtual environment (create once, reuse afterwards) -----------------
if [ ! -d ML_MLP ]; then
    python3 -m venv ML_MLP
    source ML_MLP/bin/activate
    pip install --upgrade pip
    pip install -r hpc/requirements-train.txt
else
    source ML_MLP/bin/activate
fi

# --- 3. Make `import swiftcfd` work from the repo root ----------------------
export PYTHONPATH=$PWD:$PYTHONPATH

# --- 4. Show the GPU (look for "Device: cuda" in training output too) -------
nvidia-smi 2>/dev/null | head -15 || echo "(no GPU visible — running on CPU)"

# --- 5. TRAIN three seeds (identical config, only --seed changes) -----------
for S in 0 1 2; do
    echo "==================== TRAIN seed $S ===================="
    python3 swiftcfd.py --train --equation-type ns \
        --model mlp --input-variables u,v,p --output-variables p \
        --epochs 200 --seed "$S" \
        --weight-pde 0.0 --weight-data 1.0 \
        --output-dir "output/run_dataonly_s$S"
done

# --- 6. EVALUATE each with the standard protocol ----------------------------
for S in 0 1 2; do
    echo "==================== EVAL seed $S ===================="
    python3 -m swiftcfd.machineLearning.evaluate_warmstart \
        --config input/generated/val_01.toml \
        --model "output/run_dataonly_s$S/pinn_model_mlp.pth" \
        --norm  "output/run_dataonly_s$S/norm_params_mlp.pth" \
        --spinup 10 --eval-steps 10 --tol 1e-4 --true-residual \
        2>&1 | grep -E "zero init|prev init|ML   init|ML vs prev" || true
done

deactivate
echo "=== noise-floor job complete ==="
