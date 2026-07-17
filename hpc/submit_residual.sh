#!/bin/bash
# ============================================================================
#  HPC training job for the RESIDUAL pressure warm-start model (delta = p^{n+1}-p^n).
#
#  Trains the MLP or RNN via train_residual_ns.py and writes
#  output/run_residual_<model>/{pinn_model_<model>.pth, norm_params_<model>.pth}.
#  PETSc-free: needs ONLY torch/numpy/pandas + the per-cell CSV training data.
#
#  Submit (MLP):  sbatch hpc/submit_residual.sh mlp
#  Submit (RNN):  sbatch hpc/submit_residual.sh rnn
# ============================================================================

#SBATCH --job-name=swiftcfd_resid
#SBATCH --output=hpc/logs/resid_%j.out
#SBATCH --error=hpc/logs/resid_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# #SBATCH --partition=gpu          # <-- set your cluster's GPU partition
# #SBATCH --account=YOUR_ACCOUNT   # <-- set your allocation code

set -euo pipefail
MODEL="${1:-mlp}"                  # first arg: mlp (default) or rnn
EPOCHS="${2:-200}"                 # second arg: epochs (default 200)
mkdir -p hpc/logs output

module load python/3.10 2>/dev/null || true
# module load cuda/12.1 2>/dev/null || true

if [ ! -d venv ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -r hpc/requirements-train.txt
else
    source venv/bin/activate
fi

export PYTHONPATH=$PWD:$PYTHONPATH
nvidia-smi 2>/dev/null | head -15 || echo "(no GPU visible — training on CPU)"

# Residual target -> always pure data loss (--weight-pde 0).
python3 -m swiftcfd.machineLearning.train_residual_ns \
        --model "$MODEL" \
        --input-variables u,v,p \
        --output-variable p \
        --epochs "$EPOCHS" \
        --weight-pde 0 \
        --seed 0 \
        --output-dir "output/run_residual_${MODEL}"

echo "=== done: models in output/run_residual_${MODEL} ==="
