#!/bin/bash
# ============================================================================
#  HPC training job for the whole-field U-Net pressure model (G3).
#
#  Trains train_unet.py on the field snapshots produced LOCALLY by
#  gen_field_data.py (that generator needs PETSc, so it CANNOT run here).
#  Writes output/run_unet/{unet_model.pth, unet_norm.pth}.
#  PETSc-free: needs ONLY torch/numpy + the uploaded output/field_data/*.npz.
#
#  Prereq: generate fields locally (Docker) and upload output/field_data/.
#  Submit:  sbatch hpc/submit_unet.sh
# ============================================================================

#SBATCH --job-name=swiftcfd_unet
#SBATCH --output=hpc/logs/unet_%j.out
#SBATCH --error=hpc/logs/unet_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# #SBATCH --partition=gpu          # <-- set your cluster's GPU partition
# #SBATCH --account=YOUR_ACCOUNT   # <-- set your allocation code

set -euo pipefail
EPOCHS="${1:-100}"                 # first arg: epochs (default 100)
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

if ! ls output/field_data/fields_*.npz >/dev/null 2>&1; then
    echo "ERROR: no output/field_data/fields_*.npz found."
    echo "Generate them LOCALLY first (needs PETSc):"
    echo "  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.gen_field_data \\"
    echo "      --configs 'input/generated/training_*.toml' --timesteps 200 --output-dir output/field_data"
    echo "then scp output/field_data to the cluster."
    exit 1
fi

python3 -m swiftcfd.machineLearning.train_unet \
        --data-dir output/field_data \
        --target delta \
        --epochs "$EPOCHS" \
        --output-dir output/run_unet

echo "=== done: model in output/run_unet ==="
