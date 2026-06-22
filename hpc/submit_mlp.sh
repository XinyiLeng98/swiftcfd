#!/bin/bash
# ============================================================================
#  HPC training job for the NS lid-driven pressure warm-start MLP.
#
#  Trains ONE model and writes output/run_dataonly/{pinn_model_mlp.pth,
#  norm_params_mlp.pth}, which you copy back to your laptop for the hybrid
#  solver. Training needs ONLY torch/numpy/pandas — no PETSc, no CFD solver.
#
#  Submit:   sbatch hpc/submit_mlp.sh
# ============================================================================

# --- SLURM resource requests (the scheduler reads these #SBATCH lines) ------
#SBATCH --job-name=swiftcfd_mlp          # name shown in `squeue`
#SBATCH --output=hpc/logs/mlp_%j.out     # stdout  (%j = job id)
#SBATCH --error=hpc/logs/mlp_%j.err      # stderr
#SBATCH --time=04:00:00                  # max walltime (HH:MM:SS)
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --mem=32G                        # RAM (the CSVs load fully into memory)
#SBATCH --gres=gpu:1                     # 1 GPU (the code now uses it automatically)
# #SBATCH --partition=gpu                # <-- UNCOMMENT + set your cluster's GPU partition
# #SBATCH --account=YOUR_ACCOUNT         # <-- UNCOMMENT + set your allocation code

set -euo pipefail                        # stop on first error
mkdir -p hpc/logs output                 # make sure log + output dirs exist

# --- 1. Load a Python module (adjust the name to your cluster) --------------
module load python/3.10 2>/dev/null || true
# If your cluster needs a CUDA module for the GPU, load it too, e.g.:
# module load cuda/12.1 2>/dev/null || true

# --- 2. Create the venv once, reuse it afterwards ---------------------------
if [ ! -d venv ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    # GPU torch matching the cluster CUDA (change cu121 if needed); then numpy/pandas
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -r hpc/requirements-train.txt
else
    source venv/bin/activate
fi

# --- 3. Make `import swiftcfd` resolve (we run from the repo root) -----------
export PYTHONPATH=$PWD:$PYTHONPATH

# --- 4. Show the GPU (confirms it is visible to the job) ---------------------
nvidia-smi 2>/dev/null | head -15 || echo "(no GPU visible — training on CPU)"

# --- 5. Train.  Data is auto-discovered from output/*/trainingData_{u,v,p}.csv
#        --train            : run the training branch (not the CFD solver)
#        --equation-type ns : Navier-Stokes physics-loss definition
#        --model mlp        : multilayer perceptron
#        --input-variables  : features are built from u, v and p stencils
#        --output-variables : the network predicts the next-step pressure p
#        --epochs 200       : training epochs (early stopping may cut it short)
#        --seed 0           : reproducible run
#        --weight-pde 0.0   : pure data loss (the proven non-degenerate setting)
#        --weight-data 1.0  : data-loss weight
#        --output-dir       : where the .pth files are written
python3 swiftcfd.py --train \
        --equation-type ns \
        --model mlp \
        --input-variables u,v,p \
        --output-variables p \
        --epochs 200 \
        --seed 0 \
        --weight-pde 0.0 \
        --weight-data 1.0 \
        --output-dir output/run_dataonly

echo "=== done: models in output/run_dataonly ==="
