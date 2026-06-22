# swiftCFD_ML — Local + HPC Workflow Guide

This project compares an **ML-warm-started** Navier–Stokes hybrid solver against
a **pure PDE** solver. There are two ways to run it, and you can use both:

| Mode | Machine | Needs PETSc? | Does what |
|------|---------|--------------|-----------|
| **Local (full pipeline)** | your laptop, via Docker | yes (in the image) | generate data → train → hybrid solver → compare |
| **HPC (training only)** | cluster, via a plain venv | **no** | train many epochs fast → produce `.pth` |

You keep the **training data on both machines**. The HPC only ever does the
heavy training; everything that needs the CFD solver (data generation, the
hybrid solver, the warm-start evaluation) stays local.

### Why the HPC environment is so simple
Training imports only **torch / numpy / pandas** — it never calls PETSc.
`swiftcfd/__init__.py` loads the PETSc-based solver *lazily* (wrapped in
`try/except`), so `import swiftcfd` works on a machine with no PETSc and the
`--train` path runs fine. That is why the HPC needs only a 3-line
`requirements-train.txt`, no PETSc compile, no conda, no container.

---

## Part A — Local: the whole pipeline

Everything local runs inside the Docker container (`docker compose run --rm
swiftcfd …`), which already has PETSc + torch installed. Run all commands from
the repo root.

### A0. One-time: build the image
```bash
docker compose build          # only needed once, or after changing the Dockerfile
```

### A1. Generate training data (CFD solver)
```bash
docker compose run --rm swiftcfd python3 swiftcfd/machineLearning/generate_bc.py \
  --case lid-driven --n-train 20 --n-val 3 --seed 42
```
- Writes configs `input/generated/training_01..20.toml` + `val_01..03.toml`.
- Runs each sim → `output/training_XX/trainingData_{u,v,p}.csv` (the data the
  model learns from). `val_*` cases are held out (no training CSV).
- `--seed 42` makes the random Reynolds numbers reproducible.
- Reruns overwrite existing cases. To change the Reynolds spread use
  `--nu-lo/--nu-hi` and `--lid-vel-lo/--lid-vel-hi`.

### A2. Train the model
```bash
docker compose run --rm swiftcfd python3 swiftcfd.py --train \
  --equation-type ns --model mlp \
  --input-variables u,v,p --output-variables p \
  --epochs 200 --seed 0 --weight-pde 0.0 --weight-data 1.0 \
  --output-dir output/run_dataonly
```
- Data is **auto-discovered** from `output/*/trainingData_{u,v,p}.csv` — there is
  no `--data` / `--config` flag for training.
- Produces `output/run_dataonly/pinn_model_mlp.pth` and `norm_params_mlp.pth`.
- `--weight-pde 0.0 --weight-data 1.0` = pure-data loss (the proven
  non-degenerate setting). Prints `Device: cpu` (or `cuda` if a GPU is present).

### A3. Evaluate warm-start quality (cheap, seconds)
```bash
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.evaluate_warmstart \
  --config input/generated/val_01.toml \
  --model output/run_dataonly/pinn_model_mlp.pth \
  --norm  output/run_dataonly/norm_params_mlp.pth \
  --spinup 10 --eval-steps 10 --tol 1e-4 --true-residual
```
- Reports GS iterations for ML init vs `prev` init vs `zero` init on a held-out
  case. `ML vs prev` must be **> 0** to be useful. Use this to pick models
  without paying for a full simulation. `--spinup 10 --true-residual` is the
  standard, discriminative protocol (see `PHASE0_REPORT.md`).

### A4. Hybrid solver — the ML-vs-PDE comparison (slow, the headline number)
```bash
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.hybrid_solver_ns \
  --config input/generated/val_01.toml \
  --model output/run_dataonly/pinn_model_mlp.pth \
  --norm  output/run_dataonly/norm_params_mlp.pth \
  --dt 0.005 --tol 1e-2
```
- Runs a full simulation, warm-starting the pressure solve with the ML
  prediction, and reports average iterations / speedup vs the pure PDE run.
- `--dt 0.005` is required: the base `dt=0.3` is unstable for this explicit
  solver (see `PHASE0_REPORT.md`).
- Plots/CSVs land in the output dir; visualise with
  `... -m swiftcfd.machineLearning.post_ns --results-dir output/hybrid_ns_results`.

---

## Part B — HPC: training only (many epochs, fast)

Use this when A2 is too slow (large `--epochs`, many seeds/weights). You copy
the **code + training data** up, train, and copy the **`.pth` back** to run A3/A4
locally.

### B1. Environment on the HPC — a plain venv

The cluster needs only Python + three pip packages. The file
`hpc/requirements-train.txt`:
```
torch
numpy
pandas
```
> If your cluster has GPUs, install a CUDA-matched torch build instead of the
> generic one. Check the cluster's CUDA version, then e.g.
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> (replace `cu121` with your version). The plain `torch` in the requirements
> file is the CPU/auto build — fine for CPU nodes.

The job scripts build this venv automatically on first run, so you normally
don't create it by hand. To do it manually:
```bash
module load python/3.10           # name varies by cluster: `module avail python`
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build
pip install -r hpc/requirements-train.txt
```

### B2. Which files to upload

| Upload | Why |
|--------|-----|
| `swiftcfd/` | the package (training code) |
| `swiftcfd.py` | training entry point |
| `pyproject.toml` | package metadata |
| `hpc/` | the job scripts + `requirements-train.txt` |
| `output/training_*/trainingData_{u,v,p}.csv` | the training data |
| `output/training_*/simulationParameters.csv` | per-case parameters used in the loss |

**Do not upload:** `output/run_*`, `*.png`, `*.dat`, `trainingData_T.csv`,
`.git/`, the Docker image, or PETSc.

### B3. Pack and copy (on your laptop)
```bash
# CSVs compress ~10x → small tarballs
tar czf swiftcfd_code.tgz swiftcfd swiftcfd.py pyproject.toml hpc
tar czf swiftcfd_data.tgz output/training_*/trainingData_{u,v,p}.csv \
                          output/training_*/simulationParameters.csv

scp swiftcfd_code.tgz swiftcfd_data.tgz USER@hpc.address:~/swiftcfd_ml/
```

### B4. Unpack on the HPC — at the repo root, NOT inside `hpc/`
```bash
cd ~/swiftcfd_ml
tar xzf swiftcfd_code.tgz        # -> swiftcfd/ swiftcfd.py pyproject.toml hpc/
tar xzf swiftcfd_data.tgz        # -> output/training_*/...
mkdir -p hpc/logs output
```
> ⚠️ Extracting inside `hpc/` creates a confusing nested duplicate. Always
> extract from `~/swiftcfd_ml`.

### B5. Submit the training job

Edit the resource header of `hpc/submit_mlp.sh` first — set your cluster's
`--partition` and `--account` (ask your HPC admin / docs):
```bash
sbatch hpc/submit_mlp.sh
```
What the SLURM directives mean:
- `--time=04:00:00` max walltime, `--cpus-per-task=8` cores, `--mem=32G` RAM
  (the CSVs load fully into memory), `--gres=gpu:1` request one GPU.

Monitor it:
```bash
squeue -u $USER                  # your queued/running jobs
tail -f hpc/logs/mlp_*.out       # follow the log; confirm it prints "Device: cuda"
```
Output: `output/run_dataonly/pinn_model_mlp.pth` + `norm_params_mlp.pth`.

**For many configurations at once** (e.g. seeds + physics-weights), use the job
array instead — six models in parallel:
```bash
sbatch --array=0-5 hpc/train.slurm
```
(The six configs are listed in `hpc/train.slurm`; edit them as you like.)

### B6. Bring the model back, run the solver locally
```bash
# on your laptop
rsync -avz 'USER@hpc.address:~/swiftcfd_ml/output/run_*' ./output/
```
Then run **A3** (evaluate) and **A4** (hybrid solver) locally on the returned
`.pth` — those need PETSc and stay on your laptop.

---

## Quick reference

| Task | Where | Command (from repo root) |
|------|-------|--------------------------|
| Generate data | local | `docker compose run --rm swiftcfd python3 swiftcfd/machineLearning/generate_bc.py --case lid-driven --n-train 20 --n-val 3 --seed 42` |
| Train | local | `docker compose run --rm swiftcfd python3 swiftcfd.py --train --equation-type ns --model mlp --input-variables u,v,p --output-variables p --epochs 200 --output-dir output/run_dataonly` |
| Train | HPC | `sbatch hpc/submit_mlp.sh` |
| Evaluate | local | `docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.evaluate_warmstart --config input/generated/val_01.toml --model <pth> --norm <pth> --spinup 10 --eval-steps 10 --tol 1e-4 --true-residual` |
| Hybrid solver | local | `docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.hybrid_solver_ns --config input/generated/val_01.toml --model <pth> --norm <pth> --dt 0.005 --tol 1e-2` |
| Pack for HPC | local | `tar czf swiftcfd_code.tgz swiftcfd swiftcfd.py pyproject.toml hpc` |
| Get models back | local | `rsync -avz 'USER@hpc:~/swiftcfd_ml/output/run_*' ./output/` |

### Files involved
- `hpc/requirements-train.txt` — torch/numpy/pandas (HPC training deps)
- `hpc/submit_mlp.sh` — single-model SLURM job (auto-builds the venv)
- `hpc/train.slurm` — multi-config SLURM job array
- `requirements.txt` (repo root) — the **full** local deps incl. PETSc (Docker only)
