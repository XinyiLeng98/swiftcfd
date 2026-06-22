# Running swiftCFD_ML training on the HPC

The work is split across three machines, and **only training runs on the HPC**:

| Stage | Where | Why |
|-------|-------|-----|
| 1. Data generation (CFD solver) | your laptop (Docker) | needs PETSc; already done — `output/training_01..20/` |
| 2. **Model training** | **HPC (GPU)** | expensive; the part you offload |
| 3. Bring back `*.pth` | laptop | small files |
| 4. Hybrid solver / evaluation | your laptop (Docker) | needs PETSc |

**Key fact:** training imports only **torch / numpy / pandas** — *no PETSc*.
(`swiftcfd/__init__.py` now loads the PETSc solver lazily, so the package
imports fine on a PETSc-free machine.) So the HPC env is a plain venv — exactly
your original workflow, no PETSc compile, no conda needed.

---

## 1. What to upload

| Item | Path |
|------|------|
| Code | `swiftcfd/`, `swiftcfd.py`, `pyproject.toml` |
| Job scripts | `hpc/` (`submit_mlp.sh`, `train.slurm`, `requirements-train.txt`) |
| **Training data** | `output/training_*/trainingData_{u,v,p}.csv` + `output/training_*/simulationParameters.csv` |

Do **not** upload: `output/run_*`, `*.png`, `*.dat`, `trainingData_T.csv`, `.git/`.

```bash
# --- on your laptop, from the repo root ---
# CSVs compress ~10x, so these tarballs are small
tar czf swiftcfd_code.tgz swiftcfd swiftcfd.py pyproject.toml hpc
tar czf swiftcfd_data.tgz output/training_*/trainingData_{u,v,p}.csv \
                          output/training_*/simulationParameters.csv

scp swiftcfd_code.tgz swiftcfd_data.tgz USER@hpc.address:~/swiftcfd_ml/
```
> ⚠️ On the HPC, extract these **at `~/swiftcfd_ml/`, NOT inside `hpc/`**:
> ```bash
> cd ~/swiftcfd_ml && tar xzf swiftcfd_code.tgz && tar xzf swiftcfd_data.tgz
> ```
> (Extracting inside `hpc/` makes a confusing nested duplicate.)

---

## 2. Run the training

`submit_mlp.sh` builds the venv automatically on first run (creates `venv/`,
pip-installs torch + numpy + pandas) and reuses it afterwards.

Edit `--partition` / `--account` in the script header to your cluster's values,
then:

```bash
cd ~/swiftcfd_ml
sbatch hpc/submit_mlp.sh                 # ONE data-only model -> output/run_dataonly
# or the full sweep (6 models in parallel: seeds 0/1/2 + weights 1e-7/1e-6/1e-5):
sbatch --array=0-5 hpc/train.slurm

squeue -u $USER                          # watch the queue
tail -f hpc/logs/mlp_*.out               # follow the log; look for "Device: cuda"
```

Each run writes `output/<label>/pinn_model_mlp.pth` + `norm_params_mlp.pth`.

---

## 3. Bring the models back, evaluate on your laptop

```bash
# --- on your laptop ---
rsync -avz 'USER@hpc.address:~/swiftcfd_ml/output/run_*' ./output/
```
Then run the warm-start metric / hybrid solver locally (needs Docker/PETSc) —
see `PHASE0_REPORT.md` §Appendix.
