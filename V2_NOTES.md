# NS ML-Hybrid — v2 notes: why change, what's new, how to run

This document accompanies two new files (your old code is untouched):

- `swiftcfd/machineLearning/train_residual_ns.py` — trains the model on the
  **pressure correction** `delta = p^(n+1) - p^n` (Approach #1, "SIMPLE-PINN").
- `swiftcfd/machineLearning/hybrid_solver_ns_petsc_v2.py` — a **controlled
  preconditioner × initial-guess sweep** that produces the honest
  "iterations-vs-preconditioner" curve (Approach #2 as a *reported finding*).

---

## 1. What was done so far, and why it needs to change

**Pipeline built (heat eq → NS):**
- Fractional-step explicit NS solver; pressure Poisson solved by Gauss–Seidel
  (GS), then by **PETSc** (CG/BiCGStab + GAMG/ILU/…). Ghost-cell BCs.
- A per-cell **MLP** predicts `p^(n+1)` from a 21-feature local stencil
  (u,v,p at n, plus centre values at n-1, n-2). Used as the Krylov **initial
  guess**.
- Reproducible training (`--seed`), surrogate eval (`evaluate_warmstart`),
  HPC training path (PETSc-free venv), dataset of ~20+3 lid-driven cases.

**Results / what we learned:**

| Setting | Baseline | ML warm-start result |
|---|---|---|
| GS pressure solve, transient (sp=10) | prev-step guess | **ML −51% iters (good)** |
| GS, near-steady (sp=40) | prev-step guess | ML loses (−83%); handled by skip-logic |
| **PETSc + GAMG** | prev-step guess | **~no gain ("looks useless")** |

Two root causes, both structural — **not bugs**:

1. **Wrong lever for a Krylov solver.** CG/BiCGStab convergence is set by the
   *spectrum of the preconditioned operator* `κ(M·A)`, **not** the initial
   guess. A better guess only lowers the *starting* residual → saves a roughly
   constant few iterations. With GAMG converging in ~10 iters there is nothing
   to win. GS has no spectral acceleration, so there the guess mattered a lot —
   that's why GS showed +51% and PETSc shows ~0%.
2. **Wrong objective + wrong architecture.** Training minimised *pointwise*
   `||p_pred − p_true||`, but solver cost depends on the *residual*
   `||∇²p − rhs||`. And a **local stencil** MLP cannot represent the **global
   (elliptic)** coupling of pressure — the solution at a point depends on the
   whole domain — so its residual is intrinsically high.

**Conclusion:** "ML initial guess for a GAMG-preconditioned Krylov solve" is a
near-dead end *by the math*. We must change the **target** (learn the
correction, not the full field), change how we **report** (show the gain as a
function of preconditioner strength), and ultimately change the **architecture**
(local MLP → global multiscale CNN/GNN).

---

## 2. The three papers, mapped to our problem

1. **SIMPLE-PINN (`p = p* + p'`)** — learn the *correction*, not the full
   pressure. Smaller, smoother target; can't be worse than prev-step; encodes
   velocity–pressure coupling. → `train_residual_ns.py`.
2. **Learned preconditioners for CG** — attack the *right* lever (the spectrum
   `M·A`), not the initial guess. A GNN emits a sparse approximate inverse /
   factorisation; trained to cluster eigenvalues, not to be pointwise accurate.
   Honest caveat: beating GAMG is hard — target *simple* preconditioners.
3. **Multiscale GNN for the pressure Poisson** — downsample/upsample = learned
   multigrid, so the output respects *global* boundary constraints. Directly
   fixes our locality problem. On our uniform grid a **U-Net CNN** is the
   pragmatic equivalent. → the main thrust (#3, next).

---

## 3. Where is YOUR contribution? (the PhD gap)

Reproducing any one paper is not a thesis. The defensible novelties available
to you, in increasing ambition:

- **G1 — A controlled "when does an ML warm-start actually pay off?" study.**
  The papers each fix one solver. *Nobody cleanly reports ML-guess benefit as a
  function of preconditioner strength on the same trajectory.* Your `v2` sweep
  produces exactly that curve (NONE→JACOBI→ILU→SOR→GAMG). The result
  "preconditioner strength erases ML-guess gains" is a genuine, citable,
  slightly-contrarian finding. **This is your cheapest real contribution and it
  already runs.**
- **G2 — Residual/correction warm-start + cost-aware switching.** Combine
  Approach #1 with a *learned decision* of when to invoke ML vs prev-step
  (your skip-logic, but learned/threshold-tuned and quantified end-to-end in
  wall-clock, not just iterations). Net-time speedup including inference cost is
  rarely reported honestly.
- **G3 — Architecture transfer + generalisation.** Multiscale model (#3) but
  studied for **out-of-distribution generalisation** across Re / BC / grid that
  the single-case papers don't test, using your 20+3 dataset and held-out cases.
- **G4 — Learned preconditioner that targets the *cheap* regime** (beats
  Jacobi/ILU at low setup cost, approaches GAMG) — highest risk, highest
  ceiling.

Recommended framing for the thesis chapter: lead with **G1** (negative/contrarian
result, fully controlled), then **G2/G3** as the constructive answer. A
well-characterised negative result + a principled fix is a strong, honest story.

---

## 4. Step-by-step: exact commands (Docker)

Assumes you already have training data + a baseline (`output/run_dataonly/...`).

### Step A — Quick win: the honest preconditioner sweep (no retraining)

Run the sweep with your **existing** data-only model to get the headline curve:

```
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.hybrid_solver_ns_petsc_v2 \
  --config input/generated/val_01.toml \
  --model output/run_dataonly/pinn_model_mlp.pth \
  --norm  output/run_dataonly/norm_params_mlp.pth \
  --solver BCGS --pc-list NONE,JACOBI,ILU,SOR,GAMG \
  --timesteps 200 --spinup 10 \
  --output-dir output/sweep_dataonly
```

Output: `output/sweep_dataonly/sweep_table.txt` (printed table) and `sweep.csv`.
Read the **"ML vs prev"** column: it should shrink toward 0% as the
preconditioner strengthens. That single table is result **G1**.

> Note: use `--solver BCGS` whenever the `--pc-list` includes non-symmetric
> preconditioners (ILU, SOR). Use `--solver CG` only for a symmetric subset
> (`--pc-list NONE,JACOBI,GAMG`) — CG assumes an SPD preconditioned operator.

### Step B — Train the residual (correction) model — Approach #1

```
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.train_residual_ns \
  --input-variables u,v,p --output-variable p \
  --epochs 200 --weight-pde 0 \
  --output-dir output/run_residual
```

Watch the printed line `target std: full p=… -> delta=…` — the delta std should
be much smaller (that's the point). Produces
`output/run_residual/pinn_model_mlp.pth` + `norm_params_mlp.pth` (tagged
`residual_target=True`).

### Step C — Sweep again with the residual model and compare

```
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.hybrid_solver_ns_petsc_v2 \
  --config input/generated/val_01.toml \
  --model output/run_residual/pinn_model_mlp.pth \
  --norm  output/run_residual/norm_params_mlp.pth \
  --solver BCGS --pc-list NONE,JACOBI,ILU,SOR,GAMG \
  --timesteps 200 --spinup 10 \
  --output-dir output/sweep_residual
```

Compare `output/sweep_dataonly/sweep.csv` vs `output/sweep_residual/sweep.csv`.
Expectation: residual model's `ml` column is ≤ `prev` everywhere (it can't be
worse than the previous step by construction), and beats the data-only model in
the weak-preconditioner columns.

### Step D (optional) — pure preconditioner study, no ML

```
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.hybrid_solver_ns_petsc_v2 \
  --config input/generated/val_01.toml \
  --solver CG --pc-list NONE,JACOBI,GAMG --timesteps 100 \
  --output-dir output/sweep_pc_only
```

---

## 5. Pipeline of the new code (how it actually runs)

### `train_residual_ns.py`
```
DataManager.get_training_data("u,v,p")        # same loader as before
        │   per-variable x(7 cols)=[c^n,e,w,n,s, c^{n-1}, c^{n-2}], y(5)=p^{n+1} stencil
        ▼
build_residual_targets(...)                   # y  ->  y - x[:,0:5]  == p^{n+1} - p^n  (delta)
        ▼
create_model("mlp", input_size=21, output_size=5, equation_type="ns")
        ▼
model.train_network(..., weight_pde=0)        # REUSED trainer: normalise delta, MSE, early stop
        ▼
save pinn_model_mlp.pth + norm_params_mlp.pth
torch.load(norm).update(residual_target=True) # tag so the solver adds p^n back
```
Key idea: normalisation stats are computed on the **delta**, so `MLInference`
already returns deltas at inference; the solver only has to add `p^n`.

### `hybrid_solver_ns_petsc_v2.py` (sweep)
```
load_ns_config(toml)  ─►  cfg (grid, BCs, nu/rho/dt, tol)
            │
            ▼  build ONE PetscPressurePoisson per preconditioner (matrix + pc set up once)
solvers = {NONE, JACOBI, ILU, SOR, GAMG}
            │
            ▼  single physical trajectory, advanced by (ref_pc=GAMG, init=prev):
for each timestep:
    u*,v* = _momentum_step(...)                       # predictor (reused)
    ml_guess = p^n + delta_pred   (or p_pred)         # if model given & step>=2
    guesses = { zero, prev=p^n, ml }
    for pc in pc_list:
        solver[pc].build_rhs(u*,v*)                   # identical b for all inits
        for init in guesses:
            records[pc,init] += _count_iters(...)     # solve, COUNT iters, discard solution
    p = ref.solve(p^n, u*, v*)                        # the real solve advances physics
    u,v = _velocity_correction(...); shift history
            │
            ▼
print_and_save → sweep_table.txt + sweep.csv
```
Why this is rigorous: every `(pc, init)` cell solves the **same** linear systems
along the **same** trajectory, so differences are attributable *only* to the
preconditioner and the initial guess — a clean controlled experiment.

---

## 6. G1 / G2 / G3 — files, commands, status

Key empirical result (val_01, 200 steps, spinup 10) confirming the thesis:

| pc | prev (beat me) | data-only ml | residual ml |
|---|---|---|---|
| NONE | 168.8 | 212.7 (−26%) | 194.1 (−15%) |
| JACOBI | 159.6 | 205.0 (−28%) | 185.3 (−16%) |
| GAMG | 11.5 | 17.9 (−56%) | 14.4 (−26%) |

Read: the **data-only ml column ≈ the zero column** (full-field prediction is no
better than a cold start); **residual learning recovers ~half** the warm-start
value; and the absolute gap shrinks as the preconditioner strengthens.

### G1 — controlled "when does ML warm-start pay off?" (DONE)
- `hybrid_solver_ns_petsc_v2.py` → `sweep.csv` per model.
- `plot_sweep.py` turns one/many `sweep.csv` into the publication figure.
```
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.plot_sweep \
  --csv data-only=output/sweep_dataonly/sweep.csv \
        residual=output/sweep_residual/sweep.csv \
  --out output/sweep_compare.png
```
- Transient regime (where ML can actually beat prev): add `--spinup 0 --timesteps 30`.

### G2 — cost-aware warm-start in WALL-CLOCK (DONE)
- `g2_costaware.py` compares prev / ml-always / ml-switch including inference
  time; ml-switch only invokes the net when `rms(p^n,p^{n-1})` > threshold.
```
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.g2_costaware \
  --config input/generated/val_01.toml \
  --model output/run_residual/pinn_model_mlp.pth \
  --norm  output/run_residual/norm_params_mlp.pth \
  --solver BCGS --pc JACOBI --timesteps 200
```
Honest read: speedup > 1 means ML wins *after* paying inference. With a strong
pc (GAMG) it will usually be <1 — that's the finding, not a bug.

### RNN warm-start model (DONE, temporal — NOT a fix for spatial locality)
- `model/recurrentNeuralNetwork.py` rewritten: reshapes the 21 features into a
  length-3 TIME sequence (n-2,n-1,n) and decodes the last hidden state. Works for
  ns (3 vars) and heat (1 var). Train via the residual trainer with `--model rnn`:
```
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.train_residual_ns \
  --model rnn --input-variables u,v,p --output-variable p \
  --epochs 200 --weight-pde 0 --output-dir output/run_residual_rnn
# then sweep it like any other model (it saves pinn_model_rnn.pth / norm_params_rnn.pth)
```
NOTE: RNN adds temporal modelling but is still per-cell / spatially local, so it
cannot beat prev where the MLP can't either. The spatial fix is G3.

### G3 — whole-field U-Net (SCAFFOLDED + verified to run; needs full training)
Three new pieces (the per-cell CSV pipeline can't feed a U-Net):
- `gen_field_data.py` — runs the NS solver, dumps whole-field snapshots
  `[u,v,p,rhs]` + target `p^{n+1}`/`delta` as `fields_*.npz`.
- `model/uNet.py` — NCHW U-Net (learned multigrid); handles any grid size.
- `train_unet.py` — loads npz, per-channel normalise, trains, saves
  `unet_model.pth` + `unet_norm.pth`.
```
# 1) generate fields for several cases
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.gen_field_data \
  --configs "input/generated/training_*.toml" --timesteps 200 --output-dir output/field_data
# 2) train the U-Net (residual target by default)
docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.train_unet \
  --data-dir output/field_data --epochs 100 --output-dir output/run_unet
```
STILL TODO for G3 (the remaining integration): add a whole-field U-Net guess
branch to the sweep solver (predict the full p field at once instead of per-cell),
then evaluate it with the same v2 sweep. This is where the spatial-locality
ceiling should finally break in the weak/medium-pc columns.
```
