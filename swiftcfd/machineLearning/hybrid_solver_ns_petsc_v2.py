"""
NS ML-Hybrid pressure solve -- v2 (honest preconditioner study).

Why v2 exists
-------------
v1 (`hybrid_solver_ns_petsc.py`) asked "does an ML initial guess beat the
previous-step guess for one Krylov+preconditioner choice?".  For a strong
preconditioner (GAMG) the answer is ~no, and that is EXPECTED: a Krylov
solver's iteration count is governed by the spectrum of the *preconditioned*
operator, not by the initial guess.  A better guess only lowers the *starting*
residual and therefore saves a roughly constant handful of iterations -- which
is negligible when GAMG already converges in ~10.

v2 turns that into a publishable, controlled experiment.  It sweeps the
pressure-Poisson solve over

    {preconditioners}  x  {initial-guess strategies}

on a SINGLE shared physical trajectory, so every cell of the resulting table is
solving the *same* linear systems and differs only in (pc, init).  The output
is the "iterations vs preconditioner" curve that makes the headline finding
explicit:

    The weaker the preconditioner, the more an ML initial guess helps;
    a strong preconditioner (GAMG) erases the gain.

It also supports Approach #1 (residual learning): if the model's norm file is
tagged `residual_target=True`, the ML guess is reconstructed as
`p_guess = p^n + delta_pred` instead of `p_guess = p_pred`.

Initial-guess strategies
------------------------
  zero : x0 = 0                          (cold start; worst case)
  prev : x0 = p^n                        (standard CFD warm start; strong)
  ml   : x0 = ML prediction              (full-field or residual, per norm tag)

Run a full sweep (Docker):
  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.hybrid_solver_ns_petsc_v2 \
      --config input/generated/val_01.toml \
      --model output/run_residual/pinn_model_mlp.pth \
      --norm  output/run_residual/norm_params_mlp.pth \
      --solver BCGS --pc-list NONE,JACOBI,ILU,SOR,GAMG \
      --timesteps 200
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
from petsc4py import PETSc

from swiftcfd.machineLearning.hybrid_solver_ns import (
    load_ns_config,
    apply_all_bc,
    _apply_one_bc,
    _is_all_neumann_p,
    rms,
    _momentum_step,
    _velocity_correction,
)
from swiftcfd.machineLearning.hybrid_solver_ns_petsc import PetscPressurePoisson
from swiftcfd.machineLearning.hybrid_solver import MLInference


# ══════════════════════════════════════════════════════════════════════════════
# ML pressure guess (residual-aware)
# ══════════════════════════════════════════════════════════════════════════════

def ml_pressure_guess(ml, residual, bc, dx, dy, nx, ny, all_neumann,
                      u_n, v_n, p_n, u_nm1, v_nm1, p_nm1, u_nm2, v_nm2, p_nm2):
    """21-feature stencil -> centre value.  If `residual`, the network output is
    the correction delta and the guess is p^n + delta; otherwise it is the full
    predicted pressure."""
    rows, coords = [], []
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            rows.append([
                u_n[i, j],   u_n[i+1, j],  u_n[i-1, j],  u_n[i, j+1],  u_n[i, j-1],
                u_nm1[i, j], u_nm2[i, j],
                v_n[i, j],   v_n[i+1, j],  v_n[i-1, j],  v_n[i, j+1],  v_n[i, j-1],
                v_nm1[i, j], v_nm2[i, j],
                p_n[i, j],   p_n[i+1, j],  p_n[i-1, j],  p_n[i, j+1],  p_n[i, j-1],
                p_nm1[i, j], p_nm2[i, j],
            ])
            coords.append((i, j))

    preds = ml.predict(np.array(rows, dtype=np.float32))   # (N, 5)
    p_guess = p_n.copy()
    for idx, (i, j) in enumerate(coords):
        if residual:
            p_guess[i, j] = p_n[i, j] + preds[idx, 0]      # p^n + delta
        else:
            p_guess[i, j] = preds[idx, 0]                  # full prediction
    _apply_one_bc(p_guess, bc, dx, dy, "p")
    if all_neumann:
        p_guess -= p_guess[1:-1, 1:-1].mean()
    return p_guess


# ══════════════════════════════════════════════════════════════════════════════
# Whole-field U-Net guess (G3)  --  ONE forward pass for the entire grid
# ══════════════════════════════════════════════════════════════════════════════

class UNetInference:
    """Loads unet_model.pth + unet_norm.pth and predicts the whole pressure
    (or correction) field in a single forward pass."""

    def __init__(self, model_path, norm_path):
        from swiftcfd.machineLearning.model.uNet import UNet
        d = torch.load(norm_path, weights_only=False)
        self.X_mean = torch.as_tensor(d["X_mean"], dtype=torch.float32)
        self.X_std = torch.as_tensor(d["X_std"], dtype=torch.float32)
        self.Y_mean = float(d["Y_mean"])
        self.Y_std = float(d["Y_std"])
        self.channels = d.get("channels", ["u", "v", "p", "rhs"])
        self.residual = bool(d.get("residual_target", d.get("target") == "delta"))
        self.model = UNet(in_channels=int(d["in_channels"]), out_channels=1,
                          base=int(d["base"]), levels=int(d["levels"]))
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model.eval()
        print(f"  U-Net loaded: channels={self.channels} base={d['base']} "
              f"levels={d['levels']} residual={self.residual}")

    def predict_field(self, X):          # X: (1, C, H, W) numpy -> (H, W)
        with torch.no_grad():
            Xt = torch.as_tensor(X, dtype=torch.float32)
            Xn = (Xt - self.X_mean) / (self.X_std + 1e-8)
            Y = self.model(Xn) * self.Y_std + self.Y_mean
        return Y.numpy()[0, 0]


def unet_field_guess(unet: UNetInference, bc, dx, dy, rho, dt, all_neumann,
                     u_n, v_n, p_n, u_star, v_star):
    """Build [u^n, v^n, p^n, rhs] on the interior, run the U-Net once, and fold
    the predicted field back onto the full ghosted grid."""
    interior = (slice(1, -1), slice(1, -1))
    rhs = (rho / dt) * (
        (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2.0 * dx) +
        (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2.0 * dy))
    X = np.stack([u_n[interior], v_n[interior], p_n[interior], rhs],
                 axis=0)[None]          # (1, 4, H, W)
    out = unet.predict_field(X)         # (H, W) delta or full p

    p_guess = p_n.copy()
    p_guess[interior] = (p_n[interior] + out) if unet.residual else out
    _apply_one_bc(p_guess, bc, dx, dy, "p")
    if all_neumann:
        p_guess -= p_guess[1:-1, 1:-1].mean()
    return p_guess


# ══════════════════════════════════════════════════════════════════════════════
# Guess-function builders (per-cell MLP/RNN  vs  whole-field U-Net)
# ══════════════════════════════════════════════════════════════════════════════

def make_percell_guess(ml, residual, cfg):
    bc, dx, dy = cfg["bc"], cfg["dx"], cfg["dy"]
    nx, ny = cfg["nx"], cfg["ny"]
    all_neumann = _is_all_neumann_p(bc)

    def fn(step, u_n, v_n, p_n, u_star, v_star, hist):
        if step < 2:                    # need 2 history levels for the stencil
            return None
        return ml_pressure_guess(
            ml, residual, bc, dx, dy, nx, ny, all_neumann,
            u_n, v_n, p_n,
            hist["u_nm1"], hist["v_nm1"], hist["p_nm1"],
            hist["u_nm2"], hist["v_nm2"], hist["p_nm2"])
    return fn


def make_unet_guess(unet, cfg):
    bc, dx, dy = cfg["bc"], cfg["dx"], cfg["dy"]
    rho, dt = cfg["rho"], cfg["dt"]
    all_neumann = _is_all_neumann_p(bc)

    def fn(step, u_n, v_n, p_n, u_star, v_star, hist):
        return unet_field_guess(unet, bc, dx, dy, rho, dt, all_neumann,
                                u_n, v_n, p_n, u_star, v_star)
    return fn


# ══════════════════════════════════════════════════════════════════════════════
# Iteration counting helper
# ══════════════════════════════════════════════════════════════════════════════

# int -> name map for KSP converged reasons ("CONVERGED_ATOL", "DIVERGED_DTOL"…)
_REASON_NAMES = {int(v): k for k, v in vars(PETSc.KSP.ConvergedReason).items()
                 if isinstance(v, int)}


def _count_iters(solver: PetscPressurePoisson, p_init: np.ndarray, atol=None):
    """Solve `solver`'s ALREADY-BUILT rhs from initial guess `p_init`.  The
    solution is discarded -- this is a measurement only.  Returns a dict:

      iters  : KSP iterations
      ok     : True only for a genuine converged solve
      r0     : ‖b − A·x0‖  (true starting residual of THIS guess)
      rfinal : ‖b − A·x‖   after the solve (must equal atol for every guess)
      reason : PETSc converged-reason name

    ok=False flags a BAD measurement: a non-finite (NaN/Inf) guess, a KSP that
    did not truly converge, or the fake win where PETSc reports 0 iterations
    with a converged reason although r0 > atol (loose-finish-line artifact --
    the source of the earlier bogus "ml = 0 iterations, +100%" result)."""
    if not np.isfinite(p_init).all():
        return {"iters": np.nan, "ok": False, "r0": np.nan,
                "rfinal": np.nan, "reason": "GUESS_NANORINF"}
    solver.load_guess(p_init)
    r0 = solver.true_residual_norm()
    solver.ksp.solve(solver.b, solver.x)
    reason = solver.ksp.getConvergedReason()  # >0 converged, <0 diverged/NaN
    iters = solver.ksp.getIterationNumber()
    rfinal = solver.true_residual_norm()
    name = _REASON_NAMES.get(int(reason), str(int(reason)))
    ok = reason > 0
    if ok and iters == 0 and atol is not None and r0 > atol:
        ok = False
        name += "+FAKE_ZERO"                  # "converged" at 0 iters, r0 > atol
    return {"iters": iters, "ok": ok, "r0": r0, "rfinal": rfinal, "reason": name}


# ══════════════════════════════════════════════════════════════════════════════
# Controlled sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(cfg, pc_list, krylov, rtol, num_t, guess_fn=None,
              ref_pc="GAMG", spinup=0, fair=True, target_reduction=None,
              curve=None):
    """
    Advance ONE physical trajectory and, at every step, measure KSP iterations
    for every (preconditioner, initial-guess) pair on the identical linear
    system.  The trajectory itself is advanced with (ref_pc, prev-p).

    `guess_fn(step, u_n, v_n, p_n, u_star, v_star, hist) -> guess or None`
    abstracts the model: per-cell MLP/RNN or whole-field U-Net.

    fair=True gives every initial guess the SAME finish line: the KSP stops on
    the UNPRECONDITIONED true residual, and the absolute tolerance is set PER
    STEP to `target_reduction * ‖b‖` (‖b‖ = the zero-guess residual r0).  A
    purely relative rtol·‖r0‖ criterion would instead demand a tighter absolute
    accuracy from a BETTER guess (smaller r0) -- systematically unfair to warm
    starts and capable of flipping the conclusion.

    curve = {"step": s, "pc": PC} (optional): at measured step s and that pc,
    record the residual-vs-iteration history for zero/prev/ml.

    Returns (records, init_names, curve_hist, diag, dp_hist):
      records  : dict[(pc, init)] -> list[iters per step]  (nan = invalid)
      diag     : list of per-(step, pc, init) dicts with r0 / rfinal / atol /
                 reason -- the Phase-0 fairness evidence and Phase-1 r0 data
      dp_hist  : list of (step, ‖p^{n+1} − p^n‖) along the trajectory (steady-
                 tail detector: if it decays to ~0, prev is unbeatable there)
    """
    nx, ny, dx, dy = cfg["nx"], cfg["ny"], cfg["dx"], cfg["dy"]
    nu, dt, rho = cfg["nu"], cfg["dt"], cfg["rho"]
    bc = cfg["bc"]
    target_reduction = target_reduction if target_reduction is not None else rtol

    # one solver instance per preconditioner (matrix + pc set up ONCE each)
    track = curve is not None
    def _mk(pc):
        return PetscPressurePoisson(cfg, solver=krylov, preconditioner=pc,
                                    rtol=rtol, unpreconditioned_norm=fair,
                                    track_history=track)
    solvers = {pc: _mk(pc) for pc in pc_list}
    if ref_pc not in solvers:
        solvers[ref_pc] = _mk(ref_pc)
    ref = solvers[ref_pc]
    curve_hist = {}

    init_names = ["zero", "prev"] + (["ml"] if guess_fn is not None else [])
    records = {(pc, ini): [] for pc in pc_list for ini in init_names}
    bad_count = {ini: 0 for ini in init_names}      # invalid (NaN/non-converged)
    bad_warned = {ini: False for ini in init_names}
    diag = []                                       # per-(step, pc, init) evidence
    dp_hist = []                                    # (step, ‖p^{n+1} − p^n‖)

    # fields + two history levels for the ML feature vector
    u = np.zeros((nx, ny)); v = np.zeros((nx, ny)); p = np.zeros((nx, ny))
    apply_all_bc(u, v, p, bc, dx, dy)
    hist = {k: f.copy() for k, f in
            (("u_nm1", u), ("v_nm1", v), ("p_nm1", p),
             ("u_nm2", u), ("v_nm2", v), ("p_nm2", p))}

    zero_field = np.zeros((nx, ny))
    t0 = time.time()

    for step in range(num_t):
        u_n, v_n, p_n = u.copy(), v.copy(), p.copy()

        u_star, v_star = _momentum_step(u, v, dx, dy, nu, dt, bc)
        if np.any(np.isnan(u_star)) or np.any(np.isnan(v_star)):
            raise RuntimeError(f"NaN at step {step}: dt={dt:.5f} unstable.")

        # build the ML guess once (shared by every preconditioner)
        ml_guess = None
        if guess_fn is not None:
            ml_guess = guess_fn(step, u_n, v_n, p_n, u_star, v_star, hist)

        guesses = {"zero": zero_field, "prev": p_n}
        if guess_fn is not None:
            guesses["ml"] = ml_guess if ml_guess is not None else p_n  # boot=prev

        # measure iterations for every (pc, init) on the SAME system ----------
        if step >= spinup:
            for pc in pc_list:
                solver = solvers[pc]
                solver.build_rhs(u_star, v_star)   # identical b across inits
                atol_step = float("nan")
                if fair:
                    # same absolute finish line for zero/prev/ml this step:
                    # target_reduction * ‖b‖  (‖b‖ = zero-guess residual r0)
                    atol_step = target_reduction * solver.rhs_norm()
                    solver.set_absolute_tol(atol_step)
                meas_here = {}
                for ini in init_names:
                    m = _count_iters(solver, guesses[ini],
                                     atol=atol_step if fair else None)
                    iters, ok = m["iters"], m["ok"]
                    meas_here[ini] = m
                    diag.append({"step": step, "pc": pc, "init": ini,
                                 "atol": atol_step, **m})
                    # a bad (NaN/non-converged) guess is recorded as nan, NOT as
                    # a misleadingly-perfect 0 -- it is a failure, not a win.
                    records[(pc, ini)].append(iters if ok else np.nan)
                    if not ok:
                        bad_count[ini] += 1
                        if not bad_warned[ini]:
                            print(f"  [!] '{ini}' guess INVALID (NaN/non-converged) "
                                  f"at step {step}, pc={pc} -- recorded as failure, "
                                  f"not 0. (further occurrences counted silently)")
                            bad_warned[ini] = True
                    if (curve is not None and step == curve["step"]
                            and pc == curve["pc"]):
                        curve_hist[ini] = np.asarray(solver.convergence_history(),
                                                     dtype=float)

                # decisive r0 diagnostic: is the ml guess ACTUALLY converged at
                # iter 0, or is atol just loose?  Print ‖b‖, atol, and each
                # init's true initial residual ‖r0‖ = ‖b - A·x0‖ (computed
                # explicitly by a mat-vec in _count_iters).
                if (curve is not None and step == curve["step"]
                        and pc == curve["pc"]):
                    bn = solver.rhs_norm()
                    at = target_reduction * bn if fair else float("nan")
                    print(f"\n  --- r0 DIAGNOSTIC (step {step}, pc {pc}) ---")
                    print(f"    ‖b‖ = {bn:.3e}"
                          + (f"   atol = {at:.3e} (= {target_reduction:.0e}·‖b‖)"
                             if fair else "   (legacy relative rtol)"))
                    for ini in init_names:
                        m = meas_here.get(ini)
                        if m is None:
                            continue
                        r0 = m["r0"]
                        tag = ("r0 is NaN/Inf -> BROKEN guess"
                               if not np.isfinite(r0) else
                               ("<= atol -> 0 iters (guess already 'converged')"
                                if fair and r0 <= at else "> atol -> must iterate"))
                        print(f"    {ini:>4}: r0 = {r0:.3e}   "
                              f"[{m['iters']} iters, {m['reason']}]   {tag}")
                    print(f"    (if ml's r0 is only ~1e-8x smaller than zero's, the")
                    print(f"     win is just a loose atol; tighten --target-reduction.)\n")

        # advance the real trajectory with the reference solver ---------------
        p, _, _ = ref.solve(p_n, u_star, v_star)
        u, v = _velocity_correction(u_star, v_star, p, dx, dy, rho, dt, bc)
        dp_hist.append((step, float(np.linalg.norm(p[1:-1, 1:-1]
                                                   - p_n[1:-1, 1:-1]))))

        hist["u_nm2"], hist["v_nm2"], hist["p_nm2"] = hist["u_nm1"], hist["v_nm1"], hist["p_nm1"]
        hist["u_nm1"], hist["v_nm1"], hist["p_nm1"] = u_n, v_n, p_n

        if num_t <= 10 or step % max(1, num_t // 10) == 0:
            print(f"  step={step:4d}/{num_t}  (advancing with {ref_pc}+prev)")

    elapsed = time.time() - t0
    print(f"\n  Sweep trajectory done in {elapsed:.1f}s "
          f"({num_t} steps, {len(pc_list)} pcs x {len(init_names)} inits)")
    n_meas = max(num_t - spinup, 0) * len(pc_list)
    for ini in init_names:
        if bad_count[ini]:
            print(f"  [!] '{ini}' produced {bad_count[ini]}/{n_meas} INVALID "
                  f"(NaN/non-converged) guesses -> shown as 'nan' in the table, "
                  f"NOT as a 0-iteration win.")
    return records, init_names, curve_hist, diag, dp_hist


def save_residual_curves(curve_hist, out_dir, meta):
    """Semilog plot of ‖r_k‖ vs iteration for zero/prev/ml on ONE step & pc.
    Same slope (set by the preconditioner spectrum) but vertically shifted by
    initial-guess quality is the healthy picture; an ml curve that starts lower
    but bends to a shallower slope means the ML guess excited slow-converging
    (high-frequency) modes."""
    if not curve_hist:
        return
    os.makedirs(out_dir, exist_ok=True)

    # tidy CSV (ragged -> one column block per init)
    csv_path = os.path.join(out_dir, "residual_curves.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["init", "iteration", "residual"])
        for ini, hist in curve_hist.items():
            for k, r in enumerate(hist):
                w.writerow([ini, k, f"{r:.8e}"])

    png_path = os.path.join(out_dir, "residual_curves.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"zero": "0.6", "prev": "tab:blue", "ml": "tab:red"}
        plt.figure(figsize=(7, 5))
        for ini, hist in curve_hist.items():
            plt.semilogy(range(len(hist)), hist, marker="o", ms=3,
                         color=colors.get(ini), label=f"{ini} (n={len(hist)-1})")
        plt.xlabel("KSP iteration")
        plt.ylabel(r"unpreconditioned residual $\|r_k\|_2$")
        plt.title(f"Residual decay  (pc={meta['pc']}, step={meta['step']}, "
                  f"{meta['krylov']})")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=140)
        plt.close()
        print(f"  Saved residual-decay figure: {png_path}")
    except Exception as e:
        print(f"  (residual curve CSV written; plot skipped: {e})")
    print(f"  Saved residual-decay data  : {csv_path}")


def print_fairness_check(diag, pc_list, init_names):
    """Phase-0 acceptance check.  Under the fair criterion every (pc, init)
    pair must (a) finish at the SAME absolute residual, i.e. rfinal <= atol for
    every converged solve, and (b) never score a 0-iteration 'win' unless its
    r0 genuinely started below atol.  Prints a per-(pc, init) summary and an
    overall PASS/FAIL verdict."""
    if not diag:
        return True
    print("\n" + "=" * 78)
    print("  PHASE-0 FAIRNESS CHECK  (same finish line for every guess?)")
    print("=" * 78)
    print(f"  {'pc':<10}{'init':<8}{'max rfinal/atol':>16}{'fake 0-iter':>13}"
          f"{'true 0-iter':>13}{'not conv.':>11}")
    print("  " + "-" * 76)
    ok_all = True
    for pc in pc_list:
        for ini in init_names:
            cells = [d for d in diag if d["pc"] == pc and d["init"] == ini]
            if not cells:
                continue
            ratios = [d["rfinal"] / d["atol"] for d in cells
                      if d["ok"] and np.isfinite(d["rfinal"]) and d["atol"] > 0]
            worst = max(ratios) if ratios else float("nan")
            fake = sum("FAKE_ZERO" in d["reason"] for d in cells)
            true0 = sum(d["ok"] and d["iters"] == 0 for d in cells)
            nconv = sum((not d["ok"]) and "FAKE_ZERO" not in d["reason"]
                        for d in cells)
            # the KSP stops at the first iterate BELOW atol, so the ratio is
            # <=1 up to the drift between the Krylov method's recursive
            # residual and the recomputed true one; 5% slack covers rounding,
            # anything larger means the finish line genuinely moved.
            bad = fake > 0 or (np.isfinite(worst) and worst > 1.05)
            ok_all &= not bad
            flag = "  <-- UNFAIR" if bad else ""
            print(f"  {pc:<10}{ini:<8}{worst:>16.3f}{fake:>13d}"
                  f"{true0:>13d}{nconv:>11d}{flag}")
    print("  " + "-" * 76)
    if ok_all:
        print("  PASS: every converged solve finished at/below the shared atol and")
        print("  no guess scored a fake 0-iteration win. Iteration counts are")
        print("  comparable across preconditioners AND initial guesses.")
    else:
        print("  FAIL: at least one (pc, init) cell violated the shared finish")
        print("  line -- the sweep table above is NOT trustworthy. Check that")
        print("  fair mode is on and atol is anchored to ‖b‖ of the cold start.")
    print("=" * 78)
    return ok_all


def save_diagnostics(diag, dp_hist, out_dir):
    """diagnostics.csv : per-(step, pc, init) r0 / iters / rfinal / atol /
    reason -- the raw evidence for the Phase-1 decision table.
    dp_norm.csv : ‖p^{n+1} − p^n‖ per trajectory step (steady-tail detector)."""
    os.makedirs(out_dir, exist_ok=True)
    dpath = os.path.join(out_dir, "diagnostics.csv")
    with open(dpath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "pc", "init", "r0", "iters", "rfinal", "atol",
                    "reason", "converged"])
        for d in diag:
            w.writerow([d["step"], d["pc"], d["init"], f"{d['r0']:.8e}",
                        d["iters"], f"{d['rfinal']:.8e}", f"{d['atol']:.8e}",
                        d["reason"], int(d["ok"])])
    ppath = os.path.join(out_dir, "dp_norm.csv")
    with open(ppath, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "dp_norm"])
        for step, dp in dp_hist:
            w.writerow([step, f"{dp:.8e}"])
    print(f"  Saved: {dpath}  and  {ppath}")


def print_and_save(records, pc_list, init_names, out_dir, meta):
    os.makedirs(out_dir, exist_ok=True)

    def avg(pc, ini):
        # nanmean: average the VALID measurements; invalid (NaN/non-converged)
        # guesses are excluded, not silently counted as 0.
        xs = records[(pc, ini)]
        if not xs or np.all(np.isnan(xs)):
            return float("nan")
        return float(np.nanmean(xs))

    def n_bad(pc, ini):
        xs = records[(pc, ini)]
        return int(np.sum(np.isnan(xs))) if xs else 0

    header = f"  {'preconditioner':<16}" + "".join(f"{n:>10}" for n in init_names)
    if "ml" in init_names and "prev" in init_names:
        header += f"{'ML vs prev':>14}"

    lines = [
        "\n" + "=" * 78,
        "  ITERATIONS-VS-PRECONDITIONER  (avg KSP iters per step, lower=better)",
        "=" * 78,
        f"  Krylov={meta['krylov']}  rtol={meta['rtol']:.1e}  "
        f"grid={meta['nc_x']}x{meta['nc_y']}  steps={meta['steps_measured']}  "
        f"residual_ML={meta['residual']}",
        "",
        header,
        "  " + "-" * 76,
    ]
    any_bad = False
    for pc in pc_list:
        row = f"  {pc:<16}" + "".join(f"{avg(pc, n):>10.1f}" for n in init_names)
        if "ml" in init_names and "prev" in init_names:
            pv, ml = avg(pc, "prev"), avg(pc, "ml")
            red = (1.0 - ml / pv) * 100 if (pv and not np.isnan(ml)) else float("nan")
            row += f"{red:>+13.1f}%"
            if n_bad(pc, "ml"):
                row += f"   ({n_bad(pc, 'ml')} invalid)"
                any_bad = True
        lines.append(row)
    lines.append("=" * 78)
    if any_bad:
        lines.append("  'nan' / 'invalid' = the ML guess was NaN/non-converged for")
        lines.append("  those steps (a FAILURE misreported by PETSc as 0 iters).")
        lines.append("  Investigate the model/normalisation, NOT a 100% speed-up.")
    else:
        lines.append("  Read it this way: 'ML vs prev' shrinks toward 0% as the")
        lines.append("  preconditioner gets stronger -> the ML guess matters less.")
    lines.append("")
    text = "\n".join(lines)
    print(text)

    with open(os.path.join(out_dir, "sweep_table.txt"), "w") as fh:
        fh.write(text)

    # tidy CSV for plotting
    with open(os.path.join(out_dir, "sweep.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["preconditioner", "init", "avg_iters", "std_iters",
                    "n_valid", "n_invalid"])
        for pc in pc_list:
            for ini in init_names:
                xs = records[(pc, ini)]
                valid = [x for x in xs if not np.isnan(x)]
                w.writerow([pc, ini,
                            f"{np.mean(valid):.4f}" if valid else "",
                            f"{np.std(valid):.4f}" if valid else "",
                            len(valid), int(np.sum(np.isnan(xs))) if xs else 0])
    print(f"  Saved: {out_dir}/sweep_table.txt  and  {out_dir}/sweep.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="v2: controlled preconditioner x initial-guess sweep for the "
                    "NS pressure Poisson solve.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None,
                    help="ML model .pth (optional; omit for a pure pc study)")
    ap.add_argument("--norm", default=None, help="norm params .pth")
    ap.add_argument("--unet", action="store_true",
                    help="force whole-field U-Net path (auto-detected from the "
                         "norm file otherwise)")
    ap.add_argument("--solver", default="BCGS",
                    choices=["CG", "BCGS", "GMRES", "RICHARDSON"],
                    help="Krylov method held FIXED across the sweep (default BCGS, "
                         "which is valid for every pc; use CG only with a symmetric "
                         "pc subset like NONE,JACOBI,GAMG)")
    ap.add_argument("--pc-list", default="NONE,JACOBI,ILU,SOR,GAMG",
                    help="Comma-separated preconditioners to sweep")
    ap.add_argument("--ref-pc", default="GAMG",
                    help="Preconditioner used to advance the physical trajectory")
    ap.add_argument("--tol", type=float, default=None, help="KSP rtol override")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--spinup", type=int, default=0,
                    help="Skip measuring the first N steps (let the flow develop)")
    ap.add_argument("--legacy-rtol", action="store_true",
                    help="Use the OLD unfair criterion (relative rtol·‖r0‖, "
                         "preconditioned norm).  Default is the fair per-step "
                         "absolute finish line on the true residual.")
    ap.add_argument("--target-reduction", type=float, default=None,
                    help="Fair mode: residual reduction factor defining the "
                         "per-step absolute finish line atol=factor·‖b‖ "
                         "(default: the KSP rtol, e.g. 1e-8)")
    ap.add_argument("--curve-step", type=int, default=None,
                    help="Record the residual-vs-iteration decay curve for "
                         "zero/prev/ml at this measured step (-> residual_curves.png)")
    ap.add_argument("--curve-pc", default=None,
                    help="Preconditioner for the residual-decay curve "
                         "(default: first entry of --pc-list)")
    ap.add_argument("--output-dir",
                    default=os.path.join("output", "hybrid_ns_petsc_v2"))
    args = ap.parse_args()

    cfg = load_ns_config(args.config)
    if args.tol is not None:       cfg["tol_p"] = args.tol
    if args.dt is not None:        cfg["dt"] = args.dt
    if args.timesteps is not None: cfg["timesteps"] = args.timesteps

    # explicit-time-integration stability clamp (same as v1)
    dt_diff = cfg["dx"] ** 2 / (4.0 * cfg["nu"])
    dt_cfl = min(cfg["dx"], cfg["dy"])
    dt_stable = min(dt_diff, dt_cfl) * 0.4
    if cfg["dt"] > dt_stable:
        print(f"\n  AUTO-CLAMP: dt={cfg['dt']:.5f} -> {dt_stable:.5f} (explicit limit).")
        cfg["dt"] = dt_stable

    rtol = cfg["tol_p"]
    pc_list = [s.strip().upper() for s in args.pc_list.split(",") if s.strip()]
    num_t = cfg["timesteps"]

    guess_fn, residual, kind = None, False, "none"
    if args.model:
        if not args.norm:
            ap.error("--norm is required when --model is given")
        norm = torch.load(args.norm, weights_only=False)
        is_unet = args.unet or ("in_channels" in norm)   # auto-detect whole-field model
        if is_unet:
            print("\n--- Loading U-Net (whole-field) ---")
            unet = UNetInference(args.model, args.norm)
            residual, kind = unet.residual, "unet"
            guess_fn = make_unet_guess(unet, cfg)
        else:
            print("\n--- Loading per-cell ML model ---")
            ml = MLInference(args.model, args.norm)
            residual, kind = bool(norm.get("residual_target", False)), "per-cell"
            guess_fn = make_percell_guess(ml, residual, cfg)
        print(f"  model kind = {kind}   residual_target = {residual} "
              f"({'p_guess = p^n + delta' if residual else 'p_guess = p_pred'})")

    fair = not args.legacy_rtol
    target_reduction = args.target_reduction if args.target_reduction is not None else rtol
    curve = None
    if args.curve_step is not None:
        curve_pc = (args.curve_pc or pc_list[0]).upper()
        curve = {"step": args.curve_step, "pc": curve_pc}

    print(f"\n{'='*65}\n  NS pressure sweep (v2)\n{'='*65}")
    print(f"  Config : {args.config}")
    print(f"  Grid   : {cfg['nc_x']} x {cfg['nc_y']}  interior cells")
    # nullspace hygiene (Phase 0.4): record which case applies
    if _is_all_neumann_p(cfg["bc"]):
        print(f"  Nullsp : p is ALL-NEUMANN -> constant nullspace attached to A, "
              f"b projected, guesses mean-projected")
    else:
        print(f"  Nullsp : p has a DIRICHLET reference (e.g. fixed outlet) -> "
              f"system non-singular, no nullspace handling needed")
    print(f"  Krylov : {args.solver}  rtol={rtol}")
    print(f"  PCs    : {pc_list}   (trajectory driven by {args.ref_pc})")
    print(f"  Steps  : {num_t}  (spinup skip={args.spinup})")
    if fair:
        print(f"  Conv   : FAIR — true (unpreconditioned) residual, per-step "
              f"atol = {target_reduction:.1e}·‖b‖  (same finish line for all inits)")
    else:
        print(f"  Conv   : LEGACY — relative rtol·‖r0‖ (unfair to warm starts)")
    if curve:
        print(f"  Curve  : residual decay at step={curve['step']} pc={curve['pc']}")

    records, init_names, curve_hist, diag, dp_hist = run_sweep(
        cfg, pc_list, args.solver, rtol, num_t,
        guess_fn=guess_fn, ref_pc=args.ref_pc.upper(), spinup=args.spinup,
        fair=fair, target_reduction=target_reduction, curve=curve)

    steps_measured = max(num_t - args.spinup, 0)
    if fair:
        print_fairness_check(diag, pc_list, init_names)
    save_diagnostics(diag, dp_hist, args.output_dir)
    print_and_save(records, pc_list, init_names, args.output_dir, {
        "krylov": args.solver, "rtol": rtol,
        "nc_x": cfg["nc_x"], "nc_y": cfg["nc_y"],
        "steps_measured": steps_measured, "residual": residual,
    })
    if curve is not None:
        save_residual_curves(curve_hist, args.output_dir, {
            "krylov": args.solver, "pc": curve["pc"], "step": curve["step"],
        })


if __name__ == "__main__":
    main()
