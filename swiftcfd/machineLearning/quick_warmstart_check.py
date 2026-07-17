"""
Quick Phase-0 warm-start acceptance check (channel case).

Answers ONE question in seconds, BEFORE any expensive sweep: is the
measurement fair?  It advances the channel trajectory a few spin-up steps,
then measures a SINGLE pressure solve with the three initial guesses
(zero / prev / ml) under TWO different preconditioners, with all four
Phase-0 fixes active in the fair criterion:

  0.1  unpreconditioned convergence norm (+ right PC side for GMRES)
  0.2  one absolute finish line per system: atol = target_reduction * ||b||
       (anchored to the cold start), rtol=1e-50, divtol=1e10
  0.3  setInitialGuessNonzero(True)
  0.4  nullspace hygiene if p is all-Neumann (channel: Dirichlet outlet ->
       non-singular, nothing to do; printed either way)

Acceptance criteria (from the Phase-0 plan):
  (a) every converged (guess, pc) pair finishes at the SAME absolute
      residual: rfinal <= atol (within 5% Krylov-recursion slack);
  (b) no guess scores 0 iterations with a converged reason unless its
      r0 = ||b - A x0|| genuinely started below atol.

Exit code 0 = PASS (the sweep numbers are trustworthy), 1 = FAIL.

Run (Docker, needs PETSc):
  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.quick_warmstart_check \
      --config input/generated/val_01.toml
or via the pipeline:
  ... python3 -m swiftcfd.machineLearning.channel_pipeline --stage check
"""

import argparse
import os
import sys

import numpy as np
import torch

from swiftcfd.machineLearning.hybrid_solver_ns import (
    load_ns_config,
    _is_all_neumann_p,
)
from swiftcfd.machineLearning.hybrid_solver import MLInference
from swiftcfd.machineLearning.hybrid_solver_ns_petsc_v2 import (
    run_sweep,
    print_fairness_check,
    make_percell_guess,
    make_unet_guess,
    UNetInference,
)
from swiftcfd.machineLearning.channel_pipeline import (
    default_paths,
    _warn_if_stale,
)


def _load_guess_fn(model_path, norm_path, cfg):
    """Same auto-detection as the v2 main: U-Net if the norm file carries
    'in_channels', per-cell MLP/RNN otherwise."""
    norm = torch.load(norm_path, weights_only=False)
    if "in_channels" in norm:
        unet = UNetInference(model_path, norm_path)
        return make_unet_guess(unet, cfg), unet.residual, "unet"
    ml = MLInference(model_path, norm_path)
    residual = bool(norm.get("residual_target", False))
    return make_percell_guess(ml, residual, cfg), residual, "per-cell"


def print_evidence_table(diag, target_reduction):
    """One row per (step, pc, init): the raw numbers the acceptance verdict
    rests on.  With --steps 1 this is exactly the plan's 'one step, three
    guesses, two PCs' matrix."""
    print("\n  EVIDENCE  (fair criterion: every row of one pc shares one atol)")
    print(f"  {'step':<6}{'pc':<8}{'init':<6}{'r0':>12}{'iters':>7}"
          f"{'rfinal':>12}{'atol':>12}{'rfinal/atol':>13}  reason")
    print("  " + "-" * 100)
    for d in diag:
        ratio = (d["rfinal"] / d["atol"]
                 if np.isfinite(d["rfinal"]) and d["atol"] > 0 else float("nan"))
        note = ""
        if d["iters"] == 0 and d["ok"]:
            note = ("  (genuine: r0 <= atol)" if d["r0"] <= d["atol"]
                    else "  <-- FAKE ZERO")
        elif not d["ok"]:
            note = "  <-- INVALID"
        print(f"  {d['step']:<6}{d['pc']:<8}{d['init']:<6}{d['r0']:>12.3e}"
              f"{d['iters']:>7}{d['rfinal']:>12.3e}{d['atol']:>12.3e}"
              f"{ratio:>13.3f}  {d['reason']}{note}")
    print("  " + "-" * 100)
    print(f"  atol = {target_reduction:.0e} * ||b||  per step "
          f"(||b|| = the zero-guess residual r0)")


def print_r0_reading(diag, init_names):
    """Phase-1 preview: does the ML guess actually START closer than prev?
    Iteration counts follow r0 once the finish line is shared."""
    if "ml" not in init_names:
        return
    print("\n  r0 READING (does the guess help before any iteration happens?)")
    for pc in sorted({d["pc"] for d in diag}):
        for step in sorted({d["step"] for d in diag if d["pc"] == pc}):
            row = {d["init"]: d for d in diag
                   if d["pc"] == pc and d["step"] == step}
            if not all(k in row for k in ("zero", "prev", "ml")):
                continue
            r0z, r0p, r0m = (row[k]["r0"] for k in ("zero", "prev", "ml"))
            if r0m < r0p:
                verdict = "ml starts BELOW prev -> model improves the start"
            elif r0m < r0z:
                verdict = "ml between prev and zero -> carries info, but prev wins"
            else:
                verdict = "ml >= zero -> guess carries NO information"
            print(f"    step {step}, {pc:<6}: r0 zero={r0z:.3e}  "
                  f"prev={r0p:.3e}  ml={r0m:.3e}   {verdict}")


def main():
    ap = argparse.ArgumentParser(
        description="Quick Phase-0 acceptance check: one measured step, three "
                    "guesses, two preconditioners, shared absolute finish line.")
    ap.add_argument("--config", default=os.path.join("input", "generated",
                                                     "val_01.toml"),
                    help="held-out channel TOML (default: input/generated/val_01.toml)")
    ap.add_argument("--model-type", default="mlp",
                    choices=["mlp", "rnn", "lstm", "transformer"],
                    help="which trained model to pick up from output/run_residual "
                         "(pinn_model_<type>.pth; default mlp)")
    ap.add_argument("--model", default=None,
                    help="model .pth (default: output/run_residual/"
                         "pinn_model_<model-type>.pth if it exists)")
    ap.add_argument("--norm", default=None,
                    help="norm .pth (default: output/run_residual/"
                         "norm_params_<model-type>.pth if it exists)")
    ap.add_argument("--no-model", action="store_true",
                    help="check zero/prev only (fairness holds regardless of ml)")
    ap.add_argument("--solver", default="BCGS",
                    choices=["CG", "BCGS", "GMRES", "RICHARDSON"])
    ap.add_argument("--pc-list", default="NONE,GAMG",
                    help="TWO different preconditioners (plan default: NONE,GAMG)")
    ap.add_argument("--spinup", type=int, default=10,
                    help="trajectory steps before the measured one (default 10)")
    ap.add_argument("--steps", type=int, default=1,
                    help="measured steps (default 1 = the plan's acceptance run)")
    ap.add_argument("--target-reduction", type=float, default=1e-12,
                    help="finish line atol = factor * ||b|| (plan default 1e-12)")
    ap.add_argument("--dt", type=float, default=None)
    args = ap.parse_args()

    cfg = load_ns_config(args.config)
    if args.dt is not None:
        cfg["dt"] = args.dt

    # explicit-time-integration stability clamp (same as v2 main)
    dt_stable = min(cfg["dx"] ** 2 / (4.0 * cfg["nu"]),
                    min(cfg["dx"], cfg["dy"])) * 0.4
    if cfg["dt"] > dt_stable:
        print(f"  AUTO-CLAMP: dt={cfg['dt']:.5f} -> {dt_stable:.5f} (explicit limit).")
        cfg["dt"] = dt_stable

    pc_list = [s.strip().upper() for s in args.pc_list.split(",") if s.strip()]
    if len(pc_list) < 2:
        ap.error("--pc-list needs at least TWO preconditioners; the acceptance "
                 "check is 'same finish line under two different PCs'.")

    guess_fn, residual, kind = None, False, "none"
    if not args.no_model:
        def_model, def_norm = default_paths(args.model_type)
        model = args.model or (def_model if os.path.exists(def_model) else None)
        norm = args.norm or (def_norm if os.path.exists(def_norm) else None)
        if model and norm:
            missing = [p for p in (model, norm) if not os.path.exists(p)]
            if missing:
                sys.exit(f"  ERROR: {missing[0]} does not exist -- train it first "
                         f"(--stage train with the matching --model-type), or pass "
                         f"--no-model to check fairness without an ML guess.")
            _warn_if_stale(model)
            guess_fn, residual, kind = _load_guess_fn(model, norm, cfg)
        else:
            print("  NOTE: no trained model found -> checking zero/prev only "
                  "(pass --model/--norm, or --no-model to silence this).")

    print(f"\n{'=' * 78}\n  QUICK WARM-START ACCEPTANCE CHECK (Phase 0)\n{'=' * 78}")
    print(f"  Config : {args.config}   grid {cfg['nc_x']}x{cfg['nc_y']}   "
          f"dt={cfg['dt']:.5f}")
    print(f"  Model  : {kind}   residual_target={residual}")
    print(f"  Krylov : {args.solver}   PCs: {pc_list}")
    print(f"  Steps  : spinup {args.spinup} -> measure {args.steps}")
    print(f"  Finish : atol = {args.target_reduction:.0e} * ||b||  "
          f"(unpreconditioned true residual, nonzero initial guess)")
    if _is_all_neumann_p(cfg["bc"]):
        print("  Nullsp : p is ALL-NEUMANN -> constant nullspace attached, "
              "b projected, guesses mean-projected")
    else:
        print("  Nullsp : p has a DIRICHLET reference (channel outlet p=0) -> "
              "non-singular, no nullspace handling needed")

    _, init_names, _, diag, _ = run_sweep(
        cfg, pc_list, args.solver, rtol=cfg["tol_p"],
        num_t=args.spinup + args.steps, guess_fn=guess_fn,
        ref_pc="GAMG", spinup=args.spinup,
        fair=True, target_reduction=args.target_reduction)

    print_evidence_table(diag, args.target_reduction)
    print_r0_reading(diag, init_names)
    ok = print_fairness_check(diag, pc_list, init_names)
    # the plan's acceptance is stricter than mere fairness: EVERY (guess, pc)
    # pair must actually REACH the shared finish line.  A cell that stagnates
    # (max_it, DIVERGED_*) never got there, so its iteration count is not a
    # comparable number -- reject, and say which combo to change.
    unconv = [d for d in diag if not d["ok"]]
    if ok and unconv:
        combos = sorted({(d["pc"], d["init"]) for d in unconv})
        print(f"\n  BUT: {len(unconv)} cell(s) never reached atol: "
              + ", ".join(f"{pc}/{ini}" for pc, ini in combos))
        print("  Their iteration counts are 'nan', not comparable numbers. Use a")
        print("  Krylov/pc combo that converges (e.g. BCGS), or loosen")
        print("  --target-reduction, then re-run the check.")
        ok = False
    if ok:
        print("\n  ACCEPTED: the measurement is fair -- iteration counts from the")
        print("  full sweep (--stage compare) are comparable across PCs and guesses.")
    else:
        print("\n  REJECTED: fix the criterion before trusting ANY sweep table.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
