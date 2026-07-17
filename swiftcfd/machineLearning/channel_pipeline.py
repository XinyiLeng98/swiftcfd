"""
Channel-flow ML-hybrid pipeline -- runs the WHOLE process in one file.

This is the SAME machine-learning workflow already used for the lid-driven
cavity, just pointed at the `channel` base case.  It only chains the three
existing, tested scripts together so you can run everything with one command:

    1. gen     -- generate_bc.py --case channel
                  Writes N training + M validation TOMLs with DIFFERENT boundary
                  conditions (randomised inlet velocity u_inlet and viscosity nu
                  -> a spread of Reynolds numbers) and runs the pure-PDE swiftCFD
                  solver on each training case to produce training data.
                     training cases -> output/training_XX/trainingData_{u,v,p}.csv
                     validation     -> input/generated/val_XX.toml (config only)

    2. train   -- train_residual_ns.py
                  Trains the residual pressure-correction model
                  (delta = p^{n+1} - p^n) on that data.  PETSc-free (torch only).
                     -> output/run_residual/pinn_model_mlp.pth + norm_params_mlp.pth

    3. check   -- quick_warmstart_check.py
                  FAST (seconds) Phase-0 acceptance gate: one measured step,
                  three initial guesses (zero/prev/ml), two preconditioners,
                  one shared absolute finish line atol = 1e-12*||b||.  Verifies
                  the measurement is FAIR (same rfinal for every pair, no fake
                  0-iteration wins) before any expensive sweep.  Exits nonzero
                  on failure, which aborts '--stage all' before 'compare'.

    4. compare -- hybrid_solver_ns_petsc_v2.py
                  Runs the ML-hybrid pressure solve on a held-out channel case and
                  compares it against the pure-PDE solver (previous-step warm
                  start, the 'prev' column) across a range of preconditioners on
                  one shared trajectory.
                     -> output/hybrid_ns_petsc_v2/sweep_table.txt + sweep.csv

Base case: input/channel.toml (inlet = west dirichlet u, outlet = east dirichlet
p, north/south no-slip walls).  Generated cases keep that BC STRUCTURE but vary
the values, so the model trains on "different boundary conditions".

Stages 1 and 3 need the PDE solver (PETSc) -> run in Docker.  Stage 2 is
PETSc-free -> venv or Docker.  Run one stage with --stage, or all of them with
--stage all (default).
"""

import argparse
import os
import subprocess
import sys

# defaults chosen to mirror the lid-driven workflow
VAL_CONFIG   = os.path.join("input", "generated", "val_01.toml")
RUN_DIR      = os.path.join("output", "run_residual")


def default_paths(model_type="mlp"):
    """train_residual_ns saves pinn_model_<type>.pth / norm_params_<type>.pth,
    so the check/compare defaults must follow --model-type (an rnn training run
    does NOT overwrite the mlp files -- loading the mlp default silently
    compares a stale model)."""
    return (os.path.join(RUN_DIR, f"pinn_model_{model_type}.pth"),
            os.path.join(RUN_DIR, f"norm_params_{model_type}.pth"))


MODEL_PATH, NORM_PATH = default_paths("mlp")   # back-compat for importers


def _run(cmd):
    print("\n$ " + " ".join(cmd) + "\n")
    ret = subprocess.run([sys.executable, "-m"] + cmd)
    if ret.returncode != 0:
        sys.exit(f"\n  stage failed (exit {ret.returncode}): {' '.join(cmd)}")


def stage_gen(a):
    print("\n" + "=" * 68 + "\n  STAGE 1: generate channel data (varying inlet velocity)\n" + "=" * 68)
    cmd = [
        "swiftcfd.machineLearning.generate_bc",
        "--case", "channel",
        "--n-train", str(a.n_train),
        "--n-val", str(a.n_val),
        "--seed", str(a.seed),
        "--nu-lo", str(a.nu_lo), "--nu-hi", str(a.nu_hi),
        "--u-inlet-lo", str(a.u_inlet_lo), "--u-inlet-hi", str(a.u_inlet_hi),
    ]
    # temporal richness: channel.toml ships timesteps=5 (only ~4 usable time
    # levels) -> far too little for the model to learn the transient. Default to
    # a long run so the flow develops and the pressure builds its streamwise ramp.
    # dt is auto-sized per case for CFL stability unless --gen-dt forces it.
    cmd += ["--timesteps", str(a.gen_timesteps), "--cfl", str(a.gen_cfl)]
    if a.gen_dt is not None: cmd += ["--dt", str(a.gen_dt)]
    # domain-size overrides (only pass when set, so base config is the default)
    if a.x_end       is not None: cmd += ["--x-end", str(a.x_end)]
    if a.y_end       is not None: cmd += ["--y-end", str(a.y_end)]
    if a.num_cells_x is not None: cmd += ["--num-cells-x", str(a.num_cells_x)]
    if a.num_cells_y is not None: cmd += ["--num-cells-y", str(a.num_cells_y)]
    _run(cmd)


def stage_train(a):
    print("\n" + "=" * 68 + "\n  STAGE 2: train residual pressure-correction model\n" + "=" * 68)
    _run([
        "swiftcfd.machineLearning.train_residual_ns",
        "--input-variables", "u,v,p", "--output-variable", "p",
        "--model", a.model_type,
        "--epochs", str(a.epochs), "--seed", str(a.seed),
        "--output-dir", RUN_DIR,
    ])


def _warn_if_stale(model_path):
    """Guard against the classic mistake: comparing a model that was trained
    BEFORE the current training data was (re)generated (e.g. an old lid-driven
    model against fresh channel data)."""
    import glob
    if not os.path.exists(model_path):
        return
    m_time = os.path.getmtime(model_path)
    data = glob.glob(os.path.join("output", "*", "trainingData_p.csv"))
    if data and max(os.path.getmtime(f) for f in data) > m_time:
        print("\n  " + "!" * 64)
        print(f"  WARNING: {model_path}")
        print(f"  is OLDER than the newest training data — it may be a STALE model")
        print(f"  trained on a different case/dataset. Re-run '--stage train' first.")
        print("  " + "!" * 64 + "\n")


def stage_check(a):
    print("\n" + "=" * 68 + "\n  STAGE 3: quick warm-start acceptance check (Phase 0)\n" + "=" * 68)
    model, norm = default_paths(a.model_type)
    _warn_if_stale(a.model or model)
    cmd = [
        "swiftcfd.machineLearning.quick_warmstart_check",
        "--config", a.config,
        "--model", a.model or model,
        "--norm", a.norm or norm,
        "--solver", a.solver,
        "--pc-list", a.check_pc_list,
        "--spinup", str(a.spinup),
        "--target-reduction", str(a.target_reduction
                                  if a.target_reduction is not None else 1e-12),
    ]
    _run(cmd)   # exits nonzero on FAIL -> aborts '--stage all' before compare


def stage_compare(a):
    print("\n" + "=" * 68 + "\n  STAGE 4: ML-hybrid vs pure-PDE sweep\n" + "=" * 68)
    model, norm = default_paths(a.model_type)
    _warn_if_stale(a.model or model)
    cmd = [
        "swiftcfd.machineLearning.hybrid_solver_ns_petsc_v2",
        "--config", a.config,
        "--model", a.model or model,
        "--norm", a.norm or norm,
        "--solver", a.solver,
        "--pc-list", a.pc_list,
        "--timesteps", str(a.timesteps),
        "--spinup", str(a.spinup),
    ]
    # fair per-step absolute finish line is the DEFAULT in v2; expose the
    # residual-decay figure and the legacy escape hatch here too.
    if a.legacy_rtol:              cmd += ["--legacy-rtol"]
    if a.target_reduction is not None:
        cmd += ["--target-reduction", str(a.target_reduction)]
    if a.curve_step is not None:   cmd += ["--curve-step", str(a.curve_step)]
    if a.curve_pc  is not None:    cmd += ["--curve-pc", a.curve_pc]
    _run(cmd)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["gen", "train", "check", "compare", "all"],
                    default="all", help="which stage to run (default: all)")

    # stage 1 (gen)
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--n-val",   type=int, default=3)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--nu-lo",      type=float, default=0.005)
    ap.add_argument("--nu-hi",      type=float, default=0.1)
    ap.add_argument("--u-inlet-lo", type=float, default=0.2)
    ap.add_argument("--u-inlet-hi", type=float, default=1.5)
    ap.add_argument("--gen-timesteps", type=int, default=500,
                    help="timesteps per training case (default 500; channel.toml's "
                         "5 is far too few to learn the transient)")
    ap.add_argument("--gen-cfl", type=float, default=0.3,
                    help="target CFL for per-case dt sizing during generation "
                         "(default 0.3; lower = more stable but slower)")
    ap.add_argument("--gen-dt", type=float, default=None,
                    help="force a fixed dt for data generation (default: auto CFL-safe)")
    # make the channel larger / finer (default: use input/channel.toml as-is)
    ap.add_argument("--x-end",       type=float, default=None, help="channel length")
    ap.add_argument("--y-end",       type=float, default=None, help="channel height")
    ap.add_argument("--num-cells-x", type=int,   default=None)
    ap.add_argument("--num-cells-y", type=int,   default=None)

    # stage 2 (train)
    ap.add_argument("--model-type", default="mlp",
                    choices=["mlp", "rnn", "lstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=200)

    # stage 3 (compare)
    ap.add_argument("--config", default=VAL_CONFIG,
                    help="held-out channel TOML (default: input/generated/val_01.toml)")
    ap.add_argument("--model", default=None, help="trained model .pth override")
    ap.add_argument("--norm",  default=None, help="norm params .pth override")
    ap.add_argument("--solver", default="BCGS",
                    choices=["CG", "BCGS", "GMRES", "RICHARDSON"])
    ap.add_argument("--pc-list", default="NONE,JACOBI,ILU,SOR,GAMG")
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--spinup",    type=int, default=10)
    ap.add_argument("--legacy-rtol", action="store_true",
                    help="use the old unfair relative-rtol convergence criterion")
    ap.add_argument("--target-reduction", type=float, default=None,
                    help="fair-mode finish line: atol = factor * ||b|| per step "
                         "(e.g. 1e-12; default: the KSP rtol)")
    ap.add_argument("--curve-step", type=int, default=None,
                    help="record the residual-decay curve at this step (Stage 3)")
    ap.add_argument("--curve-pc",   default=None,
                    help="preconditioner for the residual-decay curve")

    # stage 3 (check) -- the plan's acceptance run uses exactly TWO pcs
    ap.add_argument("--check-pc-list", default="NONE,GAMG",
                    help="two preconditioners for the quick acceptance check "
                         "(default NONE,GAMG)")

    a = ap.parse_args()

    if a.stage in ("gen", "all"):
        stage_gen(a)
    if a.stage in ("train", "all"):
        stage_train(a)
    if a.stage in ("check", "all"):
        stage_check(a)          # fails fast if the measurement is unfair
    if a.stage in ("compare", "all"):
        stage_compare(a)


if __name__ == "__main__":
    main()
