"""
G2: residual warm-start + COST-AWARE switching, judged in WALL-CLOCK time.

Iteration count (what the v2 sweep reports) is only half the story.  An ML guess
that saves iterations can still LOSE in wall-clock if the network inference costs
more than the solver time it saves.  G2 answers the honest end-to-end question:

    Including the cost of running the network, does an ML warm start actually
    make the simulation finish faster than the standard previous-step warm
    start -- and can a cheap switching rule keep us from ever losing?

Three strategies are run on the SAME case (each advances its own trajectory but
to the SAME tolerance, so the physics matches):

  prev      : always warm-start from p^n                    (the baseline)
  ml-always : always use the ML guess                       (pays inference every step)
  ml-switch : use ML only when the flow is changing fast    (cost-aware policy)
              rms(p^n, p^{n-1}) > switch_thresh ;  else fall back to prev
              -> spend inference only where a warm start has room to help.

Reported per strategy: wall-clock (total / solve / inference), avg KSP iters,
ML-usage %, and net speedup vs the prev baseline.

Run (Docker):
  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.g2_costaware \
      --config input/generated/val_01.toml \
      --model output/run_residual/pinn_model_mlp.pth \
      --norm  output/run_residual/norm_params_mlp.pth \
      --solver BCGS --pc JACOBI --timesteps 200
"""

import argparse
import os
import time

import numpy as np
import torch

from swiftcfd.machineLearning.hybrid_solver_ns import (
    load_ns_config, apply_all_bc, _is_all_neumann_p, rms,
    _momentum_step, _velocity_correction,
)
from swiftcfd.machineLearning.hybrid_solver_ns_petsc import PetscPressurePoisson
from swiftcfd.machineLearning.hybrid_solver_ns_petsc_v2 import (
    make_percell_guess, make_unet_guess, UNetInference,
)
from swiftcfd.machineLearning.hybrid_solver import MLInference


def run_strategy(cfg, poisson, strategy, guess_fn=None, switch_thresh=None):
    """strategy in {'prev', 'ml-always', 'ml-switch'}.  Returns a results dict.
    `guess_fn` is the model closure (per-cell or U-Net); None -> prev only."""
    nx, ny, dx, dy = cfg["nx"], cfg["ny"], cfg["dx"], cfg["dy"]
    nu, dt, rho = cfg["nu"], cfg["dt"], cfg["rho"]
    bc = cfg["bc"]
    num_t = cfg["timesteps"]

    u = np.zeros((nx, ny)); v = np.zeros((nx, ny)); p = np.zeros((nx, ny))
    apply_all_bc(u, v, p, bc, dx, dy)
    hist = {k: f.copy() for k, f in
            (("u_nm1", u), ("v_nm1", v), ("p_nm1", p),
             ("u_nm2", u), ("v_nm2", v), ("p_nm2", p))}

    total_iters = 0
    infer_time = 0.0
    ml_used = 0
    t0 = time.time()

    for step in range(num_t):
        u_n, v_n, p_n = u.copy(), v.copy(), p.copy()
        u_star, v_star = _momentum_step(u, v, dx, dy, nu, dt, bc)
        if np.any(np.isnan(u_star)) or np.any(np.isnan(v_star)):
            raise RuntimeError(f"NaN at step {step}: dt={dt:.5f} unstable.")

        # ---- choose the initial guess ----
        use_ml = False
        if strategy != "prev" and guess_fn is not None:
            if strategy == "ml-always":
                use_ml = True
            elif strategy == "ml-switch":
                use_ml = rms(p_n, hist["p_nm1"]) > switch_thresh

        p_init = p_n
        if use_ml:
            ti = time.time()
            g = guess_fn(step, u_n, v_n, p_n, u_star, v_star, hist)
            infer_time += time.time() - ti
            if g is not None:                 # per-cell returns None during boot
                p_init = g
                ml_used += 1

        p, n_iter, _ = poisson.solve(p_init, u_star, v_star)
        u, v = _velocity_correction(u_star, v_star, p, dx, dy, rho, dt, bc)
        total_iters += n_iter

        hist["u_nm2"], hist["v_nm2"], hist["p_nm2"] = hist["u_nm1"], hist["v_nm1"], hist["p_nm1"]
        hist["u_nm1"], hist["v_nm1"], hist["p_nm1"] = u_n, v_n, p_n

    wall = time.time() - t0
    return {
        "strategy": strategy,
        "wall": wall,
        "infer": infer_time,
        "solve": wall - infer_time,
        "avg_iters": total_iters / max(num_t, 1),
        "ml_used_pct": 100.0 * ml_used / max(num_t, 1),
        "u": u, "v": v, "p": p,
    }


def main():
    ap = argparse.ArgumentParser(description="G2: cost-aware ML warm-start in wall-clock.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--solver", default="BCGS",
                    choices=["CG", "BCGS", "GMRES", "RICHARDSON"])
    ap.add_argument("--pc", default="JACOBI",
                    choices=["NONE", "JACOBI", "ILU", "SOR", "GAMG"])
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--switch-thresh", type=float, default=None,
                    help="rms(p^n,p^{n-1}) threshold for ml-switch "
                         "(default: 10x KSP rtol)")
    ap.add_argument("--output-dir", default=os.path.join("output", "g2_costaware"))
    args = ap.parse_args()

    cfg = load_ns_config(args.config)
    if args.tol is not None:       cfg["tol_p"] = args.tol
    if args.dt is not None:        cfg["dt"] = args.dt
    if args.timesteps is not None: cfg["timesteps"] = args.timesteps

    dt_stable = min(cfg["dx"] ** 2 / (4.0 * cfg["nu"]),
                    min(cfg["dx"], cfg["dy"])) * 0.4
    if cfg["dt"] > dt_stable:
        print(f"\n  AUTO-CLAMP: dt -> {dt_stable:.5f} (explicit limit).")
        cfg["dt"] = dt_stable

    rtol = cfg["tol_p"]
    switch_thresh = args.switch_thresh if args.switch_thresh is not None else rtol * 10

    norm = torch.load(args.norm, weights_only=False)
    if "in_channels" in norm:               # auto-detect whole-field U-Net
        print("\n--- Loading U-Net (whole-field) ---")
        unet = UNetInference(args.model, args.norm)
        residual = unet.residual
        guess_fn = make_unet_guess(unet, cfg)
    else:
        print("\n--- Loading per-cell ML model ---")
        ml = MLInference(args.model, args.norm)
        residual = bool(norm.get("residual_target", False))
        guess_fn = make_percell_guess(ml, residual, cfg)

    print(f"\n{'='*65}\n  G2 cost-aware warm-start  ({args.solver}+{args.pc})\n{'='*65}")
    print(f"  Config={args.config}  grid={cfg['nc_x']}x{cfg['nc_y']}  "
          f"steps={cfg['timesteps']}")
    print(f"  residual_ML={residual}  switch_thresh={switch_thresh:.2e}")

    # one matrix/preconditioner shared by all strategies (identical systems)
    poisson = PetscPressurePoisson(cfg, solver=args.solver,
                                   preconditioner=args.pc, rtol=rtol)

    results = []
    for strat in ("prev", "ml-always", "ml-switch"):
        print(f"\n  running: {strat} ...")
        results.append(run_strategy(cfg, poisson, strat, guess_fn=guess_fn,
                                     switch_thresh=switch_thresh))

    base = results[0]["wall"]
    lines = [
        "\n" + "=" * 84,
        f"  G2 WALL-CLOCK  ({args.solver}+{args.pc}, {cfg['timesteps']} steps, "
        f"grid {cfg['nc_x']}x{cfg['nc_y']})",
        "=" * 84,
        f"  {'strategy':<12}{'wall(s)':>9}{'solve(s)':>9}{'infer(s)':>9}"
        f"{'avg_iters':>11}{'ML use%':>9}{'speedup':>9}",
        "  " + "-" * 82,
    ]
    for r in results:
        spd = base / max(r["wall"], 1e-9)
        lines.append(
            f"  {r['strategy']:<12}{r['wall']:>9.2f}{r['solve']:>9.2f}"
            f"{r['infer']:>9.2f}{r['avg_iters']:>11.1f}{r['ml_used_pct']:>8.0f}%"
            f"{spd:>8.2f}x")
    lines.append("=" * 84)
    lines.append("  Honest read: 'speedup' > 1 means ML wins in WALL-CLOCK after")
    lines.append("  paying inference. ml-switch should dominate ml-always when the")
    lines.append("  flow has steady stretches (it skips useless inference there).\n")
    text = "\n".join(lines)
    print(text)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "g2_table.txt"), "w") as fh:
        fh.write(text)
    print(f"  Saved: {args.output_dir}/g2_table.txt")


if __name__ == "__main__":
    main()
