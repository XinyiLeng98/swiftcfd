"""
Phase-0 surrogate metric for the NS ML-hybrid pressure warm-start.

Instead of running a full simulation for every hyperparameter setting (slow),
this measures the thing the model is actually for: how many Gauss-Seidel
iterations the pressure-Poisson solve needs when it is *initialised* by the ML
guess, compared with the two cheap baselines the method must beat:

    - "zero"  : p_init = 0                      (cold start, worst case)
    - "prev"  : p_init = p^n                    (reuse previous pressure;
                                                 this is exactly what the hybrid
                                                 falls back to when ML is skipped)
    - "ml"    : p_init = ML(u,v,p history)      (the model under test)

A *useful* model must satisfy two conditions:
    1. CORRECTNESS  : the converged pressure is identical (within a few * tol)
                      regardless of the initial guess.  A warm start may only
                      change *speed*, never the answer.
    2. SPEED        : iters(ml) < iters(prev).  If not, the model is dead weight.

The script spins the case up for a few steps to reach a representative
(non-trivial, non-steady) flow state, then averages the iteration counts over
several consecutive steps so the number is not a single-step fluke.

Usage
-----
    python -m swiftcfd.machineLearning.evaluate_warmstart \
        --config input/lid.toml \
        --model  output/pinn_model_mlp.pth \
        --norm   output/norm_params_mlp.pth \
        --spinup 40 --eval-steps 10 --tol 1e-4

Use a --config whose Reynolds number is NOT in the training set to make this a
genuine held-out generalisation probe.
"""

import argparse

import numpy as np

from swiftcfd.machineLearning.hybrid_solver_ns import (
    load_ns_config,
    apply_all_bc,
    _momentum_step,
    _gs_pressure_step,
    _velocity_correction,
    _is_all_neumann_p,
    HybridSolverNS,
)
from swiftcfd.machineLearning.hybrid_solver import MLInference


def _clamp_dt(cfg):
    """Replicate the explicit-stability dt clamp used by the hybrid solver so the
    spin-up here matches the real run."""
    dt_diff   = cfg["dx"] ** 2 / (4.0 * cfg["nu"])
    dt_cfl    = min(cfg["dx"], cfg["dy"])
    dt_stable = min(dt_diff, dt_cfl) * 0.4
    if cfg["dt"] > dt_stable:
        print(f"  AUTO-CLAMP dt: {cfg['dt']:.5f} -> {dt_stable:.5f} "
              f"(diff={dt_diff:.5f}, cfl={dt_cfl:.5f})")
        cfg["dt"] = dt_stable
    return cfg


def _one_pde_step(state, cfg, pin):
    """Advance the reference (pure-PDE) solution by one fractional step and
    return the new state with the time-level history shifted."""
    (u, v, p, u_nm1, v_nm1, p_nm1, u_nm2, v_nm2, p_nm2) = state
    dx, dy = cfg["dx"], cfg["dy"]

    u_n, v_n, p_n = u.copy(), v.copy(), p.copy()
    u_star, v_star = _momentum_step(u, v, dx, dy, cfg["nu"], cfg["dt"], cfg["bc"])
    p_new, _, _ = _gs_pressure_step(
        p.copy(), u_star, v_star, dx, dy, cfg["rho"], cfg["dt"],
        cfg["bc"], cfg["max_iters_p"], cfg["tol_p"], pin)
    if pin:
        p_new[1:-1, 1:-1] -= p_new[1:-1, 1:-1].mean()
    u_new, v_new = _velocity_correction(
        u_star, v_star, p_new, dx, dy, cfg["rho"], cfg["dt"], cfg["bc"])

    return (u_new, v_new, p_new,
            u_n, v_n, p_n,        # n   -> n-1
            u_nm1, v_nm1, p_nm1)  # n-1 -> n-2


def _spinup(cfg, steps, pin):
    """Run the pure PDE solver for `steps` steps, returning the full
    (u, v, p) history needed to build the ML feature vector."""
    nx, ny = cfg["nx"], cfg["ny"]
    u = np.zeros((nx, ny)); v = np.zeros((nx, ny)); p = np.zeros((nx, ny))
    apply_all_bc(u, v, p, cfg["bc"], cfg["dx"], cfg["dy"])

    state = (u, v, p,
             u.copy(), v.copy(), p.copy(),
             u.copy(), v.copy(), p.copy())
    for _ in range(steps):
        state = _one_pde_step(state, cfg, pin)
    return state


def _gs_with_init(p_init, u_star, v_star, cfg, pin, tol, max_iters, true_residual=False):
    p_conv, n_iter, res = _gs_pressure_step(
        p_init.copy(), u_star, v_star, cfg["dx"], cfg["dy"],
        cfg["rho"], cfg["dt"], cfg["bc"], max_iters, tol, pin,
        true_residual=true_residual)
    if pin:
        p_conv[1:-1, 1:-1] -= p_conv[1:-1, 1:-1].mean()
    return p_conv, n_iter, res


def evaluate(cfg, ml, spinup, eval_steps, tol, max_iters, true_residual=False):
    pin = _is_all_neumann_p(cfg["bc"])
    # reuse the exact ML feature-row builder used in production inference
    hybrid = HybridSolverNS(cfg, ml)

    print(f"\n  Spinning up {spinup} steps to a representative flow state ...")
    state = _spinup(cfg, spinup, pin)

    rows = []          # per-step iteration counts and consistency errors
    for k in range(eval_steps):
        (u, v, p, u_nm1, v_nm1, p_nm1, u_nm2, v_nm2, p_nm2) = state
        dx, dy = cfg["dx"], cfg["dy"]

        u_star, v_star = _momentum_step(u, v, dx, dy, cfg["nu"], cfg["dt"], cfg["bc"])

        # ----- three initial guesses for the SAME pressure-Poisson problem -----
        zero_init = np.zeros_like(p)
        prev_init = p.copy()
        ml_init   = hybrid._ml_pressure_guess(
            u, v, p, u_nm1, v_nm1, p_nm1, u_nm2, v_nm2, p_nm2)

        p_zero, it_zero, _ = _gs_with_init(zero_init, u_star, v_star, cfg, pin, tol, max_iters, true_residual)
        p_prev, it_prev, _ = _gs_with_init(prev_init, u_star, v_star, cfg, pin, tol, max_iters, true_residual)
        p_ml,   it_ml,   _ = _gs_with_init(ml_init,   u_star, v_star, cfg, pin, tol, max_iters, true_residual)

        # CORRECTNESS: converged pressures must agree (gauge-fixed for all-Neumann)
        ref = p_prev[1:-1, 1:-1]
        d_ml   = float(np.max(np.abs(p_ml[1:-1, 1:-1]   - ref)))
        d_zero = float(np.max(np.abs(p_zero[1:-1, 1:-1] - ref)))

        rows.append((it_zero, it_prev, it_ml, d_ml, d_zero))

        # advance the reference solution with the converged (true) pressure
        u_new, v_new = _velocity_correction(
            u_star, v_star, p_prev, dx, dy, cfg["rho"], cfg["dt"], cfg["bc"])
        state = (u_new, v_new, p_prev,
                 u, v, p,
                 u_nm1, v_nm1, p_nm1)

    arr = np.array(rows, dtype=float)
    it_zero, it_prev, it_ml = arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 2].mean()
    d_ml_max, d_zero_max     = arr[:, 3].max(), arr[:, 4].max()

    red_vs_prev = (1.0 - it_ml / it_prev) * 100 if it_prev else 0.0
    red_vs_zero = (1.0 - it_ml / it_zero) * 100 if it_zero else 0.0

    conv_mode = "true Poisson residual" if true_residual else "sweep-to-sweep change"
    print(f"\n{'='*64}")
    print(f"  WARM-START SURROGATE METRIC   (tol={tol:g}, {eval_steps} steps)")
    print(f"  convergence criterion: {conv_mode}")
    print(f"{'='*64}")
    print(f"  Avg GS iterations to converge:")
    print(f"    zero init : {it_zero:8.2f}")
    print(f"    prev init : {it_prev:8.2f}   <-- baseline (hybrid fallback)")
    print(f"    ML   init : {it_ml:8.2f}")
    print(f"  Iteration reduction:")
    print(f"    ML vs prev: {red_vs_prev:+6.1f}%   (must be > 0 to be useful)")
    print(f"    ML vs zero: {red_vs_zero:+6.1f}%")
    print(f"  Converged-pressure consistency (must be ~tol):")
    print(f"    max|p_ml   - p_prev| : {d_ml_max:.2e}")
    print(f"    max|p_zero - p_prev| : {d_zero_max:.2e}")
    verdict = "USEFUL" if (red_vs_prev > 0 and d_ml_max < 100 * tol) else "NOT USEFUL"
    print(f"  Verdict: {verdict}")
    print(f"{'='*64}\n")

    return {
        "iters_zero": it_zero, "iters_prev": it_prev, "iters_ml": it_ml,
        "reduction_vs_prev_pct": red_vs_prev, "reduction_vs_zero_pct": red_vs_zero,
        "consistency_ml": d_ml_max, "consistency_zero": d_zero_max,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Cheap surrogate metric for the NS pressure warm-start model.")
    ap.add_argument("--config", required=True, help="NS TOML config (use a held-out Re).")
    ap.add_argument("--model",  required=True, help="Trained model .pth")
    ap.add_argument("--norm",   required=True, help="Normalization .pth")
    ap.add_argument("--spinup",     type=int,   default=40,
                    help="PDE steps to reach a representative flow state (default: 40)")
    ap.add_argument("--eval-steps", type=int,   default=10,
                    help="Consecutive steps to average the metric over (default: 10)")
    ap.add_argument("--tol",        type=float, default=1e-4,
                    help="GS convergence tolerance for the probe (default: 1e-4)")
    ap.add_argument("--max-iters",  type=int,   default=5000,
                    help="Safety cap on GS iterations (default: 5000)")
    ap.add_argument("--true-residual", action="store_true",
                    help="Stop GS on the actual Poisson residual instead of the "
                         "sweep-to-sweep change, so every init converges to the "
                         "same solution and iteration counts are comparable (fair).")
    ap.add_argument("--dt",         type=float, default=None, help="Override dt.")
    args = ap.parse_args()

    cfg = load_ns_config(args.config)
    if args.dt is not None:
        cfg["dt"] = args.dt
    cfg = _clamp_dt(cfg)

    print(f"  Config : {args.config}")
    print(f"  Grid   : {cfg['nc_x']} x {cfg['nc_y']}   nu={cfg['nu']}  rho={cfg['rho']}")
    print(f"  dt     : {cfg['dt']:.5f}")

    ml = MLInference(args.model, args.norm)
    evaluate(cfg, ml, args.spinup, args.eval_steps, args.tol, args.max_iters,
             true_residual=args.true_residual)


if __name__ == "__main__":
    main()
