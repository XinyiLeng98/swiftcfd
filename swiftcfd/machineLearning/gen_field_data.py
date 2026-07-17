"""
Whole-field training-data generator for the U-Net (G3).

The per-cell CSV pipeline (DataManager) cannot feed a U-Net -- it stores local
stencils, not fields.  This script runs the SAME fractional-step NS solver and
dumps, per time step, the WHOLE interior fields needed for supervised learning:

  inputs  (channels):  u^n, v^n, p^n, rhs   (rhs = (rho/dt) div u*  = Poisson source)
  target            :  p^{n+1}  (converged, via PETSc)         -> also store delta

Saved as one .npz per config in --output-dir, each holding arrays of shape
(T, H, W).  `train_unet.py` globs them.

Run (Docker), one or many cases:
  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.gen_field_data \
      --configs input/generated/training_01.toml input/generated/training_02.toml \
      --timesteps 200 --output-dir output/field_data
"""

import argparse
import glob
import os

import numpy as np

from swiftcfd.machineLearning.hybrid_solver_ns import (
    load_ns_config, apply_all_bc, _is_all_neumann_p,
    _momentum_step, _velocity_correction,
)
from swiftcfd.machineLearning.hybrid_solver_ns_petsc import PetscPressurePoisson


def generate_one(cfg, num_t, solver="BCGS", pc="GAMG"):
    nx, ny, dx, dy = cfg["nx"], cfg["ny"], cfg["dx"], cfg["dy"]
    nu, dt, rho = cfg["nu"], cfg["dt"], cfg["rho"]
    bc = cfg["bc"]

    poisson = PetscPressurePoisson(cfg, solver=solver, preconditioner=pc,
                                   rtol=cfg["tol_p"])

    u = np.zeros((nx, ny)); v = np.zeros((nx, ny)); p = np.zeros((nx, ny))
    apply_all_bc(u, v, p, bc, dx, dy)

    U, V, P, RHS, PNEXT = [], [], [], [], []
    interior = (slice(1, -1), slice(1, -1))

    for step in range(num_t):
        p_n = p.copy()
        u_star, v_star = _momentum_step(u, v, dx, dy, nu, dt, bc)
        if np.any(np.isnan(u_star)):
            raise RuntimeError(f"NaN at step {step}: dt={dt:.5f} unstable.")

        # Poisson source on the interior  = (rho/dt) div(u*)
        rhs = (rho / dt) * (
            (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2.0 * dx) +
            (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2.0 * dy))

        p, _, _ = poisson.solve(p_n, u_star, v_star)   # converged target
        u, v = _velocity_correction(u_star, v_star, p, dx, dy, rho, dt, bc)

        # record interior fields for this step
        U.append(u[interior].copy())
        V.append(v[interior].copy())
        P.append(p_n[interior].copy())      # p BEFORE this step's solve (the "n" input)
        RHS.append(rhs.copy())
        PNEXT.append(p[interior].copy())    # converged p^{n+1} (target)

        if num_t <= 10 or step % max(1, num_t // 10) == 0:
            print(f"    step {step:4d}/{num_t}")

    stack = lambda lst: np.asarray(lst, dtype=np.float32)
    return {
        "u": stack(U), "v": stack(V), "p": stack(P),
        "rhs": stack(RHS), "p_next": stack(PNEXT),
        "delta": stack(PNEXT) - stack(P),
        "dx": np.float32(dx), "dy": np.float32(dy),
        "nu": np.float32(nu), "rho": np.float32(rho), "dt": np.float32(dt),
    }


def main():
    ap = argparse.ArgumentParser(description="Generate whole-field U-Net training data.")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="TOML configs (globs allowed)")
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--solver", default="BCGS")
    ap.add_argument("--pc", default="GAMG")
    ap.add_argument("--output-dir", default=os.path.join("output", "field_data"))
    args = ap.parse_args()

    paths = []
    for c in args.configs:
        paths.extend(sorted(glob.glob(c)) or [c])
    os.makedirs(args.output_dir, exist_ok=True)

    for path in paths:
        cfg = load_ns_config(path)
        if args.dt is not None:
            cfg["dt"] = args.dt
        cfg["timesteps"] = args.timesteps
        dt_stable = min(cfg["dx"] ** 2 / (4.0 * cfg["nu"]),
                        min(cfg["dx"], cfg["dy"])) * 0.4
        if cfg["dt"] > dt_stable:
            cfg["dt"] = dt_stable

        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n[gen] {name}  grid={cfg['nc_x']}x{cfg['nc_y']}  "
              f"dt={cfg['dt']:.5f}  steps={cfg['timesteps']}")
        data = generate_one(cfg, cfg["timesteps"], args.solver, args.pc)
        out = os.path.join(args.output_dir, f"fields_{name}.npz")
        np.savez_compressed(out, **data)
        print(f"  saved {out}  ({data['p'].shape[0]} snapshots of "
              f"{data['p'].shape[1]}x{data['p'].shape[2]})")


if __name__ == "__main__":
    main()
