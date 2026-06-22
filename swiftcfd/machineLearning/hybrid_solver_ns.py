import argparse
import os
import time

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# ══════════════════════════════════════════════════════════════════════════════
# Config reader
# ══════════════════════════════════════════════════════════════════════════════

def load_ns_config(toml_path: str) -> dict:

    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    nu        = float(cfg["solver"]["fluid"]["nu"])
    rho       = float(cfg["solver"]["fluid"]["rho"])
    dt        = float(cfg["solver"]["time"]["dt"])
    timesteps = int(cfg["solver"]["time"]["timesteps"])

    max_iters_p = int(cfg["solver"]["linearSolver"]["maxIterations"]["p"])
    tol_p       = float(cfg["solver"]["linearSolver"]["tolerance"]["p"])

    mesh = cfg["mesh"]["block1"]
    nc_x = int(mesh["x"]["numCells"])
    nc_y = int(mesh["y"]["numCells"])
    L_x  = float(mesh["x"]["end"]) - float(mesh["x"]["start"])
    L_y  = float(mesh["y"]["end"]) - float(mesh["y"]["start"])
    dx   = L_x / nc_x
    dy   = L_y / nc_y
    # total nodes including ghost cells
    nx   = nc_x + 2
    ny   = nc_y + 2

    bc_raw = cfg["boundaryCondition"]["block1"]
    bc: dict = {}
    for face in ("east", "west", "north", "south"):
        bc[face] = {}
        for var in ("u", "v", "p"):
            bc[face][var] = (
                bc_raw[face][var]["type"],
                float(bc_raw[face][var]["value"]),
            )

    return {
        "nu": nu, "rho": rho, "dt": dt, "timesteps": timesteps,
        "max_iters_p": max_iters_p, "tol_p": tol_p,
        "nc_x": nc_x, "nc_y": nc_y, "nx": nx, "ny": ny,
        "dx": dx, "dy": dy, "L_x": L_x, "L_y": L_y,
        "bc": bc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Boundary conditions
# ══════════════════════════════════════════════════════════════════════════════

def _apply_one_bc(field: np.ndarray, bc: dict, dx: float, dy: float, var: str):

    # (face, step size, ghost slice, interior-neighbor slice)
    faces = [
        ("west",  dx, (0,            slice(None)), (1,             slice(None))),
        ("east",  dx, (-1,           slice(None)), (-2,            slice(None))),
        ("south", dy, (slice(None),  0           ), (slice(None),  1           )),
        ("north", dy, (slice(None),  -1          ), (slice(None),  -2          )),
    ]
    for face, h, ghost_sl, nbr_sl in faces:
        bc_type, val = bc[face][var]
        if bc_type == "dirichlet":
            field[ghost_sl] = 2.0 * val - field[nbr_sl]     # dirichlet: φ_ghost = 2*val - φ_nbr  (mirror across boundary)
        else:                       
            field[ghost_sl] = field[nbr_sl] + val * h       # neumann: ∂φ/∂n = val  (outward normal)


def apply_all_bc(u: np.ndarray, v: np.ndarray, p: np.ndarray,
                 bc: dict, dx: float, dy: float):
    _apply_one_bc(u, bc, dx, dy, "u")
    _apply_one_bc(v, bc, dx, dy, "v")
    _apply_one_bc(p, bc, dx, dy, "p")


def _is_all_neumann_p(bc: dict) -> bool:
    return all(bc[f]["p"][0] == "neumann"
               for f in ("east", "west", "north", "south"))


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# ML inference  (reuse the heat-solver wrapper)
# ══════════════════════════════════════════════════════════════════════════════

from swiftcfd.machineLearning.hybrid_solver import MLInference


# ══════════════════════════════════════════════════════════════════════════════
# Fractional-step sub-steps
# ══════════════════════════════════════════════════════════════════════════════

def _momentum_step(u: np.ndarray, v: np.ndarray,
                   dx: float, dy: float, nu: float, dt: float,
                   bc: dict) -> tuple:
    # interior values only
    u_c = u[1:-1, 1:-1]
    v_c = v[1:-1, 1:-1]

    # First-order upwind (donor-cell) convection: use the upstream neighbour
    # based on the sign of the advecting velocity.  Suppresses wiggles when
    # cell-Péclet > 2 (high Re); diffusion stays central.
    u_pos, u_neg = np.maximum(u_c, 0.0), np.minimum(u_c, 0.0)
    v_pos, v_neg = np.maximum(v_c, 0.0), np.minimum(v_c, 0.0)

    du_dx_back = (u_c              - u[:-2, 1:-1]) / dx
    du_dx_fwd  = (u[2:,   1:-1]    - u_c          ) / dx
    du_dy_back = (u_c              - u[1:-1, :-2]) / dy
    du_dy_fwd  = (u[1:-1, 2:]      - u_c          ) / dy

    dv_dx_back = (v_c              - v[:-2, 1:-1]) / dx
    dv_dx_fwd  = (v[2:,   1:-1]    - v_c          ) / dx
    dv_dy_back = (v_c              - v[1:-1, :-2]) / dy
    dv_dy_fwd  = (v[1:-1, 2:]      - v_c          ) / dy

    conv_u = (u_pos * du_dx_back + u_neg * du_dx_fwd
              + v_pos * du_dy_back + v_neg * du_dy_fwd)
    conv_v = (u_pos * dv_dx_back + u_neg * dv_dx_fwd
              + v_pos * dv_dy_back + v_neg * dv_dy_fwd)

    # viscous diffusion nu (∂²u/∂x² + ∂²u/∂y²)  and  nu (∂²v/∂x² + ∂²v/∂y²)
    diff_u = nu * ((u[2:, 1:-1] - 2.0*u_c + u[:-2, 1:-1]) / dx**2
                   + (u[1:-1, 2:] - 2.0*u_c + u[1:-1, :-2]) / dy**2)
    diff_v = nu * ((v[2:, 1:-1] - 2.0*v_c + v[:-2, 1:-1]) / dx**2
                   + (v[1:-1, 2:] - 2.0*v_c + v[1:-1, :-2]) / dy**2)
    # intermediate velocity
    u_star = u.copy()
    v_star = v.copy()
    u_star[1:-1, 1:-1] = u_c + dt * (-conv_u + diff_u)
    v_star[1:-1, 1:-1] = v_c + dt * (-conv_v + diff_v)
    # enforce BCs on intermediate velocity
    _apply_one_bc(u_star, bc, dx, dy, "u")
    _apply_one_bc(v_star, bc, dx, dy, "v")
    return u_star, v_star


# ══════════════════════════════════════════════════════════════════════════════
# GS solver for pressure Poisson
# ══════════════════════════════════════════════════════════════════════════════

def _gs_pressure_step(p_init: np.ndarray,
                      u_star: np.ndarray, v_star: np.ndarray,
                      dx: float, dy: float, rho: float, dt: float,
                      bc: dict, max_iters: int, tol: float,
                      pin_p: bool = False, use_ser: bool = False,
                      true_residual: bool = False) -> tuple:
    # true_residual=False : stop when the sweep-to-sweep CHANGE is small (cheap,
    #   but can stop early/falsely when the initial guess is already near a fixed
    #   point).  true_residual=True : stop when the actual Poisson residual
    #   ‖∇²p - rhs‖ / ‖rhs‖ is small, so every initial guess converges to the SAME
    #   solution and iteration counts are comparable across warm starts.

    p  = p_init.copy()
    ap = 2.0 / dx**2 + 2.0 / dy**2
    nx, ny = p.shape

    # divergence of u* at interior cells
    rhs = (rho / dt) * (
        (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2.0 * dx) +
        (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2.0 * dy)
    )

    omega    = 1.0
    res_prev = None
    res_norm = 0.0

    for k in range(max_iters):
        p_old = p.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                ae   = p_old[i+1, j] / dx**2   # east  (not yet updated)
                aw   = p[i-1, j]     / dx**2    # west  (already updated)
                an   = p_old[i, j+1] / dy**2   # north (not yet updated)
                as_  = p[i, j-1]     / dy**2    # south (already updated)
                p_gs = (ae + aw + an + as_ - rhs[i-1, j-1]) / ap

                if use_ser:
                    p[i, j] = (1.0 - omega) * p_old[i, j] + omega * p_gs
                else:
                    p[i, j] = p_gs

        _apply_one_bc(p, bc, dx, dy, "p")

        if pin_p:
            p -= p[1:-1, 1:-1].mean()

        if true_residual:
            # actual Poisson residual: ∇²p - rhs  at interior cells
            pc  = p[1:-1, 1:-1]
            lap = ((p[2:, 1:-1] - 2.0*pc + p[:-2, 1:-1]) / dx**2
                   + (p[1:-1, 2:] - 2.0*pc + p[1:-1, :-2]) / dy**2)
            res      = float(np.sqrt(np.mean((lap - rhs)**2)))
            denom    = float(np.sqrt(np.mean(rhs**2))) + 1e-30
            res_norm = res / denom
        else:
            diff     = p[1:-1, 1:-1] - p_old[1:-1, 1:-1]
            res      = float(np.sqrt(np.mean(diff**2)))
            denom    = float(np.sqrt(np.mean(p[1:-1, 1:-1]**2))) + 1e-30
            res_norm = res / denom

        if use_ser and res_prev is not None and res > 1e-30:
            omega = omega * (res_prev / res) ** 0.3
            omega = max(0.5, min(omega, 1.8))
        res_prev = res

        if res_norm < tol:
            return p, k + 1, res_norm

    return p, max_iters, res_norm


def _velocity_correction(u_star: np.ndarray, v_star: np.ndarray,
                          p: np.ndarray,
                          dx: float, dy: float, rho: float, dt: float,
                          bc: dict) -> tuple:

    u = u_star.copy()
    v = v_star.copy()
    u[1:-1, 1:-1] -= (dt / rho) * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2.0 * dx)
    v[1:-1, 1:-1] -= (dt / rho) * (p[1:-1, 2:] - p[1:-1, :-2]) / (2.0 * dy)
    _apply_one_bc(u, bc, dx, dy, "u")
    _apply_one_bc(v, bc, dx, dy, "v")
    return u, v


# ══════════════════════════════════════════════════════════════════════════════
# Shared solver base
# ══════════════════════════════════════════════════════════════════════════════

class _BaseSolverNS:
    def __init__(self, cfg: dict, use_ser: bool = False, label: str = ""):
        self.cfg       = cfg
        self.bc        = cfg["bc"]
        self.use_ser   = use_ser
        self.label     = label
        self.nu        = cfg["nu"]
        self.rho       = cfg["rho"]
        self.dt        = cfg["dt"]
        self.num_t     = cfg["timesteps"]
        self.nx        = cfg["nx"]
        self.ny        = cfg["ny"]
        self.dx        = cfg["dx"]
        self.dy        = cfg["dy"]
        self.max_iters = cfg["max_iters_p"]
        self.tol       = cfg["tol_p"]
        self.pin_p     = _is_all_neumann_p(self.bc)


# ══════════════════════════════════════════════════════════════════════════════
# Pure PDE solver
# ══════════════════════════════════════════════════════════════════════════════

class PDESolverNS(_BaseSolverNS):

    def __init__(self, cfg: dict, use_ser: bool = False, label: str = "PDE(GS)"):
        super().__init__(cfg, use_ser, label)

    def run(self):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        u = np.zeros((nx, ny))
        v = np.zeros((nx, ny))
        p = np.zeros((nx, ny))
        apply_all_bc(u, v, p, self.bc, dx, dy)

        total_gs  = 0
        residuals = []
        t0        = time.time()

        for step in range(self.num_t):
            u_star, v_star = _momentum_step(
                u, v, dx, dy, self.nu, self.dt, self.bc)

            if np.any(np.isnan(u_star)) or np.any(np.isnan(v_star)):
                raise RuntimeError(
                    f"NaN at step {step}: dt={self.dt:.5f} is unstable. "
                    f"Try --dt {self.dx**2 / (8.0 * self.nu):.5f}"
                )

            p, n_iter, res_norm = _gs_pressure_step(
                p, u_star, v_star,
                dx, dy, self.rho, self.dt,
                self.bc, self.max_iters, self.tol, self.pin_p, self.use_ser)

            if self.pin_p:
                p[1:-1, 1:-1] -= p[1:-1, 1:-1].mean()

            u, v = _velocity_correction(
                u_star, v_star, p, dx, dy, self.rho, self.dt, self.bc)

            total_gs  += n_iter
            residuals.append(res_norm)

            if self.num_t <= 10 or step % max(1, self.num_t // 10) == 0:
                avg = total_gs / (step + 1)
                print(f"  [{self.label}] step={step:4d}  GS={n_iter:4d}  "
                      f"avg={avg:.1f}  res={res_norm:.2e}")

        self.exec_time    = time.time() - t0
        self.avg_gs_iters = total_gs / max(self.num_t, 1)
        self.residuals    = residuals
        self.u_final      = u
        self.v_final      = v
        self.p_final      = p

        ser_tag = " + SER" if self.use_ser else ""
        self.report = (
            f"\n>>> {self.label} (GS{ser_tag})\n"
            f">>> Execution time      : {self.exec_time:.2f} s\n"
            f">>> Avg GS iterations   : {self.avg_gs_iters:.1f}\n"
            f">>> Final residual      : {residuals[-1]:.2e}\n"
            f">>> Time steps          : {self.num_t}\n"
        )
        print(self.report)
        return u, v, p, residuals


# ══════════════════════════════════════════════════════════════════════════════
# ML-Hybrid solver
# ══════════════════════════════════════════════════════════════════════════════

class HybridSolverNS(_BaseSolverNS):

    def __init__(self, cfg: dict, ml: MLInference, use_ser: bool = False,
                 label: str = "ML-Hybrid"):
        super().__init__(cfg, use_ser, label)
        self.ml = ml

    def _ml_pressure_guess(self,
                           u_n, v_n, p_n,
                           u_nm1, v_nm1, p_nm1,
                           u_nm2, v_nm2, p_nm2) -> np.ndarray:

        nx, ny = self.nx, self.ny
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

        batch = np.array(rows, dtype=np.float32)
        preds = self.ml.predict(batch)          # (N, 5): p stencil at n+1

        p_guess = p_n.copy()
        for idx, (i, j) in enumerate(coords):
            p_guess[i, j] = preds[idx, 0]      # centre value only

        _apply_one_bc(p_guess, self.bc, self.dx, self.dy, "p")
        if self.pin_p:
            p_guess -= p_guess[1:-1, 1:-1].mean()
        return p_guess

    def run(self):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        u = np.zeros((nx, ny))
        v = np.zeros((nx, ny))
        p = np.zeros((nx, ny))
        apply_all_bc(u, v, p, self.bc, dx, dy)

        # keep two previous time levels for ML feature vector
        u_nm2 = u.copy(); v_nm2 = v.copy(); p_nm2 = p.copy()
        u_nm1 = u.copy(); v_nm1 = v.copy(); p_nm1 = p.copy()

        total_gs    = 0
        residuals   = []
        t0          = time.time()
        skip_thresh = self.tol * 10     # skip ML when essentially steady

        for step in range(self.num_t):
            u_n = u.copy(); v_n = v.copy(); p_n = p.copy()

            if step < 2:
                p_init = p_n
                ml_tag = "boot"
            else:
                change = rms(p_n, p_nm1)
                if change > skip_thresh:
                    p_init = self._ml_pressure_guess(
                        u_n, v_n, p_n,
                        u_nm1, v_nm1, p_nm1,
                        u_nm2, v_nm2, p_nm2)
                    ml_tag = "ML"
                else:
                    p_init = p_n
                    ml_tag = "skip"

            u_star, v_star = _momentum_step(
                u, v, dx, dy, self.nu, self.dt, self.bc)

            if np.any(np.isnan(u_star)) or np.any(np.isnan(v_star)):
                raise RuntimeError(
                    f"NaN at step {step}: dt={self.dt:.5f} is unstable. "
                    f"Try --dt {self.dx**2 / (8.0 * self.nu):.5f}"
                )

            p, n_iter, res_norm = _gs_pressure_step(
                p_init, u_star, v_star,
                dx, dy, self.rho, self.dt,
                self.bc, self.max_iters, self.tol, self.pin_p, self.use_ser)

            if self.pin_p:
                p[1:-1, 1:-1] -= p[1:-1, 1:-1].mean()

            u, v = _velocity_correction(
                u_star, v_star, p, dx, dy, self.rho, self.dt, self.bc)

            u_nm2 = u_nm1; v_nm2 = v_nm1; p_nm2 = p_nm1
            u_nm1 = u_n;   v_nm1 = v_n;   p_nm1 = p_n

            total_gs  += n_iter
            residuals.append(res_norm)

            if self.num_t <= 10 or step % max(1, self.num_t // 10) == 0:
                avg = total_gs / (step + 1)
                print(f"  [{self.label}] step={step:4d}  GS={n_iter:4d}  "
                      f"avg={avg:.1f}  res={res_norm:.2e}  [{ml_tag}]")

        self.exec_time    = time.time() - t0
        self.avg_gs_iters = total_gs / max(self.num_t, 1)
        self.residuals    = residuals
        self.u_final      = u
        self.v_final      = v
        self.p_final      = p

        ser_tag = " + SER" if self.use_ser else ""
        self.report = (
            f"\n>>> {self.label} (ML + GS{ser_tag})\n"
            f">>> Execution time      : {self.exec_time:.2f} s\n"
            f">>> Avg GS iterations   : {self.avg_gs_iters:.1f}\n"
            f">>> Final residual      : {residuals[-1]:.2e}\n"
            f">>> Time steps          : {self.num_t}\n"
        )
        print(self.report)
        return u, v, p, residuals


# ══════════════════════════════════════════════════════════════════════════════
# Comparison
# ══════════════════════════════════════════════════════════════════════════════

def print_ns_comparison(solvers, names, u_ref, v_ref, p_ref, save_path=None):
    baseline_time  = solvers[0].exec_time
    baseline_iters = solvers[0].avg_gs_iters

    lines = [
        f"\n{'='*80}",
        "NS COMPARISON TABLE",
        f"{'='*80}",
        f"  Baseline: {names[0]}",
        "",
        (f"  {'Method':<26} {'Avg Iters':>10} {'Time(s)':>9} "
         f"{'Iter-Reduct':>12} {'Speedup':>9} "
         f"{'Max|Δu|':>9} {'Max|Δv|':>9} {'Max|Δp|':>9}"),
        f"  {'─'*96}",
    ]

    for s, name in zip(solvers, names):
        eu  = float(np.max(np.abs(u_ref[1:-1, 1:-1] - s.u_final[1:-1, 1:-1])))
        ev  = float(np.max(np.abs(v_ref[1:-1, 1:-1] - s.v_final[1:-1, 1:-1])))
        ep  = float(np.max(np.abs(p_ref[1:-1, 1:-1] - s.p_final[1:-1, 1:-1])))
        red = (1.0 - s.avg_gs_iters / baseline_iters) * 100 if baseline_iters else 0.0
        spd = baseline_time / max(s.exec_time, 1e-9)
        lines.append(
            f"  {name:<26} {s.avg_gs_iters:>10.1f} {s.exec_time:>9.2f} "
            f"{red:>+11.1f}% {spd:>8.2f}x "
            f"{eu:>9.2e} {ev:>9.2e} {ep:>9.2e}"
        )

    lines.append(f"  {'─'*96}")
    if len(solvers) >= 2:
        s0, s1 = solvers[0], solvers[1]
        red_h = (1 - s1.avg_gs_iters / max(s0.avg_gs_iters, 1e-9)) * 100
        spd_h = s0.exec_time / max(s1.exec_time, 1e-9)
        lines.append(
            f"\n  Hybrid vs Pure: {red_h:+.1f}% GS iters, {spd_h:.2f}x speedup"
        )
    lines.append(f"{'='*80}\n")

    text = "\n".join(lines)
    print(text)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as fh:
            fh.write(text)
        print(f"  Comparison saved: {save_path}")
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="NS ML-Hybrid solver: pure PDE vs ML-warm-start (pressure Poisson)"
    )
    p.add_argument("--config",    required=True,
                   help="Path to NS TOML config (e.g. input/lid.toml)")
    p.add_argument("--model",     required=True,
                   help="Path to model .pth  (e.g. output/pinn_model_mlp.pth)")
    p.add_argument("--norm",      required=True,
                   help="Path to normalization .pth  (e.g. output/norm_params_mlp.pth)")
    p.add_argument("--tol",       type=float, default=None,
                   help="Override GS tolerance for pressure Poisson")
    p.add_argument("--dt",        type=float, default=None,
                   help="Override time step (use CFL-stable value for explicit solver)")
    p.add_argument("--timesteps", type=int,   default=None,
                   help="Override number of time steps")
    p.add_argument("--use-ser",   action="store_true",
                   help="Enable SER over-relaxation in GS")
    p.add_argument("--output-dir", type=str,
                   default=os.path.join("output", "hybrid_ns_results"),
                   help="Directory for saved results (default: output/hybrid_ns_results)")
    args = p.parse_args()

    cfg = load_ns_config(args.config)
    if args.tol       is not None: cfg["tol_p"]    = args.tol
    if args.dt        is not None: cfg["dt"]        = args.dt
    if args.timesteps is not None: cfg["timesteps"] = args.timesteps

    dt_diff   = cfg["dx"] ** 2 / (4.0 * cfg["nu"])
    dt_cfl    = min(cfg["dx"], cfg["dy"])       
    dt_stable = min(dt_diff, dt_cfl) * 0.4     # 40 % safety margin
    if cfg["dt"] > dt_stable:
        print(f"\n  AUTO-CLAMP: dt={cfg['dt']:.5f} exceeds explicit stability limit.")
        print(f"  Diffusion limit = {dt_diff:.5f},  CFL limit = {dt_cfl:.5f}")
        print(f"  Setting dt = {dt_stable:.5f}  (40 % of min limit).")
        print(f"  Pass --dt <value> to override.\n")
        cfg["dt"] = dt_stable

    print(f"\n{'='*65}")
    print(f"  NS ML-Hybrid Solver")
    print(f"{'='*65}")
    print(f"  Config      : {args.config}")
    print(f"  Grid        : {cfg['nc_x']} × {cfg['nc_y']}  interior cells")
    print(f"  dx / dy     : {cfg['dx']:.5f} / {cfg['dy']:.5f}")
    print(f"  nu / rho    : {cfg['nu']} / {cfg['rho']}")
    print(f"  dt / steps  : {cfg['dt']} / {cfg['timesteps']}")
    print(f"  GS tolerance: {cfg['tol_p']}")
    print(f"  SER         : {args.use_ser}")
    print(f"  All-Neumann : {_is_all_neumann_p(cfg['bc'])}")

    print(f"\n--- Loading ML model ---")
    ml = MLInference(args.model, args.norm)

    print(f"\n{'─'*65}")
    print(f"  [1/2] Pure PDE — fractional step + Gauss-Seidel")
    print(f"{'─'*65}")
    solver_pure = PDESolverNS(cfg, use_ser=args.use_ser, label="PDE(GS)")
    u_p, v_p, p_p, res_p = solver_pure.run()

    print(f"\n{'─'*65}")
    print(f"  [2/2] ML-Hybrid — ML pressure guess + Gauss-Seidel")
    print(f"{'─'*65}")
    solver_hybrid = HybridSolverNS(cfg, ml, use_ser=args.use_ser, label="ML-Hybrid")
    u_h, v_h, p_h, res_h = solver_hybrid.run()

    os.makedirs(args.output_dir, exist_ok=True)
    cmp_path = os.path.join(args.output_dir, "comparison.txt")
    print_ns_comparison(
        [solver_pure, solver_hybrid],
        ["Pure PDE (GS)", "ML-Hybrid (ML+GS)"],
        u_ref=u_p, v_ref=v_p, p_ref=p_p,
        save_path=cmp_path,
    )

    for name, arr in [
        ("u_pure", u_p), ("v_pure", v_p), ("p_pure", p_p),
        ("u_hybrid", u_h), ("v_hybrid", v_h), ("p_hybrid", p_h),
        ("res_pure", np.array(res_p)), ("res_hybrid", np.array(res_h)),
    ]:
        np.save(os.path.join(args.output_dir, f"{name}.npy"), arr)

    import json
    meta = {
        "nc_x": cfg["nc_x"],  "nc_y": cfg["nc_y"],
        "L_x":  cfg["L_x"],   "L_y":  cfg["L_y"],
        "dx":   cfg["dx"],     "dy":   cfg["dy"],
        "nu":   cfg["nu"],     "rho":  cfg["rho"],
        "dt":   cfg["dt"],     "timesteps": cfg["timesteps"],
        "tol_p": cfg["tol_p"],
        "use_ser":          args.use_ser,
        "pure_time":        solver_pure.exec_time,
        "hybrid_time":      solver_hybrid.exec_time,
        "pure_avg_iters":   solver_pure.avg_gs_iters,
        "hybrid_avg_iters": solver_hybrid.avg_gs_iters,
        "case_config":      args.config,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  Results saved to: {args.output_dir}/")

    from swiftcfd.machineLearning.post_ns import run as post_run
    post_run(args.output_dir)


if __name__ == "__main__":
    main()
