"""
NS ML-Hybrid solver with a *real* PETSc pressure-Poisson baseline.

Motivation
----------
`hybrid_solver_ns.py` compares an ML-warm-started pressure solve against a
pure-Python Gauss-Seidel (GS) solver.  GS converges slowly, so a good initial
guess saves many sweeps and the ML benefit looks large.  That is NOT a fair
yardstick for a real CFD code, which uses a Krylov solver (BiCGStab / GMRES /
CG) with a strong preconditioner (e.g. algebraic multigrid, GAMG).

This module replaces the GS pressure step with PETSc -- the same library that
`swiftcfd/equations/linearAlgebraSolver/linearAlgebraSolver.py` uses in the
production solver -- and measures `ksp.getIterationNumber()`.  The honest
question it answers is:

    Does an ML-predicted initial guess reduce Krylov iterations compared with
    the standard practice of warm-starting from the previous time step?

Both the baseline and the ML-hybrid solve the SAME discrete pressure-Poisson
system to the SAME tolerance; they differ ONLY in the initial guess handed to
KSP.  The iteration count is therefore a clean measure of "solver cost".

The discrete operator (5-point Laplacian with ghost-cell Dirichlet/Neumann
folding) is built to match the GS discretization in `hybrid_solver_ns.py`, so
results are directly comparable across the two scripts.
"""

import argparse
import os
import time
import json

import numpy as np
from petsc4py import PETSc

# Reuse all the grid / BC / momentum / correction machinery already written
# and validated in the GS version -- only the pressure solve changes.
from swiftcfd.machineLearning.hybrid_solver_ns import (
    load_ns_config,
    apply_all_bc,
    _apply_one_bc,
    _is_all_neumann_p,
    rms,
    _momentum_step,
    _velocity_correction,
    print_ns_comparison,
)
from swiftcfd.machineLearning.hybrid_solver import MLInference


# ══════════════════════════════════════════════════════════════════════════════
# PETSc pressure-Poisson solver
# ══════════════════════════════════════════════════════════════════════════════

class PetscPressurePoisson:
    """
    Solves  ∇²p = (ρ/dt) ∇·u*   on the interior cells of an (nx × ny) grid that
    carries one ring of ghost cells, using PETSc KSP.

    We assemble  M = -∇²  (symmetric positive (semi-)definite) and  b = -rhs,
    so CG / GAMG behave well.  Dirichlet / Neumann boundary conditions are
    folded into M and b via the ghost-cell relations used by `_apply_one_bc`:

        dirichlet:  p_ghost = 2·val - p_interior_neighbour
        neumann:    p_ghost = p_interior_neighbour + val·h

    Because the geometry and BC *types* never change during a run, M is built
    ONCE.  Only b is rebuilt every time step.  This mirrors a production code,
    where the preconditioner (ILU / GAMG) is set up once and reused.
    """

    def __init__(self, cfg: dict,
                 solver: str = "BCGS", preconditioner: str = "GAMG",
                 rtol: float = 1e-8, max_it: int = 10000,
                 atol: float = None, unpreconditioned_norm: bool = False,
                 track_history: bool = False):
        self.nx = cfg["nx"]
        self.ny = cfg["ny"]
        self.dx = cfg["dx"]
        self.dy = cfg["dy"]
        self.rho = cfg["rho"]
        self.dt = cfg["dt"]
        self.bc = cfg["bc"]
        self.max_it = max_it

        self.nc_x = self.nx - 2          # interior cells in x
        self.nc_y = self.ny - 2          # interior cells in y
        self.N = self.nc_x * self.nc_y   # number of unknowns

        self.all_neumann = _is_all_neumann_p(self.bc)

        # --- matrix M = -∇²  (constant for the whole run) -------------------
        self.A = PETSc.Mat().create()
        self.A.setSizes([self.N, self.N])
        self.A.setType(PETSc.Mat.Type.SEQAIJ)
        self.A.setPreallocationNNZ(5)
        self.A.setUp()
        self._assemble_matrix()

        # pure-Neumann pressure is defined only up to a constant -> tell PETSc
        if self.all_neumann:
            ns = PETSc.NullSpace().create(constant=True)
            self.A.setNullSpace(ns)
            self.A.setNearNullSpace(ns)

        self.b = PETSc.Vec().createSeq(self.N)
        self.x = PETSc.Vec().createSeq(self.N)
        self._r = None                   # work vector for true_residual_norm

        # --- Krylov solver + preconditioner --------------------------------
        self.ksp = self._make_ksp(solver, preconditioner, rtol, max_it,
                                  atol=atol,
                                  unpreconditioned_norm=unpreconditioned_norm,
                                  track_history=track_history)
        self.ksp.setOperators(self.A)
        # we ALWAYS hand in an explicit initial guess (prev-step p or ML guess)
        self.ksp.setInitialGuessNonzero(True)

        self.solver_name = solver
        self.pc_name = preconditioner

    # ----------------------------------------------------------------------
    def _k(self, i, j):
        """Flat unknown index for interior cell (i, j), i∈[1,nx-2], j∈[1,ny-2]."""
        return (i - 1) * self.nc_y + (j - 1)

    def _assemble_matrix(self):
        dx2, dy2 = self.dx ** 2, self.dy ** 2
        ap = 2.0 / dx2 + 2.0 / dy2          # base diagonal of -∇²
        cx = -1.0 / dx2                     # east / west off-diagonal
        cy = -1.0 / dy2                     # north / south off-diagonal
        A = self.A
        bc = self.bc

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                k = self._k(i, j)
                diag = ap

                # (neighbour i-offset, j-offset, coeff, face, step) ----------
                neighbours = [
                    (+1, 0, cx, "east",  self.dx),
                    (-1, 0, cx, "west",  self.dx),
                    (0, +1, cy, "north", self.dy),
                    (0, -1, cy, "south", self.dy),
                ]
                for di, dj, c, face, h in neighbours:
                    ii, jj = i + di, j + dj
                    is_interior = (1 <= ii <= self.nx - 2) and (1 <= jj <= self.ny - 2)
                    if is_interior:
                        A.setValue(k, self._k(ii, jj), c,
                                   addv=PETSc.InsertMode.ADD_VALUES)
                    else:
                        # neighbour is a ghost -> fold the BC into M (and later b)
                        bc_type, _ = bc[face]["p"]
                        if bc_type == "dirichlet":
                            diag += -c          # +1/h²  (stiffens)
                        else:                   # neumann
                            diag += c           # -1/h²  (softens)

                A.setValue(k, k, diag, addv=PETSc.InsertMode.ADD_VALUES)

        A.assemblyBegin()
        A.assemblyEnd()

    def _make_ksp(self, solver, preconditioner, rtol, max_it,
                  atol=None, unpreconditioned_norm=False, track_history=False):
        ksp = PETSc.KSP().create()

        types = {
            "RICHARDSON": PETSc.KSP.Type.RICHARDSON,
            "CG":         PETSc.KSP.Type.CG,
            "BCGS":       PETSc.KSP.Type.BCGS,
            "GMRES":      PETSc.KSP.Type.GMRES,
        }
        if solver.upper() not in types:
            raise ValueError(f"Unknown solver '{solver}'. "
                             f"Choose from {list(types)}")
        ksp.setType(types[solver.upper()])

        pc = ksp.getPC()
        pcu = preconditioner.upper()
        if pcu == "JACOBI":
            pc.setType(PETSc.PC.Type.JACOBI)
        elif pcu == "ILU":
            pc.setType(PETSc.PC.Type.ILU)
        elif pcu == "SOR":
            pc.setType(PETSc.PC.Type.SOR)
        elif pcu == "NONE":
            pc.setType(PETSc.PC.Type.NONE)
        elif pcu == "GAMG":
            pc.setType("gamg")
            opts = PETSc.Options()
            opts["pc_gamg_type"] = "agg"
            opts["pc_gamg_threshold"] = 0.01
            opts["pc_gamg_square_graph"] = 1
            opts["mg_levels_ksp_type"] = "richardson"
            opts["mg_levels_pc_type"] = "jacobi"
            opts["mg_levels_ksp_max_it"] = 3
            opts["mg_coarse_ksp_type"] = "preonly"
            opts["mg_coarse_pc_type"] = "lu"
            ksp.setFromOptions()
        else:
            raise ValueError(f"Unknown preconditioner '{preconditioner}'.")

        # --- convergence criterion (set LAST so GAMG's setFromOptions can't
        #     silently override it) ------------------------------------------
        # UNPRECONDITIONED: stop on the TRUE residual ‖b - A x‖, not the
        # preconditioner-distorted one, so every (pc, init) races to the same
        # measurable finish line.
        if unpreconditioned_norm:
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            if solver.upper() == "GMRES":
                # GMRES can only monitor the true residual with RIGHT pc
                ksp.setPCSide(PETSc.PC.Side.RIGHT)

        # When atol is given we make the ABSOLUTE tolerance the single finish
        # line (rtol≈0), so a better initial guess is NOT punished with a
        # tighter target the way a purely relative rtol·‖r0‖ criterion does.
        # divtol is raised because a poor ML guess can legitimately start with
        # r0 >> ‖b‖ without the iteration actually diverging.
        if atol is not None:
            ksp.setTolerances(rtol=1e-50, atol=atol, divtol=1e10, max_it=max_it)
        else:
            ksp.setTolerances(rtol=rtol, max_it=max_it)

        if track_history:
            # reset=True: history restarts for each solve; the default (False)
            # ACCUMULATES across solves, silently concatenating the curves of
            # consecutive initial guesses.
            ksp.setConvergenceHistory(reset=True)

        return ksp

    # ----------------------------------------------------------------------
    def set_absolute_tol(self, atol):
        """Set a per-solve ABSOLUTE finish line (rtol≈0), used by the fair sweep
        to give zero/prev/ml the identical target on each linear system."""
        self.ksp.setTolerances(rtol=1e-50, atol=atol, divtol=1e10,
                               max_it=self.max_it)

    def rhs_norm(self):
        """‖b‖₂  — equals the residual of the zero initial guess (r0 = b - A·0)."""
        return self.b.norm()

    def load_guess(self, p_init):
        """Load a full-grid (ghosted) pressure field into x as the KSP initial
        guess.  The subsequent ksp.solve(b, x) starts from it because the KSP
        was created with setInitialGuessNonzero(True)."""
        arr = self.x.getArray()
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                arr[self._k(i, j)] = p_init[i, j]
        self.x.assemble()

    def true_residual_norm(self):
        """Explicit ‖b − A·x‖₂ of the CURRENT x (the guess before ksp.solve,
        the solution after) — computed by an actual mat-vec, so it does not
        depend on the KSP's norm type or convergence bookkeeping."""
        if self._r is None:
            self._r = self.b.duplicate()
        self.A.mult(self.x, self._r)
        self._r.aypx(-1.0, self.b)          # r = b - A·x
        return self._r.norm()

    def convergence_history(self):
        """Residual norm at every iteration of the most recent solve."""
        return self.ksp.getConvergenceHistory()

    # ----------------------------------------------------------------------
    def build_rhs(self, u_star, v_star):
        """RHS vector b = -rhs_poisson + boundary contributions."""
        dx, dy = self.dx, self.dy
        dx2, dy2 = dx ** 2, dy ** 2
        cx, cy = -1.0 / dx2, -1.0 / dy2
        bc = self.bc

        # divergence of u* at interior cells  ->  ρ/dt · ∇·u*
        rhs = (self.rho / self.dt) * (
            (u_star[2:, 1:-1] - u_star[:-2, 1:-1]) / (2.0 * dx) +
            (v_star[1:-1, 2:] - v_star[1:-1, :-2]) / (2.0 * dy)
        )

        self.b.zeroEntries()
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                k = self._k(i, j)
                bk = -rhs[i - 1, j - 1]      # base RHS for  M p = b

                neighbours = [
                    (+1, 0, cx, "east",  dx),
                    (-1, 0, cx, "west",  dx),
                    (0, +1, cy, "north", dy),
                    (0, -1, cy, "south", dy),
                ]
                for di, dj, c, face, h in neighbours:
                    ii, jj = i + di, j + dj
                    is_interior = (1 <= ii <= self.nx - 2) and (1 <= jj <= self.ny - 2)
                    if not is_interior:
                        bc_type, val = bc[face]["p"]
                        if bc_type == "dirichlet":
                            bk += -2.0 * c * val        # +2·val/h²
                        else:                           # neumann
                            bk += -c * val * h          # +val/h
                self.b.setValue(k, bk, addv=PETSc.InsertMode.INSERT_VALUES)

        self.b.assemblyBegin()
        self.b.assemblyEnd()

        if self.all_neumann:
            # make b consistent with the constant nullspace
            ns = self.A.getNullSpace()
            ns.remove(self.b)

    def solve(self, p_init, u_star, v_star):
        """
        Solve the pressure Poisson system with `p_init` (full grid, incl.
        ghosts) as the KSP initial guess.  Returns (p_full, n_iter, res_norm).
        """
        self.build_rhs(u_star, v_star)
        self.load_guess(p_init)
        self.ksp.solve(self.b, self.x)

        n_iter = self.ksp.getIterationNumber()
        res_norm = self.ksp.getResidualNorm()

        # scatter solution back onto the full ghosted grid
        p = p_init.copy()
        sol = self.x.getArray()
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                p[i, j] = sol[self._k(i, j)]

        _apply_one_bc(p, self.bc, self.dx, self.dy, "p")
        if self.all_neumann:
            p[1:-1, 1:-1] -= p[1:-1, 1:-1].mean()
        return p, n_iter, res_norm


# ══════════════════════════════════════════════════════════════════════════════
# ML pressure guess  (21-feature stencil -> centre value, same as GS hybrid)
# ══════════════════════════════════════════════════════════════════════════════

def ml_pressure_guess(ml: MLInference, bc, dx, dy, nx, ny, all_neumann,
                      u_n, v_n, p_n, u_nm1, v_nm1, p_nm1, u_nm2, v_nm2, p_nm2):
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
        p_guess[i, j] = preds[idx, 0]
    _apply_one_bc(p_guess, bc, dx, dy, "p")
    if all_neumann:
        p_guess -= p_guess[1:-1, 1:-1].mean()
    return p_guess


# ══════════════════════════════════════════════════════════════════════════════
# Runners
# ══════════════════════════════════════════════════════════════════════════════

class _BasePetscRunner:
    def __init__(self, cfg, poisson: PetscPressurePoisson, label):
        self.cfg = cfg
        self.poisson = poisson
        self.label = label
        self.bc = cfg["bc"]
        self.nu = cfg["nu"]
        self.dt = cfg["dt"]
        self.num_t = cfg["timesteps"]
        self.nx, self.ny = cfg["nx"], cfg["ny"]
        self.dx, self.dy = cfg["dx"], cfg["dy"]
        self.all_neumann = _is_all_neumann_p(self.bc)

    def get_p_init(self, step, hist):
        """Override: return (p_init, tag)."""
        raise NotImplementedError

    def run(self):
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy

        u = np.zeros((nx, ny)); v = np.zeros((nx, ny)); p = np.zeros((nx, ny))
        apply_all_bc(u, v, p, self.bc, dx, dy)

        hist = {
            "u_nm1": u.copy(), "v_nm1": v.copy(), "p_nm1": p.copy(),
            "u_nm2": u.copy(), "v_nm2": v.copy(), "p_nm2": p.copy(),
        }

        total_iters = 0
        residuals = []
        t0 = time.time()

        for step in range(self.num_t):
            u_n, v_n, p_n = u.copy(), v.copy(), p.copy()

            u_star, v_star = _momentum_step(u, v, dx, dy, self.nu, self.dt, self.bc)
            if np.any(np.isnan(u_star)) or np.any(np.isnan(v_star)):
                raise RuntimeError(f"NaN at step {step}: dt={self.dt:.5f} unstable.")

            p_init, tag = self.get_p_init(step, {**hist, "u_n": u_n, "v_n": v_n, "p_n": p_n})

            p, n_iter, res_norm = self.poisson.solve(p_init, u_star, v_star)

            u, v = _velocity_correction(u_star, v_star, p, dx, dy,
                                        self.cfg["rho"], self.dt, self.bc)

            hist["u_nm2"], hist["v_nm2"], hist["p_nm2"] = hist["u_nm1"], hist["v_nm1"], hist["p_nm1"]
            hist["u_nm1"], hist["v_nm1"], hist["p_nm1"] = u_n, v_n, p_n

            total_iters += n_iter
            residuals.append(res_norm)

            if self.num_t <= 10 or step % max(1, self.num_t // 10) == 0:
                avg = total_iters / (step + 1)
                print(f"  [{self.label}] step={step:4d}  KSP={n_iter:4d}  "
                      f"avg={avg:.1f}  res={res_norm:.2e}  [{tag}]")

        self.exec_time = time.time() - t0
        self.avg_gs_iters = total_iters / max(self.num_t, 1)   # name reused by print_ns_comparison
        self.residuals = residuals
        self.u_final, self.v_final, self.p_final = u, v, p

        print(f"\n>>> {self.label}  ({self.poisson.solver_name}+{self.poisson.pc_name})\n"
              f">>> Execution time     : {self.exec_time:.2f} s\n"
              f">>> Avg KSP iterations : {self.avg_gs_iters:.1f}\n"
              f">>> Final residual     : {residuals[-1]:.2e}\n"
              f">>> Time steps         : {self.num_t}\n")
        return u, v, p, residuals


class PetscBaselineNS(_BasePetscRunner):
    """Strong baseline: warm-start from the previous time-step pressure."""
    def get_p_init(self, step, hist):
        return hist["p_n"], "prev-p"


class PetscHybridNS(_BasePetscRunner):
    """ML warm-start: initial guess from the trained model."""
    def __init__(self, cfg, poisson, ml: MLInference, label="ML-Hybrid(PETSc)"):
        super().__init__(cfg, poisson, label)
        self.ml = ml
        self.skip_thresh = cfg["tol_p"] * 10

    def get_p_init(self, step, hist):
        if step < 2:
            return hist["p_n"], "boot"
        if rms(hist["p_n"], hist["p_nm1"]) <= self.skip_thresh:
            return hist["p_n"], "skip"     # essentially steady -> ML not worth it
        guess = ml_pressure_guess(
            self.ml, self.bc, self.dx, self.dy, self.nx, self.ny, self.all_neumann,
            hist["u_n"], hist["v_n"], hist["p_n"],
            hist["u_nm1"], hist["v_nm1"], hist["p_nm1"],
            hist["u_nm2"], hist["v_nm2"], hist["p_nm2"])
        return guess, "ML"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="NS ML-Hybrid solver with a real PETSc pressure baseline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model",  required=True)
    ap.add_argument("--norm",   required=True)
    ap.add_argument("--solver", default="BCGS",
                    choices=["BCGS", "GMRES", "CG", "RICHARDSON"],
                    help="PETSc Krylov solver (default BCGS)")
    ap.add_argument("--pc", default="GAMG",
                    choices=["GAMG", "ILU", "SOR", "JACOBI", "NONE"],
                    help="PETSc preconditioner (default GAMG)")
    ap.add_argument("--tol", type=float, default=None, help="KSP rtol override")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--output-dir", default=os.path.join("output", "hybrid_ns_petsc"))
    args = ap.parse_args()

    cfg = load_ns_config(args.config)
    if args.tol       is not None: cfg["tol_p"]     = args.tol
    if args.dt        is not None: cfg["dt"]         = args.dt
    if args.timesteps is not None: cfg["timesteps"]  = args.timesteps

    # explicit-time-integration stability clamp (same as GS script)
    dt_diff   = cfg["dx"] ** 2 / (4.0 * cfg["nu"])
    dt_cfl    = min(cfg["dx"], cfg["dy"])
    dt_stable = min(dt_diff, dt_cfl) * 0.4
    if cfg["dt"] > dt_stable:
        print(f"\n  AUTO-CLAMP: dt={cfg['dt']:.5f} -> {dt_stable:.5f} (explicit limit).\n")
        cfg["dt"] = dt_stable

    rtol = cfg["tol_p"]

    print(f"\n{'='*65}\n  NS ML-Hybrid Solver  (PETSc baseline)\n{'='*65}")
    print(f"  Config   : {args.config}")
    print(f"  Grid     : {cfg['nc_x']} × {cfg['nc_y']} interior cells")
    print(f"  nu / rho : {cfg['nu']} / {cfg['rho']}")
    print(f"  dt/steps : {cfg['dt']} / {cfg['timesteps']}")
    print(f"  KSP      : {args.solver} + {args.pc}   rtol={rtol}")
    print(f"  AllNeumann(p): {_is_all_neumann_p(cfg['bc'])}")

    print(f"\n--- Loading ML model ---")
    ml = MLInference(args.model, args.norm)

    # ONE matrix, shared by both runs (identical discrete system) -------------
    poisson = PetscPressurePoisson(cfg, solver=args.solver, preconditioner=args.pc,
                                   rtol=rtol)

    print(f"\n{'─'*65}\n  [1/2] Baseline — PETSc, warm-start from previous step\n{'─'*65}")
    baseline = PetscBaselineNS(cfg, poisson, label="PETSc-baseline")
    u_b, v_b, p_b, res_b = baseline.run()

    print(f"\n{'─'*65}\n  [2/2] ML-Hybrid — PETSc, ML initial guess\n{'─'*65}")
    hybrid = PetscHybridNS(cfg, poisson, ml, label="ML-Hybrid(PETSc)")
    u_h, v_h, p_h, res_h = hybrid.run()

    os.makedirs(args.output_dir, exist_ok=True)
    print_ns_comparison(
        [baseline, hybrid],
        ["PETSc baseline (prev-p)", "ML-Hybrid (PETSc)"],
        u_ref=u_b, v_ref=v_b, p_ref=p_b,
        save_path=os.path.join(args.output_dir, "comparison.txt"),
    )

    for name, arr in [
        ("u_base", u_b), ("v_base", v_b), ("p_base", p_b),
        ("u_hybrid", u_h), ("v_hybrid", v_h), ("p_hybrid", p_h),
        ("res_base", np.array(res_b)), ("res_hybrid", np.array(res_h)),
    ]:
        np.save(os.path.join(args.output_dir, f"{name}.npy"), arr)

    meta = {
        "nc_x": cfg["nc_x"], "nc_y": cfg["nc_y"],
        "dx": cfg["dx"], "dy": cfg["dy"], "nu": cfg["nu"], "rho": cfg["rho"],
        "dt": cfg["dt"], "timesteps": cfg["timesteps"], "rtol": rtol,
        "solver": args.solver, "pc": args.pc,
        "baseline_time": baseline.exec_time, "hybrid_time": hybrid.exec_time,
        "baseline_avg_iters": baseline.avg_gs_iters,
        "hybrid_avg_iters": hybrid.avg_gs_iters,
        "case_config": args.config,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
