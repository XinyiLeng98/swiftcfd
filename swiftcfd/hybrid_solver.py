import argparse
import os
import time

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib      # pip install tomli  (Python < 3.11)


# ══════════════════════════════════════════════════════════════════════════════
# Config reader
# ══════════════════════════════════════════════════════════════════════════════

def load_config(toml_path: str) -> dict:
    """Extract the parameters needed for the standalone GS solver."""
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    alpha     = float(cfg["solver"]["fluid"]["alpha"])
    dt        = float(cfg["solver"]["time"]["dt"])
    timesteps = int(cfg["solver"]["time"]["timesteps"])
    max_iters = int(cfg["solver"]["linearSolver"]["maxIterations"]["T"])
    tol       = float(cfg["solver"]["linearSolver"]["tolerance"]["T"])

    # ── mesh: extract outer-wall BCs from the 4-block layout ──────────────
    # block1=SW, block2=SE, block3=NW, block4=NE
    # Outer walls: north→block3/4.north, south→block1/2.south,
    #              east→block2/4.east,   west→block1/3.west
    mesh   = cfg["mesh"]
    b1     = mesh["block1"]
    num_x  = int(b1["x"]["numCells"]) * 2 - 1   # 21*2-1 = 41
    num_y  = int(b1["y"]["numCells"]) * 2 - 1
    L_x    = float(mesh["block2"]["x"]["end"]) - float(b1["x"]["start"])
    L_y    = float(mesh["block4"]["y"]["end"]) - float(b1["y"]["start"])

    bc_cfg = cfg["boundaryCondition"]

    def _val(block, face):
        return float(bc_cfg[block][face]["T"]["value"])

    def _type(block, face):
        return bc_cfg[block][face]["T"]["type"]

    north_val  = _val("block3", "north")
    north_type = _type("block3", "north")
    south_val  = _val("block1", "south")
    south_type = _type("block1", "south")
    east_val   = _val("block2", "east")
    east_type  = _type("block2", "east")
    west_val   = _val("block1", "west")
    west_type  = _type("block1", "west")

    return {
        "alpha": alpha, "dt": dt, "timesteps": timesteps,
        "max_iters": max_iters, "tol": tol,
        "num_x": num_x, "num_y": num_y, "L_x": L_x, "L_y": L_y,
        "bc": {
            "north": (north_type, north_val),
            "south": (south_type, south_val),
            "east":  (east_type,  east_val),
            "west":  (west_type,  west_val),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Boundary conditions
# ══════════════════════════════════════════════════════════════════════════════

def apply_bc(T, bc, dx, dy):
    """Apply Dirichlet / Neumann BCs on all four walls in-place."""
    for face, (bc_type, val) in bc.items():
        if face == "north":
            if bc_type == "dirichlet":
                T[1:-1, -1] = val
            else:                                    # neumann
                T[1:-1, -1] = T[1:-1, -2] + val * dy

        elif face == "south":
            if bc_type == "dirichlet":
                T[1:-1, 0] = val
            else:
                T[1:-1, 0] = T[1:-1, 1] - val * dy

        elif face == "east":
            if bc_type == "dirichlet":
                T[-1, 1:-1] = val
            else:
                T[-1, 1:-1] = T[-2, 1:-1] + val * dx

        elif face == "west":
            if bc_type == "dirichlet":
                T[0, 1:-1] = val
            else:
                T[0, 1:-1] = T[1, 1:-1] - val * dx

    # Corners: average of adjacent walls
    nv = bc["north"][1]; sv = bc["south"][1]
    ev = bc["east"][1];  wv = bc["west"][1]
    T[0,  0]  = 0.5 * (wv + sv)
    T[-1, 0]  = 0.5 * (ev + sv)
    T[0,  -1] = 0.5 * (wv + nv)
    T[-1, -1] = 0.5 * (ev + nv)


def rms(A, B):
    return float(np.sqrt(np.mean((A - B) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# ML Inference wrapper
# ══════════════════════════════════════════════════════════════════════════════

class MLInference:
    """Loads MLP_model.pth + normalization.pth and wraps batch prediction."""

    def __init__(self, model_path: str, norm_path: str):
        from swiftcfd.machineLearning.model.modelFactory import create_model

        params = torch.load(norm_path, weights_only=False)
        self.X_mean = params["X_mean"]
        self.X_std  = params["X_std"]
        self.Y_mean = params["Y_mean"]
        self.Y_std  = params["Y_std"]

        input_size  = params.get("input_size",  len(self.X_mean))
        hidden_size = params.get("hidden_size", 256)
        num_layers  = params.get("num_layers",  5)
        model_type  = params.get("model_type",  "mlp")

        self.model = create_model(
            model_type, "T", "T",
            input_size=input_size, hidden_size=hidden_size,
            output_size=5, num_layers=num_layers, dropout=0.0,
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model.eval()

        print(f"  ML model loaded: type={model_type}, "
              f"input={input_size}, hidden={hidden_size}×{num_layers}")

    def predict(self, X: np.ndarray) -> np.ndarray:
 
        squeeze = X.ndim == 1
        if squeeze:
            X = X[None, :]
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            Xn = (Xt - self.X_mean) / (self.X_std + 1e-8)
            Yn = self.model(Xn)
            Y  = Yn * (self.Y_std + 1e-8) + self.Y_mean
        out = Y.numpy()
        return out[0] if squeeze else out


# ══════════════════════════════════════════════════════════════════════════════
# GS solver sub-step (shared by both methods)
# ══════════════════════════════════════════════════════════════════════════════

def _gs_step(Tnp1, Tn, bc, nx, ny, dx, dy, alpha, dt,
             max_iters, tol, use_ser=False):

    omega    = 1.0
    res_prev = None
    norm_factor = None

    for k in range(max_iters):
        T_old = Tnp1.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                ap   = 1.0/dt + 2.0*alpha/dx**2 + 2.0*alpha/dy**2
                ae   = -alpha/dx**2 * T_old[i+1, j]       # east  (not yet updated)
                aw   = -alpha/dx**2 * Tnp1[i-1, j]        # west  (already updated)
                an   = -alpha/dy**2 * T_old[i, j+1]       # north (not yet updated)
                as_  = -alpha/dy**2 * Tnp1[i, j-1]        # south (already updated)
                b    = Tn[i, j] / dt
                T_gs = (b - ae - aw - an - as_) / ap

                if use_ser:
                    Tnp1[i, j] = (1.0 - omega) * T_old[i, j] + omega * T_gs
                else:
                    Tnp1[i, j] = T_gs

        apply_bc(Tnp1, bc, dx, dy)

        res = rms(Tnp1, T_old)
        if norm_factor is None:
            norm_factor = res if res > 1e-30 else 1.0
        res_norm = res / norm_factor

        if use_ser and res_prev is not None and res > 1e-30:
            omega = omega * (res_prev / res) ** 0.3
            omega = max(0.5, min(omega, 1.8))
        res_prev = res

        if res_norm < tol:
            return k + 1, res_norm

    return max_iters, res_norm


# ══════════════════════════════════════════════════════════════════════════════
# Shared solver base
# ══════════════════════════════════════════════════════════════════════════════

class _BaseSolver:
    def __init__(self, cfg, use_ser=False, label=""):
        self.cfg       = cfg
        self.bc        = cfg["bc"]
        self.use_ser   = use_ser
        self.label     = label
        self.num_x     = cfg["num_x"]
        self.num_y     = cfg["num_y"]
        self.dx        = cfg["L_x"] / (cfg["num_x"] - 1)
        self.dy        = cfg["L_y"] / (cfg["num_y"] - 1)
        self.alpha     = cfg["alpha"]
        self.dt        = cfg["dt"]
        self.num_t     = cfg["timesteps"]
        self.max_iters = cfg["max_iters"]
        self.tol       = cfg["tol"]


# ══════════════════════════════════════════════════════════════════════════════
# Pure PDE solver
# ══════════════════════════════════════════════════════════════════════════════

class PDESolver(_BaseSolver):
    def __init__(self, cfg, use_ser=False, label="PDE(GS)"):
        super().__init__(cfg, use_ser, label)

    def run(self):
        nx, ny = self.num_x, self.num_y
        dx, dy = self.dx, self.dy

        T = np.zeros((nx, ny))
        apply_bc(T, self.bc, dx, dy)

        total_gs = 0
        residuals = []
        t0 = time.time()

        for step in range(self.num_t):
            Tn   = T.copy()
            Tnp1 = T.copy()

            n_iter, res_norm = _gs_step(
                Tnp1, Tn, self.bc, nx, ny, dx, dy,
                self.alpha, self.dt, self.max_iters, self.tol, self.use_ser,
            )
            T = Tnp1
            total_gs += n_iter
            residuals.append(res_norm)

            if self.num_t <= 10 or step % max(1, self.num_t // 10) == 0:
                avg = total_gs / (step + 1)
                print(f"  [{self.label}] step={step:4d}  GS={n_iter:4d}  "
                      f"avg={avg:.1f}  res={res_norm:.2e}")

        self.exec_time    = time.time() - t0
        self.avg_gs_iters = total_gs / max(self.num_t, 1)
        self.residuals    = residuals
        self.T_final      = T

        ser_tag = " + SER" if self.use_ser else ""
        self.report = (
            f"\n>>> {self.label} (GS{ser_tag})\n"
            f">>> Execution time        : {self.exec_time:.2f} s\n"
            f">>> Avg GS iterations     : {self.avg_gs_iters:.1f}\n"
            f">>> Final residual        : {residuals[-1]:.2e}\n"
            f">>> Time steps            : {self.num_t}\n"
        )
        print(self.report)
        return T, residuals


# ══════════════════════════════════════════════════════════════════════════════
# ML-Hybrid solver
# ══════════════════════════════════════════════════════════════════════════════

class HybridSolver(_BaseSolver):
    """
    Steps 0-1: pure GS (need two time levels to feed the ML model).
    Steps 2+:  ML warm-start → GS refinement.
    """

    def __init__(self, cfg, ml: MLInference, use_ser=False, label="ML-Hybrid"):
        super().__init__(cfg, use_ser, label)
        self.ml = ml

    def _ml_initial_guess(self, Tn, Tnm1, Tnm2):
        """Use the ML model to predict T^{n+1} for every interior cell."""
        nx, ny = self.num_x, self.num_y
        Tnp1   = np.zeros((nx, ny))

        rows, coords = [], []
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                rows.append([
                    Tn[i, j],   Tn[i+1, j],  Tn[i-1, j],
                    Tn[i, j+1], Tn[i, j-1],
                    Tnm1[i, j], Tnm2[i, j],
                ])
                coords.append((i, j))

        batch = np.array(rows, dtype=np.float32)
        preds = self.ml.predict(batch)          # (N, 5): [Tc, Te, Tw, Tn, Ts]

        for idx, (i, j) in enumerate(coords):
            Tnp1[i, j] = preds[idx, 0]         # centre value only

        apply_bc(Tnp1, self.bc, self.dx, self.dy)
        return Tnp1

    def run(self):
        nx, ny = self.num_x, self.num_y
        dx, dy = self.dx, self.dy

        T    = np.zeros((nx, ny))
        apply_bc(T, self.bc, dx, dy)
        Tnm2 = T.copy()
        Tnm1 = T.copy()

        total_gs  = 0
        residuals = []
        t0        = time.time()
        skip_thresh = self.tol * 10      # skip ML when field is essentially steady

        for step in range(self.num_t):
            Tn = T.copy()

            if step < 2:
                # Bootstrap: pure GS for first two steps
                Tnp1 = T.copy()
                n_iter, res_norm = _gs_step(
                    Tnp1, Tn, self.bc, nx, ny, dx, dy,
                    self.alpha, self.dt, self.max_iters, self.tol, self.use_ser,
                )
                ml_tag = "boot"
            else:
                change = rms(Tn, Tnm1)
                if change > skip_thresh:
                    Tnp1 = self._ml_initial_guess(Tn, Tnm1, Tnm2)
                    ml_tag = "ML"
                else:
                    Tnp1 = Tn.copy()
                    ml_tag = "skip"

                n_iter, res_norm = _gs_step(
                    Tnp1, Tn, self.bc, nx, ny, dx, dy,
                    self.alpha, self.dt, self.max_iters, self.tol, self.use_ser,
                )

            Tnm2 = Tnm1
            Tnm1 = Tn
            T    = Tnp1

            total_gs += n_iter
            residuals.append(res_norm)

            if self.num_t <= 10 or step % max(1, self.num_t // 10) == 0:
                avg = total_gs / (step + 1)
                print(f"  [{self.label}] step={step:4d}  GS={n_iter:4d}  "
                      f"avg={avg:.1f}  res={res_norm:.2e}  [{ml_tag}]")

        self.exec_time    = time.time() - t0
        self.avg_gs_iters = total_gs / max(self.num_t, 1)
        self.residuals    = residuals
        self.T_final      = T

        ser_tag = " + SER" if self.use_ser else ""
        self.report = (
            f"\n>>> {self.label} (ML + GS{ser_tag})\n"
            f">>> Execution time        : {self.exec_time:.2f} s\n"
            f">>> Avg GS iterations     : {self.avg_gs_iters:.1f}\n"
            f">>> Final residual        : {residuals[-1]:.2e}\n"
            f">>> Time steps            : {self.num_t}\n"
        )
        print(self.report)
        return T, residuals


# ══════════════════════════════════════════════════════════════════════════════
# Comparison helper
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(solvers, names, T_ref, save_path=None):
    baseline_time  = solvers[0].exec_time
    baseline_iters = solvers[0].avg_gs_iters

    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("COMPARISON TABLE")
    lines.append(f"{'='*70}")
    lines.append(f"  Baseline: {names[0]}")
    lines.append(f"")
    lines.append(f"  {'Method':<26} {'Avg Iters':>10} {'Time(s)':>9} "
                 f"{'Iter-Reduct':>12} {'Speedup':>9} {'Max|ΔT|':>10}")
    lines.append(f"  {'─'*80}")

    for s, name in zip(solvers, names):
        err  = float(np.max(np.abs(T_ref - s.T_final)))
        red  = (1.0 - s.avg_gs_iters / baseline_iters) * 100 if baseline_iters else 0.0
        spd  = baseline_time / max(s.exec_time, 1e-9)
        lines.append(
            f"  {name:<26} {s.avg_gs_iters:>10.1f} {s.exec_time:>9.2f} "
            f"{red:>+11.1f}% {spd:>8.2f}x {err:>10.2e}"
        )

    lines.append(f"  {'─'*80}")

    # Highlight hybrid vs pure
    if len(solvers) >= 2:
        s_pure   = solvers[0]
        s_hybrid = solvers[1]
        red_h = (1 - s_hybrid.avg_gs_iters / s_pure.avg_gs_iters) * 100
        spd_h = s_pure.exec_time / max(s_hybrid.exec_time, 1e-9)
        lines.append(f"\n  Hybrid vs Pure GS: {red_h:+.1f}% iterations, {spd_h:.2f}x speedup")

    lines.append(f"{'='*70}\n")

    text = "\n".join(lines)
    print(text)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(text)
        print(f"  Comparison saved: {save_path}")

    return text


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Run pure PDE and ML-hybrid solvers; compare performance"
    )
    p.add_argument("--config",  required=True,
                   help="Path to heatedCavity TOML config")
    p.add_argument("--model",   required=True,
                   help="Path to MLP_model.pth")
    p.add_argument("--norm",    required=True,
                   help="Path to normalization.pth")
    p.add_argument("--tol",     type=float, default=None,
                   help="Override GS tolerance (default: from config)")
    p.add_argument("--use-ser", action="store_true",
                   help="Enable SER over-relaxation in GS")
    p.add_argument("--north",  type=float, default=None,
                   help="Override north-wall temperature")
    p.add_argument("--south",  type=float, default=None,
                   help="Override south-wall temperature")
    p.add_argument("--east",   type=float, default=None,
                   help="Override east-wall temperature")
    p.add_argument("--west",   type=float, default=None,
                   help="Override west-wall temperature")
    p.add_argument("--output-dir", type=str, default=os.path.join("output", "hybrid_results"),
                   help="Directory for saved results")
    args = p.parse_args()

    # ── Load config ────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.tol is not None:
        cfg["tol"] = args.tol
    for face in ("north", "south", "east", "west"):
        val = getattr(args, face)
        if val is not None:
            cfg["bc"][face] = ("dirichlet", val)

    print(f"\n{'='*65}")
    print(f"  heatedCavity ML-Hybrid Solver")
    print(f"{'='*65}")
    print(f"  Config        : {args.config}")
    print(f"  Grid          : {cfg['num_x']} × {cfg['num_y']}")
    print(f"  Domain        : [{cfg['L_x']} × {cfg['L_y']}]")
    print(f"  alpha / dt    : {cfg['alpha']} / {cfg['dt']}")
    print(f"  Time steps    : {cfg['timesteps']}")
    print(f"  GS tolerance  : {cfg['tol']}")
    print(f"  SER           : {args.use_ser}")
    print(f"  BCs           : N={cfg['bc']['north'][1]} S={cfg['bc']['south'][1]} "
          f"E={cfg['bc']['east'][1]} W={cfg['bc']['west'][1]}")

    # ── Load ML model ──────────────────────────────────────────────────────
    print(f"\n--- Loading ML model ---")
    ml = MLInference(args.model, args.norm)

    # ── Run pure PDE ───────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  [1/2] Pure PDE — Gauss-Seidel")
    print(f"{'─'*65}")
    solver_pure = PDESolver(cfg, use_ser=args.use_ser, label="PDE(GS)")
    T_pure, res_pure = solver_pure.run()

    # ── Run ML-hybrid ──────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  [2/2] ML-Hybrid — ML guess + Gauss-Seidel")
    print(f"{'─'*65}")
    solver_hybrid = HybridSolver(cfg, ml, use_ser=args.use_ser, label="ML-Hybrid")
    T_hybrid, res_hybrid = solver_hybrid.run()

    # ── Comparison ────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    cmp_path = os.path.join(args.output_dir, "comparison.txt")
    print_comparison(
        [solver_pure, solver_hybrid],
        ["Pure PDE (GS)", "ML-Hybrid (ML+GS)"],
        T_ref=T_pure,
        save_path=cmp_path,
    )

    # ── Save arrays ───────────────────────────────────────────────────────
    np.save(os.path.join(args.output_dir, "T_pure.npy"),   T_pure)
    np.save(os.path.join(args.output_dir, "T_hybrid.npy"), T_hybrid)
    np.save(os.path.join(args.output_dir, "res_pure.npy"),   np.array(res_pure))
    np.save(os.path.join(args.output_dir, "res_hybrid.npy"), np.array(res_hybrid))

    # Save config for post.py
    import json
    meta = {
        "num_x": cfg["num_x"], "num_y": cfg["num_y"],
        "L_x": cfg["L_x"],     "L_y": cfg["L_y"],
        "alpha": cfg["alpha"],  "dt": cfg["dt"],
        "timesteps": cfg["timesteps"],
        "bc_north": cfg["bc"]["north"][1],
        "bc_south": cfg["bc"]["south"][1],
        "tol": cfg["tol"],
        "use_ser": args.use_ser,
        "pure_time": solver_pure.exec_time,
        "hybrid_time": solver_hybrid.exec_time,
        "pure_avg_iters": solver_pure.avg_gs_iters,
        "hybrid_avg_iters": solver_hybrid.avg_gs_iters,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Results saved to: {args.output_dir}/")

    from swiftcfd.machineLearning.post import run as post_run
    post_run(args.output_dir)


if __name__ == "__main__":
    main()
