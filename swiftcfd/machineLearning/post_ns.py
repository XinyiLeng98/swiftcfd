import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _strip_ghost(a: np.ndarray) -> np.ndarray:
    # solver arrays are (nc+2)×(nc+2) with one ghost layer on each side
    return a[1:-1, 1:-1]


def load_results(results_dir: str):
    fields = {}
    for name in ("u_pure", "v_pure", "p_pure",
                 "u_hybrid", "v_hybrid", "p_hybrid"):
        fields[name] = _strip_ghost(np.load(os.path.join(results_dir, f"{name}.npy")))
    res_pure   = np.load(os.path.join(results_dir, "res_pure.npy"))
    res_hybrid = np.load(os.path.join(results_dir, "res_hybrid.npy"))
    with open(os.path.join(results_dir, "meta.json")) as f:
        meta = json.load(f)
    return fields, res_pure, res_hybrid, meta


def _coord_grids(meta):
    nc_x, nc_y = meta["nc_x"], meta["nc_y"]
    Lx,   Ly   = meta["L_x"],  meta["L_y"]
    dx,   dy   = meta["dx"],   meta["dy"]
    # cell-centred coordinates of interior cells
    x = np.linspace(0.5 * dx, Lx - 0.5 * dx, nc_x)
    y = np.linspace(0.5 * dy, Ly - 0.5 * dy, nc_y)
    return np.meshgrid(x, y)


# ── Figure 1-3: per-variable contour comparison ───────────────────────────────

def plot_field_comparison(field_pure, field_hybrid, var_name, meta, save_path):
    X, Y     = _coord_grids(meta)
    error    = np.abs(field_pure - field_hybrid)
    avg_err  = float(error.mean())
    max_err  = float(error.max())
    peak     = float(np.max(np.abs(field_pure)))
    rel_err  = max_err / (peak + 1e-12) * 100

    vmin     = min(field_pure.min(), field_hybrid.min())
    vmax     = max(field_pure.max(), field_hybrid.max())
    levels   = 30

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    c1 = axes[0].contourf(X, Y, field_pure.T, levels=levels,
                          vmin=vmin, vmax=vmax, cmap="viridis")
    fig.colorbar(c1, ax=axes[0], label=var_name)
    axes[0].set_title("Pure PDE (GS)", fontsize=13, fontweight="bold")

    c2 = axes[1].contourf(X, Y, field_hybrid.T, levels=levels,
                          vmin=vmin, vmax=vmax, cmap="viridis")
    fig.colorbar(c2, ax=axes[1], label=var_name)
    axes[1].set_title("ML-Hybrid (ML + GS)", fontsize=13, fontweight="bold")

    c3 = axes[2].contourf(X, Y, error.T, levels=levels, cmap="viridis")
    fig.colorbar(c3, ax=axes[2], label=f"|Δ{var_name}|")
    axes[2].set_title(
        f"Absolute Error\n"
        f"avg={avg_err:.2e}  max={max_err:.2e}  ({rel_err:.2f}% of peak)",
        fontsize=11, fontweight="bold",
    )

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")

    fig.suptitle(
        f"{var_name} — Pure vs ML-Hybrid"
        f"  (steps={meta['timesteps']}, ν={meta['nu']}, ρ={meta['rho']}, "
        f"dt={meta['dt']:.4g})",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Figure 4: residual history ─────────────────────────────────────────────────

def plot_residuals(res_pure, res_hybrid, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.semilogy(res_pure,   lw=1.2, color="#2196F3", label="Pure PDE (GS)")
    ax.semilogy(res_hybrid, lw=1.2, color="#F44336", label="ML-Hybrid (ML+GS)",
                alpha=0.85)

    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel("Final pressure-Poisson residual", fontsize=11)
    ax.set_title("Pressure-Poisson residual history", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Figure 5: performance bar chart ───────────────────────────────────────────

def plot_performance(meta, save_path):
    methods = ["Pure PDE\n(GS)", "ML-Hybrid\n(ML + GS)"]
    iters   = [meta["pure_avg_iters"],  meta["hybrid_avg_iters"]]
    times   = [meta["pure_time"],        meta["hybrid_time"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ["#2196F3", "#F44336"]

    bars1 = ax1.bar(methods, iters, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars1, iters):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=11)
    ax1.set_ylabel("Average GS iterations per time step", fontsize=10)
    ax1.set_title("Iteration count", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(iters) * 1.2 if max(iters) > 0 else 1)
    ax1.grid(axis="y", ls="--", alpha=0.4)

    red_iters = (1 - iters[1] / max(iters[0], 1e-9)) * 100
    ax1.text(0.5, 0.95, f"Reduction: {red_iters:+.1f}%",
             ha="center", transform=ax1.transAxes, fontsize=10,
             color="darkgreen" if red_iters > 0 else "darkred")

    bars2 = ax2.bar(methods, times, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{val:.2f} s", ha="center", va="bottom", fontsize=11)
    ax2.set_ylabel("Wall-clock time (seconds)", fontsize=10)
    ax2.set_title("Execution time", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(times) * 1.2 if max(times) > 0 else 1)
    ax2.grid(axis="y", ls="--", alpha=0.4)

    speedup = times[0] / max(times[1], 1e-9)
    ax2.text(0.5, 0.95, f"Speedup: {speedup:.2f}×",
             ha="center", transform=ax2.transAxes, fontsize=10,
             color="darkgreen" if speedup > 1 else "darkred")

    fig.suptitle("Performance — Pure PDE vs ML-Hybrid",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Figure 6: 2×3 summary panel ───────────────────────────────────────────────

def plot_summary(fields, res_pure, res_hybrid, meta, save_path):
    X, Y      = _coord_grids(meta)
    p_pure    = fields["p_pure"]
    p_hybrid  = fields["p_hybrid"]
    err       = np.abs(p_pure - p_hybrid)

    vmin   = min(p_pure.min(), p_hybrid.min())
    vmax   = max(p_pure.max(), p_hybrid.max())
    levels = 25

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    ax_pure   = fig.add_subplot(gs[0, 0])
    ax_hybrid = fig.add_subplot(gs[0, 1])
    ax_err    = fig.add_subplot(gs[0, 2])

    for ax, fld, title, vlo, vhi in [
        (ax_pure,   p_pure,   "p — Pure PDE",        vmin, vmax),
        (ax_hybrid, p_hybrid, "p — ML-Hybrid",       vmin, vmax),
        (ax_err,    err,      "|p_pure − p_hybrid|", None, None),
    ]:
        kw = {} if vlo is None else {"vmin": vlo, "vmax": vhi}
        cs = ax.contourf(X, Y, fld.T, levels=levels, cmap="viridis", **kw)
        fig.colorbar(cs, ax=ax, shrink=0.82)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y", fontsize=9)
        ax.set_aspect("equal")

    ax_err.set_title(
        f"|p_pure − p_hybrid|\nmax={err.max():.2e}  avg={err.mean():.2e}",
        fontsize=10, fontweight="bold",
    )

    ax_res = fig.add_subplot(gs[1, :2])
    ax_res.semilogy(res_pure,   lw=1.2, color="#2196F3", label="Pure PDE (GS)")
    ax_res.semilogy(res_hybrid, lw=1.2, color="#F44336", label="ML-Hybrid",
                    alpha=0.85)
    ax_res.set_xlabel("Time step", fontsize=9)
    ax_res.set_ylabel("Final p-Poisson residual", fontsize=9)
    ax_res.set_title("Residual history", fontsize=10, fontweight="bold")
    ax_res.legend(fontsize=9)
    ax_res.grid(True, which="both", ls="--", alpha=0.35)

    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")

    iters   = [meta["pure_avg_iters"],  meta["hybrid_avg_iters"]]
    times   = [meta["pure_time"],        meta["hybrid_time"]]
    red_it  = (1 - iters[1] / max(iters[0], 1e-9)) * 100
    speedup = times[0] / max(times[1], 1e-9)

    u_err = np.abs(fields["u_pure"] - fields["u_hybrid"])
    v_err = np.abs(fields["v_pure"] - fields["v_hybrid"])

    summary_text = (
        f"Performance summary\n"
        f"{'─'*28}\n"
        f"Pure PDE\n"
        f"  Avg GS iters : {iters[0]:.1f}\n"
        f"  Wall time    : {times[0]:.2f} s\n\n"
        f"ML-Hybrid\n"
        f"  Avg GS iters : {iters[1]:.1f}\n"
        f"  Wall time    : {times[1]:.2f} s\n\n"
        f"Improvement\n"
        f"  Iter reduction: {red_it:+.1f}%\n"
        f"  Speedup       : {speedup:.2f}×\n\n"
        f"Solution quality (max)\n"
        f"  |Δu|: {u_err.max():.2e}\n"
        f"  |Δv|: {v_err.max():.2e}\n"
        f"  |Δp|: {err.max():.2e}\n"
    )
    ax_txt.text(0.05, 0.95, summary_text, transform=ax_txt.transAxes,
                fontsize=9.5, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        f"NS ML-Hybrid CFD Comparison (pressure focus)\n"
        f"grid={meta['nc_x']}×{meta['nc_y']}  steps={meta['timesteps']}  "
        f"ν={meta['nu']}  ρ={meta['rho']}  dt={meta['dt']:.4g}  tol={meta['tol_p']}",
        fontsize=12, fontweight="bold",
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def run(results_dir: str):
    """Generate all plots from a results directory. Called by hybrid_solver_ns too."""
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    print(f"\n{'='*65}")
    print(f"  NS hybrid post-processing")
    print(f"  Reading from: {results_dir}")
    print(f"{'='*65}\n")

    fields, res_pure, res_hybrid, meta = load_results(results_dir)

    print(f"  Grid:  {meta['nc_x']} × {meta['nc_y']}")
    print(f"  Steps: {meta['timesteps']}")
    print(f"  ν/ρ:   {meta['nu']} / {meta['rho']}")
    print()

    plot_field_comparison(fields["u_pure"], fields["u_hybrid"], "u", meta,
                          os.path.join(results_dir, "contours_u.png"))
    plot_field_comparison(fields["v_pure"], fields["v_hybrid"], "v", meta,
                          os.path.join(results_dir, "contours_v.png"))
    plot_field_comparison(fields["p_pure"], fields["p_hybrid"], "p", meta,
                          os.path.join(results_dir, "contours_p.png"))
    plot_residuals(res_pure, res_hybrid,
                   os.path.join(results_dir, "residuals.png"))
    plot_performance(meta,
                     os.path.join(results_dir, "performance_bar.png"))
    plot_summary(fields, res_pure, res_hybrid, meta,
                 os.path.join(results_dir, "summary.png"))

    print(f"\n{'='*65}")
    print(f"  All plots saved to: {results_dir}/")
    print(f"    contours_u.png      — u (pure | hybrid | error)")
    print(f"    contours_v.png      — v (pure | hybrid | error)")
    print(f"    contours_p.png      — p (pure | hybrid | error)")
    print(f"    residuals.png       — pressure-Poisson residual history")
    print(f"    performance_bar.png — iteration & time comparison")
    print(f"    summary.png         — combined 2×3 panel")
    print(f"{'='*65}\n")


def main():
    p = argparse.ArgumentParser(
        description="Post-processing plots for hybrid_solver_ns results"
    )
    p.add_argument("--results-dir", type=str,
                   default=os.path.join("output", "hybrid_ns_results"),
                   help="Directory written by hybrid_solver_ns.py")
    args = p.parse_args()
    run(args.results_dir)


if __name__ == "__main__":
    main()
