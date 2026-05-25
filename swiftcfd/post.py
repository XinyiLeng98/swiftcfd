"""
Post-processing: comparison plots for the ML-hybrid solver.

Reads the numpy arrays and metadata written by hybrid_solver.py and produces:

    output/hybrid_results/
        contours.png          — side-by-side: Pure PDE | ML-Hybrid | Error
        residuals.png         — GS residual history for both methods
        performance_bar.png   — bar chart: avg GS iterations & wall time
        summary.png           — 2×2 panel with all of the above

Run from project root:
    python swiftcfd/machineLearning/post.py [--results-dir output/hybrid_results]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_results(results_dir: str):
    T_pure   = np.load(os.path.join(results_dir, "T_pure.npy"))
    T_hybrid = np.load(os.path.join(results_dir, "T_hybrid.npy"))
    res_pure   = np.load(os.path.join(results_dir, "res_pure.npy"))
    res_hybrid = np.load(os.path.join(results_dir, "res_hybrid.npy"))
    with open(os.path.join(results_dir, "meta.json")) as f:
        meta = json.load(f)
    return T_pure, T_hybrid, res_pure, res_hybrid, meta


# ── Figure 1: contour comparison ──────────────────────────────────────────────

def plot_contours(T_pure, T_hybrid, meta, save_path):
    nx, ny = meta["num_x"], meta["num_y"]
    Lx, Ly = meta["L_x"],   meta["L_y"]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    error     = np.abs(T_pure - T_hybrid)
    avg_err   = float(error.mean())
    max_err   = float(error.max())
    rel_err   = max_err / (np.abs(T_pure).max() + 1e-12) * 100

    vmin = min(T_pure.min(), T_hybrid.min())
    vmax = max(T_pure.max(), T_hybrid.max())
    levels = 30

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Pure PDE
    c1 = axes[0].contourf(X, Y, T_pure.T, levels=levels,
                           vmin=vmin, vmax=vmax, cmap="hot")
    fig.colorbar(c1, ax=axes[0], label="T")
    axes[0].set_title("Pure PDE (Gauss-Seidel)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    # ML-Hybrid
    c2 = axes[1].contourf(X, Y, T_hybrid.T, levels=levels,
                           vmin=vmin, vmax=vmax, cmap="hot")
    fig.colorbar(c2, ax=axes[1], label="T")
    axes[1].set_title("ML-Hybrid (ML + GS)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    # Absolute error
    c3 = axes[2].contourf(X, Y, error.T, levels=levels, cmap="Reds")
    fig.colorbar(c3, ax=axes[2], label="|ΔT|")
    axes[2].set_title(
        f"Absolute Error\n"
        f"avg={avg_err:.2e}  max={max_err:.2e}  ({rel_err:.2f}% of peak)",
        fontsize=11, fontweight="bold",
    )
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")

    fig.suptitle(
        f"heatedCavity — Pure vs ML-Hybrid"
        f"  (N_bc={meta['bc_north']}, steps={meta['timesteps']}, "
        f"α={meta['alpha']}, dt={meta['dt']})",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return avg_err, max_err


# ── Figure 2: residual history ─────────────────────────────────────────────────

def plot_residuals(res_pure, res_hybrid, save_path):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.semilogy(res_pure,   lw=1.2, color="#2196F3", label="Pure PDE (GS)")
    ax.semilogy(res_hybrid, lw=1.2, color="#F44336", label="ML-Hybrid (ML+GS)",
                alpha=0.85)

    ax.set_xlabel("Cumulative GS sub-iterations", fontsize=11)
    ax.set_ylabel("Normalised residual", fontsize=11)
    ax.set_title("GS residual history", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Figure 3: performance bar chart ───────────────────────────────────────────

def plot_performance(meta, save_path):
    methods = ["Pure PDE\n(GS)", "ML-Hybrid\n(ML + GS)"]
    iters   = [meta["pure_avg_iters"],  meta["hybrid_avg_iters"]]
    times   = [meta["pure_time"],        meta["hybrid_time"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = ["#2196F3", "#F44336"]

    # Iterations
    bars1 = ax1.bar(methods, iters, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars1, iters):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=11)
    ax1.set_ylabel("Average GS iterations per time step", fontsize=10)
    ax1.set_title("Iteration count", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(iters) * 1.2)
    ax1.grid(axis="y", ls="--", alpha=0.4)

    red_iters = (1 - iters[1] / max(iters[0], 1e-9)) * 100
    ax1.text(0.5, 0.95,
             f"Reduction: {red_iters:+.1f}%",
             ha="center", transform=ax1.transAxes, fontsize=10,
             color="darkgreen" if red_iters > 0 else "darkred")

    # Time
    bars2 = ax2.bar(methods, times, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{val:.1f} s", ha="center", va="bottom", fontsize=11)
    ax2.set_ylabel("Wall-clock time (seconds)", fontsize=10)
    ax2.set_title("Execution time", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(times) * 1.2)
    ax2.grid(axis="y", ls="--", alpha=0.4)

    speedup = times[0] / max(times[1], 1e-9)
    ax2.text(0.5, 0.95,
             f"Speedup: {speedup:.2f}×",
             ha="center", transform=ax2.transAxes, fontsize=10,
             color="darkgreen" if speedup > 1 else "darkred")

    fig.suptitle("Performance comparison — Pure PDE vs ML-Hybrid",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Figure 4: 2×2 summary panel ───────────────────────────────────────────────

def plot_summary(T_pure, T_hybrid, res_pure, res_hybrid, meta, save_path):
    nx, ny = meta["num_x"], meta["num_y"]
    Lx, Ly = meta["L_x"],   meta["L_y"]

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    error   = np.abs(T_pure - T_hybrid)
    vmin    = min(T_pure.min(), T_hybrid.min())
    vmax    = max(T_pure.max(), T_hybrid.max())
    levels  = 25

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # --- row 0: contour comparison ---
    ax_pure   = fig.add_subplot(gs[0, 0])
    ax_hybrid = fig.add_subplot(gs[0, 1])
    ax_err    = fig.add_subplot(gs[0, 2])

    for ax, T, title, cmap, vlo, vhi in [
        (ax_pure,   T_pure,   "Pure PDE (GS)",     "hot",  vmin, vmax),
        (ax_hybrid, T_hybrid, "ML-Hybrid (ML+GS)", "hot",  vmin, vmax),
        (ax_err,    error,    "|Pure − Hybrid|",   "Reds", None, None),
    ]:
        kw = {} if vlo is None else {"vmin": vlo, "vmax": vhi}
        cs = ax.contourf(X, Y, T.T, levels=levels, cmap=cmap, **kw)
        fig.colorbar(cs, ax=ax, shrink=0.82)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.set_aspect("equal")

    ax_err.set_title(
        f"|Pure − Hybrid|\nmax={error.max():.2e}  avg={error.mean():.2e}",
        fontsize=10, fontweight="bold",
    )

    # --- row 1, col 0-1: residual history (spans 2 columns) ---
    ax_res = fig.add_subplot(gs[1, :2])
    ax_res.semilogy(res_pure,   lw=1.2, color="#2196F3", label="Pure PDE (GS)")
    ax_res.semilogy(res_hybrid, lw=1.2, color="#F44336", label="ML-Hybrid",
                    alpha=0.85)
    ax_res.set_xlabel("Cumulative GS sub-iterations", fontsize=9)
    ax_res.set_ylabel("Normalised residual", fontsize=9)
    ax_res.set_title("GS residual history", fontsize=10, fontweight="bold")
    ax_res.legend(fontsize=9)
    ax_res.grid(True, which="both", ls="--", alpha=0.35)

    # --- row 1, col 2: performance text box ---
    ax_txt = fig.add_subplot(gs[1, 2])
    ax_txt.axis("off")

    iters   = [meta["pure_avg_iters"],  meta["hybrid_avg_iters"]]
    times   = [meta["pure_time"],        meta["hybrid_time"]]
    red_it  = (1 - iters[1] / max(iters[0], 1e-9)) * 100
    speedup = times[0] / max(times[1], 1e-9)
    max_err = float(error.max())
    avg_err = float(error.mean())

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
        f"Solution quality\n"
        f"  Max |ΔT|: {max_err:.2e}\n"
        f"  Avg |ΔT|: {avg_err:.2e}\n"
    )
    ax_txt.text(0.05, 0.95, summary_text, transform=ax_txt.transAxes,
                fontsize=9.5, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(
        f"heatedCavity — ML-Hybrid CFD Comparison\n"
        f"N_bc={meta['bc_north']}  steps={meta['timesteps']}  "
        f"α={meta['alpha']}  dt={meta['dt']}  tol={meta['tol']}",
        fontsize=12, fontweight="bold",
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def run(results_dir: str):
    """Generate all plots from a results directory. Called by hybrid_solver too."""
    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    print(f"\n{'='*65}")
    print(f"  heatedCavity post-processing")
    print(f"  Reading from: {results_dir}")
    print(f"{'='*65}\n")

    T_pure, T_hybrid, res_pure, res_hybrid, meta = load_results(results_dir)

    print(f"  Grid:     {meta['num_x']} × {meta['num_y']}")
    print(f"  Steps:    {meta['timesteps']}")
    print(f"  BC north: {meta['bc_north']}")
    print()

    plot_contours(T_pure, T_hybrid, meta,
                  os.path.join(results_dir, "contours.png"))
    plot_residuals(res_pure, res_hybrid,
                   os.path.join(results_dir, "residuals.png"))
    plot_performance(meta,
                     os.path.join(results_dir, "performance_bar.png"))
    plot_summary(T_pure, T_hybrid, res_pure, res_hybrid, meta,
                 os.path.join(results_dir, "summary.png"))

    print(f"\n{'='*65}")
    print(f"  All plots saved to: {results_dir}/")
    print(f"    contours.png        — Pure | Hybrid | Error fields")
    print(f"    residuals.png       — GS residual history")
    print(f"    performance_bar.png — iteration & time comparison")
    print(f"    summary.png         — 2×2 combined panel")
    print(f"{'='*65}\n")


def main():
    p = argparse.ArgumentParser(description="Post-processing plots for hybrid_solver results")
    p.add_argument("--results-dir", type=str,
                   default=os.path.join("output", "hybrid_results"),
                   help="Directory written by hybrid_solver.py")
    args = p.parse_args()
    run(args.results_dir)


if __name__ == "__main__":
    main()
