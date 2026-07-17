"""
G1 figure: iterations-vs-preconditioner from one or more sweep.csv files.

Reads the `sweep.csv` written by `hybrid_solver_ns_petsc_v2.py` (columns:
preconditioner, init, avg_iters, std_iters, n_steps) and produces:

  * a grouped bar chart (one group per preconditioner, bars = init strategies),
  * if several CSVs are passed, the `ml` bars are drawn per model so you can
    compare e.g. data-only vs residual vs rnn warm starts side by side.

Run (Docker):
  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.plot_sweep \
      --csv data-only=output/sweep_dataonly/sweep.csv \
            residual=output/sweep_residual/sweep.csv \
      --out output/sweep_compare.png
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    """-> dict[(pc, init)] = (avg, std), preserving pc order of first appearance."""
    data, order = {}, []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            pc, init = row["preconditioner"], row["init"]
            if pc not in order:
                order.append(pc)
            if row["avg_iters"] != "":
                data[(pc, init)] = (float(row["avg_iters"]),
                                    float(row["std_iters"]) if row["std_iters"] else 0.0)
    return data, order


def main():
    ap = argparse.ArgumentParser(description="Plot iterations-vs-preconditioner (G1).")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="label=path entries, e.g. residual=output/sweep_residual/sweep.csv")
    ap.add_argument("--out", default="output/sweep_compare.png")
    args = ap.parse_args()

    runs = {}
    pc_order = None
    for entry in args.csv:
        if "=" not in entry:
            ap.error(f"--csv entries must be label=path, got '{entry}'")
        label, path = entry.split("=", 1)
        data, order = load_csv(path)
        runs[label] = data
        if pc_order is None:
            pc_order = order

    # `zero` and `prev` are model-independent -> take them from the first run.
    first = next(iter(runs.values()))
    series = []   # (name, color, value_fn)
    series.append(("zero", "#bbbbbb", lambda pc, run: first.get((pc, "zero"), (np.nan, 0))))
    series.append(("prev", "#4477aa", lambda pc, run: first.get((pc, "prev"), (np.nan, 0))))
    ml_colors = ["#ee6677", "#228833", "#ccbb44", "#aa3377"]
    for k, label in enumerate(runs):
        series.append((f"ml:{label}", ml_colors[k % len(ml_colors)],
                       (lambda lbl: (lambda pc, run: runs[lbl].get((pc, "ml"), (np.nan, 0))))(label)))

    n_groups = len(pc_order)
    n_bars = len(series)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(1.6 * n_groups + 3, 5))
    for b, (name, color, fn) in enumerate(series):
        vals = [fn(pc, None)[0] for pc in pc_order]
        errs = [fn(pc, None)[1] for pc in pc_order]
        ax.bar(x + (b - (n_bars - 1) / 2) * width, vals, width,
               yerr=errs, capsize=2, label=name, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(pc_order)
    ax.set_xlabel("preconditioner (left=weak  ->  right=strong)")
    ax.set_ylabel("avg KSP iterations / step  (lower = better)")
    ax.set_title("Pressure-Poisson warm-start: ML vs prev-step, by preconditioner")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"  Saved figure: {args.out}")

    # also print the "ML vs prev" reduction table to stdout
    print(f"\n  {'pc':<10}{'prev':>8}" + "".join(f"{('ml:'+l):>16}" for l in runs))
    for pc in pc_order:
        prev = first.get((pc, "prev"), (np.nan, 0))[0]
        row = f"  {pc:<10}{prev:>8.1f}"
        for l in runs:
            ml = runs[l].get((pc, "ml"), (np.nan, 0))[0]
            red = (1 - ml / prev) * 100 if prev else float("nan")
            row += f"{ml:>9.1f}({red:>+5.1f}%)"
        print(row)


if __name__ == "__main__":
    main()
