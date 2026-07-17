"""
Residual-target trainer for the NS pressure warm-start model.

Approach #1 from the literature ("SIMPLE-PINN", p = p* + p').
---------------------------------------------------------------
The original NS model predicts the FULL pressure field p^{n+1} directly.  Two
problems with that:

  * The target has a large, case-dependent magnitude (offset + slope), so most
    of the network's capacity is spent reproducing the bulk field that we
    ALREADY KNOW from the previous step.
  * Minimising pointwise ||p_pred - p_true|| does NOT minimise the Poisson
    *residual* ||grad^2 p - rhs||, which is what actually sets the Krylov
    iteration count.

This trainer instead learns the CORRECTION (a.k.a. residual / increment)

        delta = p^{n+1} - p^n        (per cell, 5-point stencil)

and the warm start at inference time is reconstructed as

        p_guess = p^n + delta_pred .

Because delta is small and smooth, it is far easier to fit, the guess can never
be worse than the already-strong "previous step" baseline, and the learned
correction is exactly the velocity-pressure coupling term.

Everything else (data loading, normalisation, the MLP, device handling, early
stopping) is REUSED from the existing trainer -- we only (a) replace the target
with the delta and (b) tag the saved norm file with `residual_target=True` so
the v2 solver knows to add p^n back.

Run (Docker):
  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.train_residual_ns \
      --input-variables u,v,p --output-variable p \
      --epochs 200 --output-dir output/run_residual
"""

import argparse
import os

import numpy as np
import torch

from swiftcfd.machineLearning.dataManager import DataManager
from swiftcfd.machineLearning.model.modelFactory import create_model


# Per-variable feature layout produced by DataManager.get_training_data:
#   x (7 cols): [c^n, e^n, w^n, n^n, s^n, c^{n-1}, c^{n-2}]
#   y (5 cols): [c^{n+1}, e^{n+1}, w^{n+1}, n^{n+1}, s^{n+1}]
# The first 5 columns of x are the SAME 5 stencil points as y, one step earlier,
# so the delta target is simply  y - x[:, 0:5].
_P_NOW_COLS = slice(0, 5)


def build_residual_targets(training_data: dict, output_var: str) -> None:
    """In-place: replace y (= p^{n+1}) with the delta (= p^{n+1} - p^n)."""
    for x_split, y_split in (("x_train", "y_train"),
                             ("x_validation", "y_validation")):
        p_now = training_data[output_var][x_split][:, _P_NOW_COLS]
        training_data[output_var][y_split] = (
            training_data[output_var][y_split] - p_now
        ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Train an MLP to predict the pressure CORRECTION "
                    "delta = p^{n+1} - p^n (residual learning).")
    ap.add_argument("--input-variables", default="u,v,p",
                    help="Comma-separated inputs (default: u,v,p)")
    ap.add_argument("--output-variable", default="p")
    ap.add_argument("--model", default="mlp",
                    choices=["mlp", "rnn", "lstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.0)
    # Residual learning is a DATA target; physics residual (which assumes the
    # output is the full p, not a delta) would be inconsistent, so default it off.
    ap.add_argument("--weight-pde", type=float, default=0.0)
    ap.add_argument("--weight-data", type=float, default=1.0)
    ap.add_argument("--output-dir", default=os.path.join("output", "run_residual"))
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_vars = [v.strip() for v in args.input_variables.split(",")]
    output_var = args.output_variable.strip()

    print(f"\n{'='*65}")
    print(f"  RESIDUAL trainer  (target = p^(n+1) - p^n)")
    print(f"  inputs={args.input_variables}  output={output_var}  seed={args.seed}")
    print(f"  weight_pde={args.weight_pde} (0 = pure data on delta)")
    print(f"{'='*65}")

    all_vars = list(dict.fromkeys(input_vars + [output_var]))
    training_data = DataManager.get_training_data(",".join(all_vars),
                                                  validation_percentage=0.2)

    # --- the one real change: turn the target into a delta -------------------
    y_before = training_data[output_var]["y_train"].copy()
    build_residual_targets(training_data, output_var)
    y_after = training_data[output_var]["y_train"]
    print(f"  target std: full p={y_before.std():.4e}  ->  delta={y_after.std():.4e}  "
          f"(smaller delta = easier to learn)")

    features_per_var = training_data[input_vars[0]]["x_train"].shape[1]
    input_size = features_per_var * len(input_vars)
    output_size = training_data[output_var]["y_train"].shape[1]

    model = create_model(
        args.model,
        input_variables=args.input_variables,
        output_variables=output_var,
        equation_type="ns",
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=output_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model.train_network(
        training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_pde=args.weight_pde,
        weight_data=args.weight_data,
        patience=args.patience,
        output_dir=args.output_dir,
    )

    # --- tag the norm file so the solver reconstructs p_guess = p^n + delta ---
    norm_path = os.path.join(args.output_dir, f"norm_params_{model}.pth")
    params = torch.load(norm_path, weights_only=False)
    params["residual_target"] = True
    torch.save(params, norm_path)
    print(f"\n  Tagged {norm_path} with residual_target=True")
    print(f"  Model : {os.path.join(args.output_dir, f'pinn_model_{model}.pth')}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
