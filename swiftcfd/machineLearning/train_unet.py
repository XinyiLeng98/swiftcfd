"""
Train the whole-field U-Net pressure warm-start model (G3), with an optional
SIMPLE-PINN-style pressure-Poisson residual loss.

  input  X = [u^n, v^n, p^n, rhs]   (N, 4, H, W)
  target Y = delta = p^{n+1} - p^n  (N, 1, H, W)   [default; residual learning]

Two loss terms:
  * data loss      : MSE on the (normalised) target field          -- pointwise accuracy
  * residual loss  : ‖∇²(p^n + delta_pred) - rhs‖² / ‖rhs‖²         -- SIMPLE-PINN physics

The residual loss is the discrete pressure-correction (Poisson) equation from the
SIMPLE algorithm:  ∇²p = (ρ/dt)∇·u* = rhs.  Minimising it directly targets the
quantity that sets the Krylov iteration count, which pointwise MSE does NOT.
Enable it with --weight-residual > 0 (default 0 reproduces the old MSE-only run).

Run (Docker):
  # MSE-only (old behaviour)
  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.train_unet \
      --data-dir output/field_data --epochs 100 --output-dir output/run_unet

  # SIMPLE-PINN residual loss
  docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.train_unet \
      --data-dir output/field_data --epochs 150 \
      --weight-data 1.0 --weight-residual 1.0 \
      --output-dir output/run_unet_simplepinn
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from swiftcfd.machineLearning.model.uNet import UNet


def load_fields(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "fields_*.npz")))
    if not files:
        raise FileNotFoundError(f"No fields_*.npz in {data_dir} "
                                f"(run gen_field_data.py first).")
    U, V, P, RHS, DELTA, PNEXT = [], [], [], [], [], []
    dxs, dys = [], []
    for f in files:
        d = np.load(f)
        U.append(d["u"]); V.append(d["v"]); P.append(d["p"])
        RHS.append(d["rhs"]); DELTA.append(d["delta"]); PNEXT.append(d["p_next"])
        dxs.append(float(d["dx"])); dys.append(float(d["dy"]))
    cat = lambda lst: np.concatenate(lst, axis=0)
    dx, dy = float(np.mean(dxs)), float(np.mean(dys))
    if np.std(dxs) > 1e-9 or np.std(dys) > 1e-9:
        print(f"  WARNING: cases have different dx/dy; residual loss uses the mean "
              f"(dx={dx:.5f}, dy={dy:.5f}). Keep grids uniform for a clean physics loss.")
    print(f"  loaded {len(files)} case(s), {cat(P).shape[0]} snapshots, "
          f"grid {cat(P).shape[1]}x{cat(P).shape[2]}, dx={dx:.5f} dy={dy:.5f}")
    return cat(U), cat(V), cat(P), cat(RHS), cat(DELTA), cat(PNEXT), dx, dy


def laplacian_kernel(dx, dy, device):
    """5-point anisotropic Laplacian as a (1,1,3,3) conv kernel.
    Array axis0=H=x (spacing dx), axis1=W=y (spacing dy)."""
    k = torch.zeros(1, 1, 3, 3, device=device)
    k[0, 0, 0, 1] = k[0, 0, 2, 1] = 1.0 / dx ** 2     # x-neighbours
    k[0, 0, 1, 0] = k[0, 0, 1, 2] = 1.0 / dy ** 2     # y-neighbours
    k[0, 0, 1, 1] = -2.0 / dx ** 2 - 2.0 / dy ** 2
    return k


def poisson_residual_loss(pred_norm, Xb_raw, Ym, Ys, kernel, residual_target):
    """‖∇²p_guess - rhs‖² / ‖rhs‖²  on the strict interior (valid conv region)."""
    denorm = pred_norm * Ys + Ym                    # back to physical units
    p_now = Xb_raw[:, 2:3]                           # raw p^n channel
    rhs = Xb_raw[:, 3:4]                             # raw rhs channel
    p_guess = (p_now + denorm) if residual_target else denorm
    lap = F.conv2d(p_guess, kernel)                 # (B,1,H-2,W-2)
    rhs_in = rhs[:, :, 1:-1, 1:-1]
    return ((lap - rhs_in) ** 2).mean() / ((rhs_in ** 2).mean() + 1e-12)


def main():
    ap = argparse.ArgumentParser(description="Train the whole-field U-Net (G3) "
                                             "with optional SIMPLE-PINN residual loss.")
    ap.add_argument("--data-dir", default="output/field_data")
    ap.add_argument("--target", choices=["delta", "p_next"], default="delta",
                    help="delta = residual learning (default); p_next = full field")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--weight-data", type=float, default=1.0)
    ap.add_argument("--weight-residual", type=float, default=0.0,
                    help="SIMPLE-PINN pressure-Poisson residual weight (0 = MSE only)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="output/run_unet")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    u, v, p, rhs, delta, p_next, dx, dy = load_fields(args.data_dir)
    X = np.stack([u, v, p, rhs], axis=1).astype(np.float32)      # (N,4,H,W)
    Y = (delta if args.target == "delta" else p_next)[:, None].astype(np.float32)
    residual_target = args.target == "delta"

    # per-channel normalisation (mean/std over N,H,W)
    Xm = X.mean(axis=(0, 2, 3), keepdims=True)
    Xs = X.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    Ym = float(Y.mean()); Ys = float(Y.std()) + 1e-8

    n = X.shape[0]
    idx = np.random.permutation(n)
    n_val = int(args.val_frac * n)
    tr, va = idx[n_val:], idx[:n_val]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}  target={args.target}  "
          f"X{tuple(X.shape)} -> Y{tuple(Y.shape)}  "
          f"w_data={args.weight_data} w_res={args.weight_residual}")

    # keep RAW X,Y in the loaders; normalise inside the loop (needed for physics loss)
    Xm_t = torch.tensor(Xm, device=device)
    Xs_t = torch.tensor(Xs, device=device)
    kernel = laplacian_kernel(dx, dy, device)

    def loader(ids, shuffle):
        ds = torch.utils.data.TensorDataset(torch.tensor(X[ids]), torch.tensor(Y[ids]))
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader, val_loader = loader(tr, True), loader(va, False)

    model = UNet(in_channels=4, out_channels=1, base=args.base, levels=args.levels).to(device)
    print(f"  U-Net params: {sum(pp.numel() for pp in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def step_losses(Xb_raw, Yb_raw):
        Xn = (Xb_raw - Xm_t) / Xs_t
        Yn = (Yb_raw - Ym) / Ys
        pred = model(Xn)
        l_data = F.mse_loss(pred, Yn)
        l_res = (poisson_residual_loss(pred, Xb_raw, Ym, Ys, kernel, residual_target)
                 if args.weight_residual > 0 else torch.zeros((), device=device))
        total = args.weight_data * l_data + args.weight_residual * l_res
        return total, l_data, l_res

    best, best_state = float("inf"), None
    for ep in range(args.epochs):
        model.train()
        for Xb, Yb in train_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            opt.zero_grad()
            total, _, _ = step_losses(Xb, Yb)
            total.backward(); opt.step()
        sched.step()

        model.eval(); vt = vd = vr = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb, Yb = Xb.to(device), Yb.to(device)
                t, d_, r_ = step_losses(Xb, Yb)
                vt += t.item(); vd += d_.item(); vr += r_.item()
        nb = max(len(val_loader), 1)
        vt, vd, vr = vt / nb, vd / nb, vr / nb
        if vt < best:
            best, best_state = vt, {k: t.cpu().clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}/{args.epochs}  val_total={vt:.4e}  "
                  f"data={vd:.4e}  residual={vr:.4e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(args.output_dir, exist_ok=True)
    mpath = os.path.join(args.output_dir, "unet_model.pth")
    npath = os.path.join(args.output_dir, "unet_norm.pth")
    torch.save(model.cpu().state_dict(), mpath)
    torch.save({
        "X_mean": torch.tensor(Xm), "X_std": torch.tensor(Xs),
        "Y_mean": Ym, "Y_std": Ys,
        "in_channels": 4, "channels": ["u", "v", "p", "rhs"],
        "target": args.target, "residual_target": residual_target,
        "base": args.base, "levels": args.levels,
        "weight_data": args.weight_data, "weight_residual": args.weight_residual,
        "dx": dx, "dy": dy,
    }, npath)
    print(f"\n  best val_total={best:.4e}")
    print(f"  saved {mpath}\n  saved {npath}")


if __name__ == "__main__":
    main()
