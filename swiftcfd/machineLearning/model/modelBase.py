import os
import copy
import argparse

import numpy as np
import torch
from abc import ABC, abstractmethod

from swiftcfd.machineLearning.model.physics import HeatResidual, NSPressureResidual


class ModelBase(torch.nn.Module, ABC):
    EQUATION_TYPES = ('heat', 'ns')

    def __init__(self, input_variables, output_variables, equation_type='heat',
                 input_size=10, hidden_size=256, output_size=5,
                 num_layers=5, dropout=0.1):
        super().__init__()

        if equation_type not in self.EQUATION_TYPES:
            raise ValueError(
                f"equation_type must be one of {self.EQUATION_TYPES}, got '{equation_type}'"
            )

        self.input_variables  = input_variables
        self.output_variables = output_variables
        self.equation_type    = equation_type
        self.input_size       = input_size
        self.hidden_size      = hidden_size
        self.output_size      = output_size
        self.num_layers       = num_layers
        self.dropout          = dropout

    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

    # -------------------------------------------------------------------------
    # Physics-informed loss
    # -------------------------------------------------------------------------

    def loss_function(self, X_raw, Y_pred_norm, Y_true_norm, Y_std, Y_mean,
                      params, weight_pde=1.0, weight_data=1.0):

        Y_raw = Y_pred_norm * (Y_std + 1e-8) + Y_mean

        if self.equation_type == 'heat':
            residual  = HeatResidual.compute(X_raw, Y_raw, params)
            phi_scale = torch.abs(X_raw[:, 0:1]).mean() + 1e-8
        elif self.equation_type == 'ns':
            residual  = NSPressureResidual.compute(X_raw, Y_raw, params)
            phi_scale = torch.abs(X_raw[:, 14:15]).mean() + 1e-8

        physics_loss = torch.mean((residual / phi_scale) ** 2)
        data_loss    = torch.mean((Y_pred_norm - Y_true_norm) ** 2)
        total_loss   = weight_pde * physics_loss + weight_data * data_loss

        return total_loss, data_loss, physics_loss

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train_network(self, training_data, epochs=200, batch_size=256, lr=1e-4,
                      hidden_size=256, num_layers=5, dropout=0.1,
                      weight_pde=1.0, weight_data=1.0,
                      output_dir="output", patience=300):
        # TODO: currently only works if input and output variables are the same (temperature)
        # needs to be adjsusted for Navier-Stokes where input variables are u,v,p and output is p
        input_vars = [v.strip() for v in self.input_variables.split(',')]
        output_var = self.output_variables.strip()
        # TODO: same here, hardcoding T as the variable to use, need to be generalised later
        # Validate that all required variables are present
        for v in input_vars:
            if v not in training_data:
                raise KeyError(
                    f"Input variable '{v}' not found in training_data. "
                    f"Available: {list(training_data.keys())}"
                )
        if output_var not in training_data:
            raise KeyError(
                f"Output variable '{output_var}' not found in training_data. "
                f"Available: {list(training_data.keys())}"
            )

        X_train = np.concatenate([training_data[v]['x_train']      for v in input_vars], axis=1)
        X_val   = np.concatenate([training_data[v]['x_validation'] for v in input_vars], axis=1)

        # Output and simulation parameters come from the target variable
        Y_train               = training_data[output_var]['y_train']
        Y_val                 = training_data[output_var]['y_validation']
        training_parameters       = training_data[output_var]['training_parameters']
        validation_parameters     = training_data[output_var]['validation_parameters']
        
        input_size = X_train.shape[1]

        # --- Device (GPU if available) ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # --- Normalization statistics (kept on the training device) ---
        X_mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float32, device=device)
        X_std  = torch.tensor(X_train.std(axis=0),  dtype=torch.float32, device=device)
        Y_mean = torch.tensor(Y_train.mean(axis=0), dtype=torch.float32, device=device)
        Y_std  = torch.tensor(Y_train.std(axis=0),  dtype=torch.float32, device=device)

        print(f"\nNormalization stats computed  (input_size={input_size})")
        print(f"  Device: {device}"
              + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.float32),
            torch.tensor(training_parameters, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.float32),
            torch.tensor(validation_parameters, dtype=torch.float32)),
            batch_size=batch_size, shuffle=False)

        print(f"  Training:   {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Architecture: {self}")
        print(f"  Model: {input_size} -> {hidden_size}x{num_layers} -> 5  ({total_params:,} params)")
        print(f"\n  Model       : {self}  ({total_params:,} params)")
        print(f"  Equation    : {self.equation_type}")
        print(f"  Variables   : input={self.input_variables}  output={self.output_variables}")
        print(f"  Input/Output: {input_size} → {Y_train.shape[1]}")
        print(f"  Train/Val   : {len(X_train)} / {len(X_val)} samples")
        print(f"  Epochs/LR   : {epochs} / {lr}  (patience={patience})")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        train_history = []

        # Training loop 
        print(f"\nTraining for up to {epochs} epochs ...")
        print(f"  (physics loss weight={weight_pde}, data loss weight={weight_data})")
        print(f"  (LR: CosineAnnealingLR, T_max={epochs}, lr={lr})")
        print("=" * 60)

        for epoch in range(epochs):
            self.train()   # re-enable dropout after each validation pass
            ep_total = ep_data = ep_phys = 0.0

            for Xb, Yb, Tp in train_loader:
                Xb, Yb, Tp = Xb.to(device), Yb.to(device), Tp.to(device)
                Xn = (Xb - X_mean)/(X_std + 1e-8)
                Yn = (Yb - Y_mean)/(Y_std + 1e-8)
                
                pred = self(Xn)
                loss, ld, lp = self.loss_function(Xb, pred, Yn, Y_std, Y_mean, Tp,
                                                  weight_pde=weight_pde, weight_data=weight_data)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                ep_total += loss.item(); ep_data += ld.item(); ep_phys += lp.item()

            n = len(train_loader)
            avg_t, avg_d, avg_p = ep_total/n, ep_data/n, ep_phys/n
            scheduler.step()

            # Validate
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, Yb, Vp in val_loader:
                    Xb, Yb, Vp = Xb.to(device), Yb.to(device), Vp.to(device)
                    Xn = (Xb - X_mean) / (X_std + 1e-8)
                    Yn = (Yb - Y_mean) / (Y_std + 1e-8)
                    pred = self(Xn)
                    l, _, _ = self.loss_function(Xb, pred, Yn, Y_std, Y_mean, Vp,
                                                 weight_pde=weight_pde, weight_data=weight_data)
                    val_loss += l.item()
            val_loss /= len(val_loader)

            cur_lr = optimizer.param_groups[0]['lr']
            train_history.append({
                'epoch': epoch, 'total': avg_t, 'data': avg_d,
                'physics': avg_p, 'val': val_loss, 'lr': cur_lr
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.state_dict())
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:>4d}/{epochs}, Total = {avg_t:>8.2e}, "
                    f"Physics = {avg_p:>8.2e}, Data = {avg_d:>8.2e}, "
                    f"val = {val_loss:>8.2e}, lr = {cur_lr:>8.2e}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}  (best val={best_val_loss:.6f})")
                break

        print("=" * 60)
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final training loss: {avg_t:.6f}  (Physics: {avg_p:.6f}, Data: {avg_d:.6f})")

        # save model
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"{self}"
        model_path = os.path.join(output_dir, f"pinn_model_{suffix}.pth")
        norm_path = os.path.join(output_dir, f"norm_params_{suffix}.pth")

        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print(f"  Restored best model (val_loss={best_val_loss:.6f})")

        # Move back to CPU before saving so the checkpoint loads on any machine
        # (training may run on GPU, but inference runs locally on CPU).
        self.cpu()
        X_mean, X_std = X_mean.cpu(), X_std.cpu()
        Y_mean, Y_std = Y_mean.cpu(), Y_std.cpu()

        torch.save(self.state_dict(), model_path)
        torch.save(
            {
                "X_mean": X_mean, "X_std": X_std,
                "Y_mean": Y_mean, "Y_std": Y_std,
                "input_size":       input_size,
                "hidden_size":      hidden_size,
                "num_layers":       num_layers,
                "model_type":       str(self),
                "equation_type":    self.equation_type,
                "input_variables":  self.input_variables,
                "output_variables": self.output_variables,
            },
            norm_path,
        )

        print(f"Model saved to:          {model_path}")
        print(f"Norm params saved to:    {norm_path}")
        print(f"  Best val    : {best_val_loss:.6f}  ({len(train_history)} epochs)")

        return model_path, norm_path, {
            "model_type":         str(self),
            "equation_type":      self.equation_type,
            "input_variables":    self.input_variables,
            "output_variables":   self.output_variables,
            "total_params":       total_params,
            "epochs_trained":     len(train_history),
            "best_val_loss":      best_val_loss,
            "final_train_loss":   avg_t,
            "final_physics_loss": avg_p,
            "final_data_loss":    avg_d,
            "hidden_size":        hidden_size,
            "num_layers":         num_layers,
            "lr":                 lr,
            "batch_size":         batch_size,
            "patience":           patience,
            "dropout":            dropout,
            "weight_pde":         weight_pde,
            "weight_data":        weight_data,
            "history":            train_history,
        }


def main():
    # Imports here to avoid circular dependency: modelBase → modelFactory → model files → modelBase
    from swiftcfd.machineLearning.dataManager import DataManager
    from swiftcfd.machineLearning.model.modelFactory import create_model

    p = argparse.ArgumentParser(description="Train a PINN model for heat or NS equations")
    p.add_argument("--model",            choices=["mlp", "rnn", "lstm", "transformer"], default="mlp")
    p.add_argument("--equation-type",    choices=["heat", "ns"], default="heat",
                   help="PDE type: 'heat' (default) or 'ns' (Navier-Stokes)")
    p.add_argument("--input-variables",  type=str, default="T",
                   help="Comma-separated input variable names, e.g. 'T' or 'u,v,p'")
    p.add_argument("--output-variable",  type=str, default="T",
                   help="Single output variable name, e.g. 'T' or 'p'")
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--batch-size",       type=int,   default=256)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--patience",         type=int,   default=40)
    p.add_argument("--seed",             type=int,   default=0)
    p.add_argument("--hidden-size",      type=int,   default=256)
    p.add_argument("--num-layers",       type=int,   default=5)
    p.add_argument("--dropout",          type=float, default=0.1)
    p.add_argument("--weight-pde",       type=float, default=1.0)
    p.add_argument("--weight-data",      type=float, default=1.0)
    p.add_argument("--output-dir",       type=str,   default="output")
    args = p.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"  Seed: {args.seed}")

    input_vars = [v.strip() for v in args.input_variables.split(',')]
    output_var = args.output_variable.strip()

    print(f"\n{'='*65}")
    print(f"  Training {args.model.upper()} — equation: {args.equation_type}")
    print(f"  Input variables : {args.input_variables}")
    print(f"  Output variable : {output_var}")
    print(f"{'='*65}")

    # Load all unique variables (inputs + output) with a consistent random split
    all_vars = list(dict.fromkeys(input_vars + [output_var]))
    training_data = DataManager.get_training_data(','.join(all_vars), validation_percentage=0.2)

    # Input size = 7 features per variable × number of input variables
    features_per_var = training_data[input_vars[0]]['x_train'].shape[1]
    input_size  = features_per_var * len(input_vars)
    output_size = training_data[output_var]['y_train'].shape[1]

    model = create_model(
        args.model,
        input_variables=args.input_variables,
        output_variables=output_var,
        equation_type=args.equation_type,
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=output_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model_path, norm_path, _ = model.train_network(
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

    print(f"\n{'='*65}")
    print(f"  Training complete")
    print(f"  Model : {model_path}")
    print(f"  Norm  : {norm_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
