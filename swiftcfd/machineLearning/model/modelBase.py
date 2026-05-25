import os
import copy
import argparse

import torch
from abc import ABC, abstractmethod

class ModelBase(torch.nn.Module, ABC):
    def __init__(self, input_variables, output_variables, input_size=10, hidden_size=256, output_size=5,
                 num_layers=5, dropout=0.1):
        super().__init__()

        self.input_variables = input_variables
        self.output_variables = output_variables

    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def __str__(self):
        pass

    def unsteady_heat_diffusion_residual(self, X_batch_raw, Y_prediction_raw, training_parameters):
        # training_parameters: [N, 6] = [dx, dy, dt, rho, nu, alpha]
        dx    = training_parameters[:, 0:1]
        dy    = training_parameters[:, 1:2]
        dt    = training_parameters[:, 2:3]
        alpha = training_parameters[:, 5:6]

        # inputs
        phi_center = X_batch_raw[:, 0:1]
        phi_east   = X_batch_raw[:, 1:2]
        phi_west   = X_batch_raw[:, 2:3]
        phi_north  = X_batch_raw[:, 3:4]
        phi_south  = X_batch_raw[:, 4:5]
        phi_center_n_minus_1   = X_batch_raw[:, 5:6]
        phi_center_n_minus_2   = X_batch_raw[:, 6:7]

        # predictions
        phi_center_n_plus_1 = Y_prediction_raw[:, 0:1]
        phi_east_n_plus_1   = Y_prediction_raw[:, 1:2]
        phi_west_n_plus_1   = Y_prediction_raw[:, 2:3]
        phi_north_n_plus_1  = Y_prediction_raw[:, 3:4]
        phi_south_n_plus_1  = Y_prediction_raw[:, 4:5]

        # phi_t = (phi_center_n_plus_1 - phi_center_n_minus_1) / dt
        phi_t = 2 * (phi_center - phi_center_n_minus_1) / dt - (phi_center_n_minus_1 - phi_center_n_minus_2) / dt
        phi_xx_n = (phi_east - 2 * phi_center + phi_west) / (dx * dx)
        phi_yy_n = (phi_north - 2 * phi_center + phi_south) / (dy * dy)
        phi_xx_n_plus_1 = (phi_east_n_plus_1 - 2 * phi_center_n_plus_1 + phi_west_n_plus_1) / (dx * dx)
        phi_yy_n_plus_1 = (phi_north_n_plus_1 - 2 * phi_center_n_plus_1 + phi_south_n_plus_1) / (dy * dy)
        # residual = phi_t - alpha * (phi_xx_n_plus_1 + phi_yy_n_plus_1)

        residual = phi_t - (alpha/2) * (phi_xx_n + phi_yy_n + phi_xx_n_plus_1 + phi_yy_n_plus_1)
        return residual

    def loss_function(self, X_batch_raw, Y_pred_norm, Y_true_norm, Y_std, Y_mean, training_parameters,
                    weight_pde=1.0, weight_data=1.0):
        #physics loss
        Y_raw = Y_pred_norm * (Y_std + 1e-8) + Y_mean

        # residual based on unsteady heat-diffusion equation
        residual = self.unsteady_heat_diffusion_residual(X_batch_raw, Y_raw, training_parameters)

        phi_scale = torch.abs(X_batch_raw[:, 0:1]).mean() + 1e-8
        physics_loss = torch.mean((residual / phi_scale) ** 2)
        # data_loss
        data_loss = torch.mean((Y_pred_norm - Y_true_norm) ** 2)
        # total loss
        total_loss = weight_pde * physics_loss + weight_data * data_loss

        return total_loss, data_loss, physics_loss

    def train_network(self, training_data, epochs=200, batch_size=256, lr=1e-4, hidden_size=256,
              num_layers=5, dropout=0.1, output_dir="output", patience=300):

        # TODO: currently only works if input and output variables are the same (temperature)
        # needs to be adjsusted for Navier-Stokes where input variables are u,v,p and output is p
        assert self.input_variables == 'T'
        assert self.output_variables == 'T'

        # TODO: same here, hardcoding T as the variable to use, need to be generalised later
        X_train = training_data['T']['x_train']
        Y_train = training_data['T']['y_train']
        X_val   = training_data['T']['x_validation']
        Y_val   = training_data['T']['y_validation']
        training_parameters = training_data['T']['training_parameters']
        validation_parameters = training_data['T']['validation_parameters']
            
        input_size = X_train.shape[1]      

        # --- Normalization statistics ---
        X_mean = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
        X_std  = torch.tensor(X_train.std(axis=0),  dtype=torch.float32)
        Y_mean = torch.tensor(Y_train.mean(axis=0), dtype=torch.float32)
        Y_std  = torch.tensor(Y_train.std(axis=0),  dtype=torch.float32)

        print(f"\nNormalization stats computed  (input_size={input_size})")

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
        print(f"  Input/Output: {input_size} → 5")
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
        print(f"  (physics loss weight=1.0, data loss weight=1.0)")
        print(f"  (LR: CosineAnnealingWarmRestarts, T_0=30, T_mult=2, lr={lr})")
        print("=" * 60)

        print(f"X_train shape: {X_train.shape}")   # expect (N, 10) if model needs 10
        print(f"X_mean shape:  {X_mean.shape}")

        # Check the first layer's expected input size
        first_layer = next(m for m in self.modules() if isinstance(m, torch.nn.Linear))
        print(f"First Linear layer weight shape: {first_layer.weight.shape}")
        # weight is (out_features, in_features), so .shape[1] is what it expects

        for epoch in range(epochs):
            self.train()   # re-enable dropout after each validation pass
            ep_total = ep_data = ep_phys = 0.0
            for Xb, Yb, Tp in train_loader:
                Xn = (Xb - X_mean)/(X_std + 1e-8)
                Yn = (Yb - Y_mean)/(Y_std + 1e-8)
                
                pred = self(Xn)
                loss, ld, lp = self.loss_function(Xb, pred, Yn, Y_std, Y_mean, Tp)
                
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
                    Xn = (Xb - X_mean) / (X_std + 1e-8)
                    Yn = (Yb - Y_mean) / (Y_std + 1e-8)
                    pred = self(Xn)
                    l, _, _ = self.loss_function(Xb, pred, Yn, Y_std, Y_mean, Vp)
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

        torch.save(self.state_dict(), model_path)
        torch.save({"X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std,
                    "input_size": input_size, "hidden_size": hidden_size,
                    "num_layers": num_layers, "model_type": str(self)}, norm_path)

        print(f"Model saved to:          {model_path}")
        print(f"Norm params saved to:    {norm_path}")
        print(f"  Best val    : {best_val_loss:.6f}  ({len(train_history)} epochs)")

        return model_path, norm_path, {
            "model_type": str(self), "total_params": total_params,
            "epochs_trained": len(train_history), "best_val_loss": best_val_loss,
            "final_train_loss": avg_t, "final_physics_loss": avg_p,
            "final_data_loss": avg_d, "hidden_size": hidden_size,
            "num_layers": num_layers, "lr": lr, "batch_size": batch_size,
            "patience": patience, "dropout": dropout,
            "history": train_history,
        }


def main():
    # Imports here to avoid circular dependency: modelBase → modelFactory → model files → modelBase
    from swiftcfd.machineLearning.dataManager import DataManager
    from swiftcfd.machineLearning.model.modelFactory import create_model

    p = argparse.ArgumentParser(description="Train a PINN model for heatedCavity hybrid solver")
    p.add_argument("--model",      choices=["mlp", "rnn", "lstm", "transformer"], default="mlp")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=40)
    p.add_argument("--output-dir", type=str,   default="output")
    args = p.parse_args()

    print(f"\n{'='*65}")
    print(f"  Training {args.model.upper()} — heatedCavity ML-hybrid")
    print(f"{'='*65}")

    training_data = DataManager.get_training_data("T", validation_percentage=0.2)
    input_size    = training_data["T"]["x_train"].shape[1]

    model = create_model(
        args.model, "T", "T",
        input_size=input_size,
        hidden_size=256,
        output_size=5,
        num_layers=5,
    )

    model_path, norm_path, _ = model.train_network(
        training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*65}")
    print(f"  Training complete")
    print(f"  Model : {model_path}")
    print(f"  Norm  : {norm_path}")
    print(f"\n  Example validation run:")
    print(f"    python3 -m swiftcfd.machineLearning.hybrid_solver \\")
    print(f"        --config input/generated/val_01.toml \\")
    print(f"        --model  {model_path} \\")
    print(f"        --norm   {norm_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
