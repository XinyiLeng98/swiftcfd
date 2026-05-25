"""Train a PINN model on heatedCavity training data.

Run (from project root, inside Docker):
    docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.model [options]

Options:
    --model       mlp | rnn | lstm | transformer  (default: mlp)
    --epochs      int   (default: 200)
    --batch-size  int   (default: 256)
    --lr          float (default: 1e-4)
    --patience    int   (default: 40)
    --output-dir  str   (default: output)
"""

import argparse

from swiftcfd.machineLearning.dataManager import DataManager
from swiftcfd.machineLearning.model.modelFactory import create_model


def main():
    p = argparse.ArgumentParser(description="Train PINN model for heatedCavity hybrid solver")
    p.add_argument("--model",      choices=["mlp", "rnn", "lstm", "transformer"],
                   default="mlp",  help="Neural network architecture")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--patience",   type=int,   default=40)
    p.add_argument("--output-dir", type=str,   default="output",
                   help="Directory to save model files (default: output)")
    args = p.parse_args()

    print(f"\n{'='*65}")
    print(f"  Training {args.model.upper()} — heatedCavity ML-hybrid")
    print(f"{'='*65}")

    training_data = DataManager.get_training_data("T", validation_percentage=0.2)
    input_size = training_data["T"]["x_train"].shape[1]

    model = create_model(
        args.model, "T", "T",
        input_size=input_size,
        hidden_size=256,
        output_size=5,
        num_layers=5,
    )

    model_path, norm_path, info = model.train_network(
        training_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*65}")
    print(f"  Training complete  (best val loss: {info['best_val_loss']:.6f})")
    print(f"  Model:         {model_path}")
    print(f"  Normalization: {norm_path}")
    print(f"\n  Next step:")
    print(f"    docker compose run --rm swiftcfd python3 -m swiftcfd.machineLearning.model \\")
    print(f"        --config input/generated/hc_val_01.toml \\")
    print(f"        --model  {model_path} \\")
    print(f"        --norm   {norm_path}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
