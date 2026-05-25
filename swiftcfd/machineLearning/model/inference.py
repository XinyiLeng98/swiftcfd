import torch
from swiftcfd.machineLearning.model.modelFactory import create_model

class PINNInference:
    def __init__(self, model_path, normalization_path):
        # Load normalization parameters
        params = torch.load(normalization_path, weights_only=False)
        self.X_mean = params["X_mean"]
        self.X_std  = params["X_std"]
        self.Y_mean = params["Y_mean"]
        self.Y_std  = params["Y_std"]

        input_size = params.get("input_size", len(self.X_mean))
        self.input_size = input_size
        hidden_size = params.get("hidden_size", 256)
        num_layers = params.get("num_layers", 5)
        model_type = params.get("model_type", "mlp")
        # dropout=0 at inference — dropout layers become identity
        self.model = create_model(model_type, input_size, hidden_size, 5, num_layers, dropout=0.0)
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model.eval()
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.total_params = sum(p.numel() for p in self.model.parameters())
        print(f"PINNInference loaded  (type={model_type}, input={input_size}, hidden={hidden_size}x{num_layers})")

    def predict(self, input_data):
        squeeze = False
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            squeeze = True

        with torch.no_grad():
            X = torch.tensor(input_data, dtype=torch.float32)
            Xn = (X - self.X_mean) / (self.X_std + 1e-8)
            Yn = self.model(Xn)
            Y = Yn * (self.Y_std + 1e-8) + self.Y_mean

        out = Y.numpy()
        return out[0] if squeeze else out
