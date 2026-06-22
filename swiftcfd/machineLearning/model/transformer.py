import torch
from torch import nn
from swiftcfd.machineLearning.model.modelBase import ModelBase

class Transformer(ModelBase):
    def __init__(self, input_variables, output_variables, equation_type='heat',
                 input_size=10, hidden_size=256, output_size=5, num_layers=5, dropout=0.1):
        super().__init__(input_variables, output_variables, equation_type,
                         input_size, hidden_size, output_size, num_layers, dropout)
        
        self.input_size = input_size
        self.output_size = output_size
        d_model = 64
        self.token_embed = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_size, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=hidden_size,
            dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=min(num_layers, 4))
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_size), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size))

    def forward(self, x):
        tokens = self.token_embed(x.unsqueeze(-1)) + self.pos_encoding
        encoded = self.transformer(tokens)
        return self.head(encoded.mean(dim=1))
    
    def __str__(self):
        return "transformer"
