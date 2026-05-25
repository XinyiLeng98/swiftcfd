import torch
from torch import nn
from swiftcfd.machineLearning.model.modelBase import ModelBase

class RecurrentNeuralNetwork(ModelBase):
    def __init__(self, input_variables, output_variables, input_size=10, hidden_size=256, output_size=5,
                 num_layers=3, dropout=0.1):
        super().__init__(input_variables, output_variables, input_size, hidden_size, output_size,
                 num_layers, dropout)
        
        self.input_size = input_size
        self.output_size = output_size
        rnn_layers = 2
        self.rnn = nn.RNN(input_size=5, hidden_size=hidden_size,
                          num_layers=rnn_layers, batch_first=True,
                          nonlinearity='tanh',
                          dropout=dropout if rnn_layers > 1 else 0)
        head_input = hidden_size + 3
        head = []
        head.append(nn.Linear(head_input, hidden_size))
        head.append(nn.Tanh())
        head.append(nn.Dropout(dropout))
        for _ in range(min(num_layers - 1, 3)):
            head.append(nn.Linear(hidden_size, hidden_size))
            head.append(nn.Tanh())
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(hidden_size, output_size))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        batch = x.shape[0]
        s0 = torch.zeros(batch, 5, device=x.device)
        s0[:, 0] = x[:, 6]
        s1 = torch.zeros(batch, 5, device=x.device)
        s1[:, 0] = x[:, 5]
        s2 = x[:, 0:5]
        seq = torch.stack([s0, s1, s2], dim=1)
        rnn_out, _ = self.rnn(seq)
        combined = torch.cat([rnn_out[:, -1, :], x[:, 7:10]], dim=1)
        return self.head(combined)

    def __str__(self):
        return 'rnn'