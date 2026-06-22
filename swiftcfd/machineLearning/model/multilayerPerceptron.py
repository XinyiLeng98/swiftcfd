from torch import nn
from swiftcfd.machineLearning.model.modelBase import ModelBase

class MultiLayerPerceptron(ModelBase):
    def __init__(self, input_variables, output_variables, equation_type='heat',
                 input_size=10, hidden_size=256, output_size=5, num_layers=5, dropout=0.1):
        super().__init__(input_variables, output_variables, equation_type,
                         input_size, hidden_size, output_size, num_layers, dropout)

        self.input_size = input_size
        self.output_size = output_size
        
        # first layer (input)
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout))

        # hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(dropout))

        # lasy layer (output)
        layers.append(nn.Linear(hidden_size, output_size))

        # construct network from architecture defined through layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def __str__(self):
        return 'mlp'