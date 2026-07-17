import torch
from torch import nn
from swiftcfd.machineLearning.model.modelBase import ModelBase


class RecurrentNeuralNetwork(ModelBase):
    """
    A temporal RNN warm-start model.

    The flat feature vector is reshaped into a length-3 TIME sequence
    (t = n-2, n-1, n) and fed through an RNN; the last hidden state is decoded
    to the 5-point output stencil.  This lets the model use the time-history of
    each cell explicitly, instead of treating all features as one bag (MLP).

    Feature layout produced by DataManager (per variable, 7 cols):
        [c^n, e^n, w^n, n^n, s^n, c^{n-1}, c^{n-2}]
    For NS there are 3 such blocks (u, v, p) -> input_size = 21.
    For heat there is 1 block (T)            -> input_size = 7.

    Per time step we keep a 5-slot-per-variable stencil:
        t = n   : full 5-point stencil  [c, e, w, n, s]   (we have neighbours)
        t = n-1 : centre only           [c, 0, 0, 0, 0]   (only centre stored)
        t = n-2 : centre only           [c, 0, 0, 0, 0]
    so each step has width  n_vars * 5  (=15 for NS, =5 for heat).

    NOTE (scope): this is still a spatially LOCAL, per-cell model -- the RNN adds
    temporal modelling, not the global/elliptic spatial coupling that the
    pressure Poisson equation needs.  For that, use a whole-field model (U-Net).
    """

    def __init__(self, input_variables, output_variables, equation_type='heat',
                 input_size=7, hidden_size=256, output_size=5,
                 num_layers=5, dropout=0.1):
        super().__init__(input_variables, output_variables, equation_type,
                         input_size, hidden_size, output_size, num_layers, dropout)

        # number of physical variables packed into the feature vector
        self.n_vars = input_size // 7
        if self.n_vars * 7 != input_size:
            raise ValueError(
                f"RNN expects input_size to be a multiple of 7 "
                f"(7 features/variable), got {input_size}.")
        self.step_size = self.n_vars * 5     # stencil slots per time step

        rnn_layers = 2
        self.rnn = nn.RNN(
            input_size=self.step_size, hidden_size=hidden_size,
            num_layers=rnn_layers, batch_first=True, nonlinearity='tanh',
            dropout=dropout if rnn_layers > 1 else 0.0)

        # decoder head: hidden state -> output stencil
        head = [nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(min(num_layers - 1, 3)):
            head += [nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Dropout(dropout)]
        head.append(nn.Linear(hidden_size, output_size))
        self.head = nn.Sequential(*head)

    def _build_sequence(self, x):
        """(B, n_vars*7) -> (B, 3, n_vars*5) time sequence [n-2, n-1, n]."""
        B = x.shape[0]
        blocks = x.view(B, self.n_vars, 7)        # (B, V, 7)
        sten_n  = blocks[:, :, 0:5]               # (B, V, 5) current full stencil
        cen_nm1 = blocks[:, :, 5]                 # (B, V)
        cen_nm2 = blocks[:, :, 6]                 # (B, V)

        def centre_only(cen):                     # (B, V) -> (B, V, 5), centre at slot 0
            z = torch.zeros(B, self.n_vars, 5, device=x.device, dtype=x.dtype)
            z[:, :, 0] = cen
            return z

        seq = torch.stack([
            centre_only(cen_nm2).reshape(B, -1),  # t = n-2
            centre_only(cen_nm1).reshape(B, -1),  # t = n-1
            sten_n.reshape(B, -1),                # t = n
        ], dim=1)                                 # (B, 3, V*5)
        return seq

    def forward(self, x):
        seq = self._build_sequence(x)
        rnn_out, _ = self.rnn(seq)
        return self.head(rnn_out[:, -1, :])       # decode last time step

    def __str__(self):
        return 'rnn'
