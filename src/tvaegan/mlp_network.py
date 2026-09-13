from typing import List
import torch


class MLPNetwork(torch.nn.Module):
    def __init__(self, in_size: int, hidden_sizes: List[int], out_size: int, dropout: float, device=None):
        super(MLPNetwork, self).__init__()

        out_features = hidden_sizes[0] if len(hidden_sizes) > 0 else out_size

        self.input = torch.nn.Sequential(
            torch.nn.Linear(in_size, out_features),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(),
        ).to(device=device)

        self.hidden = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                    torch.nn.Dropout(dropout),
                    torch.nn.ReLU(),
                )
                for i in range(1, len(hidden_sizes))
            ]
        ).to(device=device)

        in_features = hidden_sizes[-1] if len(hidden_sizes) > 0 else out_size
        self.output = torch.nn.Linear(in_features=in_features, out_features=out_size).to(device=device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input(x)

        for layer in self.hidden:
            x = layer(x)

        out = self.output(x)

        return out
