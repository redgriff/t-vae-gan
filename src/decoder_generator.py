from typing import List
import torch

from src.mlp_network import MLPNetwork


class DecoderGenerator(torch.nn.Module):
    def __init__(self, in_size: int, hidden_dims: List[int], out_size: int, cat_size: int, num_size: int,
                 dropout: float, device=None):
        super(DecoderGenerator, self).__init__()

        self.device = device
        self.cat_size = cat_size
        self.num_size = num_size

        self.mlp_network = MLPNetwork(in_size=in_size, hidden_sizes=hidden_dims, out_size=out_size,
                                      dropout=dropout, device=device).to(device=device)

        self.activation = torch.nn.Hardtanh().to(device=device)

    def forward(self, x):
        out = self.mlp_network(x)
        out = self.activation(out)

        return out
