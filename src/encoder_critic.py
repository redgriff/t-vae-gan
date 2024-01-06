from typing import List
import torch

from mlp_network import MLPNetwork


class EncoderCritic(torch.nn.Module):
    def __init__(self, in_size: int, hidden_dims: List[int], out_size: int, dropout: float, device=None, is_vae=False):
        super(EncoderCritic, self).__init__()

        self.device = device
        self.is_vae = is_vae

        self.network = MLPNetwork(in_size=in_size, hidden_sizes=hidden_dims, out_size=out_size, device=device,
                                  dropout=dropout).to(device=self.device)

        self.mu_fc = torch.nn.Linear(out_size, out_size).to(device=device)
        self.sigma_fc = torch.nn.Linear(out_size, out_size).to(device=device)

        torch.nn.init.xavier_uniform_(self.mu_fc.weight)
        if self.mu_fc.bias is not None:
            torch.nn.init.zeros_(self.mu_fc.bias)
        torch.nn.init.xavier_uniform_(self.sigma_fc.weight)
        if self.sigma_fc.bias is not None:
            torch.nn.init.zeros_(self.sigma_fc.bias)

        self.mu = None
        self.sigma = None

    def forward(self, x):
        out = self.network(x)

        if self.is_vae:
            self.mu = self.mu_fc(out)
            self.sigma = torch.exp(self.sigma_fc(out))
            out = self.mu + self.sigma * torch.randn_like(out)

        return out
