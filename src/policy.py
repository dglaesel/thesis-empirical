"""Policy architectures for signature-based control."""

__all__ = ["LinearPolicy", "DNNPolicy", "DNNStrategy"]

import torch
import torch.nn as nn


class LinearPolicy(nn.Module):
    """Linear policy: v = w^T features + b."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class DNNPolicy(nn.Module):
    """Two-layer ReLU network: Linear(in, h) -> ReLU -> Linear(h, 1).

    Architecture matches Chapter 7 Section 7.5: one hidden layer, h=32.
    """

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, 1, dtype=torch.float64),
        )
        nn.init.xavier_uniform_(self.net[0].weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class DNNStrategy(nn.Module):
    """Phase 1 DNN (Bank et al.): [Linear->ReLU] x nn_hidden -> Linear."""

    def __init__(self, in_dim: int, nn_hidden: int = 2, nn_dropout: float = 0.0):
        super().__init__()
        a_dim = in_dim
        hidden = in_dim + 30
        layers: list[nn.Module] = [nn.Dropout(p=nn_dropout)]
        for _ in range(nn_hidden):
            lin = nn.Linear(a_dim, hidden, dtype=torch.float64)
            nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(lin.bias)
            layers.extend([lin, nn.ReLU()])
            a_dim = hidden
        final = nn.Linear(a_dim, 1, dtype=torch.float64)
        nn.init.xavier_normal_(final.weight)
        nn.init.zeros_(final.bias)
        layers.extend([final, nn.Flatten(start_dim=0)])
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)
