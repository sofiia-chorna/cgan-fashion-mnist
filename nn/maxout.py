import torch
import torch.nn as nn


class Maxout(nn.Module):
    def __init__(self, d_in, num_units, num_piece):
        super().__init__()

        # params
        self.d_in = d_in
        self.num_units = num_units
        self.num_piece = num_piece

        # layer
        self.linear = nn.Linear(d_in, num_units * num_piece)

    def forward(self, x):
        x = self.linear(x)

        # reshape to (batch_size, units, pieces)
        x = x.view(-1, self.num_units, self.num_piece)

        # take the max across pieces => suite au nom d'activation :)
        x = torch.max(x, dim=2).values

        return x
