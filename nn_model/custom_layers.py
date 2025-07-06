import torch
from torch import Tensor
import torch.nn as nn


class SparseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, connections: int, index_tensor: Tensor = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = connections  # number of input connections per neuron

        # (out_features, n_connections): For each output neuron, indices of its input connections
        if index_tensor is None:
            self.indices = torch.randint(0, in_features, (out_features, connections))
        else:
            self.indices = index_tensor  # Tensor of shape (out_features, n_connections)

        self.weight = nn.Parameter(torch.randn(out_features, connections) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: (batch, in_features)
        # We gather input features according to fixed indices
        x_selected = x[:, self.indices]  # shape: (batch, out_features, n_connections)
        out = torch.einsum('bok,ok->bo', x_selected, self.weight) + self.bias  # shape: (batch, out_features)
        # This is equivalent, but less extendable:
        # out = (x_selected * self.weight).sum(dim=-1) + self.bias
        return out

    def set_index_tensor(self, index_tensor: Tensor):
        self.indices = index_tensor


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    # from grid_world import Terrain

    B, in_f, out_f, conn = 2, 100, 90, 5
    device = 'cpu'
    sp_lin = SparseLinear(in_features=in_f, out_features=out_f, connections=conn, index_tensor=None)
    sp_lin.to(device=device)

    x_in = torch.randn(B, in_f)
    x_out = sp_lin(x_in)

    print(x_out)

    a = 42
