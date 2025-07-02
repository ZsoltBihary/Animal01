import torch
import torch.nn as nn


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, K, index_tensor=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K  # number of input connections per neuron

        # (out_features, K): For each output neuron, indices of its input connections
        if index_tensor is None:
            self.indices = torch.randint(0, in_features, (out_features, K))
        else:
            self.indices = index_tensor  # Tensor of shape (out_features, K)

        self.weight = nn.Parameter(torch.randn(out_features, K) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: (batch, in_features)
        # We gather input features according to fixed indices
        x_selected = x[:, self.indices]  # shape: (batch, out_features, K)
        out = torch.einsum('bok,ok->bo', x_selected, self.weight) + self.bias  # shape: (batch, out_features)
        # This is equivalent, but less extendable:
        # out = (x_selected * self.weight).sum(dim=-1) + self.bias
        return out


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    # from grid_world import GridWorld

    B, in_f, out_f, k = 2, 100, 90, 5
    device = 'cpu'
    sp_lin = SparseLinear(in_features=in_f, out_features=out_f, K=k, index_tensor=None)
    sp_lin.to(device=device)

    x_in = torch.randn(B, in_f)
    x_out = sp_lin(x_in)

    a = 42
