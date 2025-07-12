import torch
from torch import Tensor
import torch.nn as nn
from utils.helper import A, C, K


class QInsect(nn.Module):
    """
    Standard residual CNN model
    """
    def __init__(self, main_channels: int):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels=C, out_channels=main_channels, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(in_channels=main_channels, out_channels=1, kernel_size=3, padding=1)
        self.output_lin = nn.Linear(in_features=K * K, out_features=A)

    def forward(self, state: Tensor):
        """
        Input: state(B, C, K, K)
        Output: q_values(B, A)
        """
        x = state
        x = self.input_conv(x)
        x = torch.relu(x)
        x = self.output_conv(x)
        x = torch.relu(x)
        x = x.flatten(1)
        q_values = self.output_lin(x)
        return q_values


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    # a = torch.tensor([[0, 0], [9, 9]]).long()
    # b = torch.tensor([[9, 6], [0, 0]]).long()
    # print(periodic_distance(a, b, H=10, W=10))
    B = 32
    state = torch.randn(B, C, K, K)
    print(state.shape)
    q_model = QInsect(main_channels=16)
    q_values = q_model(state)
    print(q_values.shape)
