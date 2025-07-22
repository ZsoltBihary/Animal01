from torch import Tensor
import torch.nn as nn
from abc import ABC, abstractmethod


class MetazoanQModel(nn.Module, ABC):
    def __init__(self, state_shape: tuple[int, ...], num_actions: int):
        super().__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor: ...


class VertebrateQModel(MetazoanQModel):
    @abstractmethod
    def Q_and_update(self, state: Tensor) -> tuple[Tensor, Tensor]: ...

    def forward(self, state: Tensor) -> Tensor:
        q_values, _ = self.Q_and_update(state)
        return q_values


class DuelingQHead(nn.Module):
    def __init__(self, in_features, hid_features, num_actions):
        """
        Args:
            in_features:  Dimension of the input features (output from trunk).
            hid_features:  Dimension of the hidden features (output.
            num_actions:  Number of discrete actions.
        """
        super().__init__()

        # Value stream: outputs a scalar V(s)
        self.value_stream = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=in_features, out_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=1)
        )

        # Advantage stream: outputs a vector A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=in_features, out_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=num_actions)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, feature_dim), the output from the trunk.
        Returns:
            q_values: Tensor of shape (B, num_actions)
        """
        V = self.value_stream(x)                   # (B, 1)
        A = self.advantage_stream(x)               # (B, A)
        A_mean = A.mean(dim=1, keepdim=True)       # (B, 1)

        # Combine streams: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        Q = V + (A - A_mean)                       # (B, A)
        return Q


# class QInsect01(nn.Module):
#     """
#     Standard residual CNN model
#     """
#     def __init__(self):
#         super().__init__()
#         hid_features = 2 * A
#         self.trunk = nn.Linear(in_features=C * K * K, out_features=hid_features)
#         self.duel_head = DuelingQHead(feature_dim=hid_features, num_actions=A)
#
#     def forward(self, state: Tensor):
#         """
#         Input: state(B, C, K, K)
#         Output: q_values(B, A)
#         """
#         x = state.flatten(1)
#         x = self.trunk(x)
#         x = torch.relu(x)
#         q_values = self.duel_head(x)
#
#         return q_values
#
#
# class QInsect02(nn.Module):
#     """
#     Standard residual CNN model
#     """
#     def __init__(self, main_channels: int):
#         super().__init__()
#         self.input_conv = nn.Conv2d(in_channels=C, out_channels=main_channels, kernel_size=3, padding=1)
#         self.output_conv = nn.Conv2d(in_channels=main_channels, out_channels=1, kernel_size=3, padding=1)
#         self.output_lin = nn.Linear(in_features=K * K, out_features=A)
#
#     def forward(self, state: Tensor):
#         """
#         Input: state(B, C, K, K)
#         Output: q_values(B, A)
#         """
#         x = state
#         x = self.input_conv(x)
#         x = torch.relu(x)
#         x = self.output_conv(x)
#         x = torch.relu(x)
#         x = x.flatten(1)
#         q_values = self.output_lin(x)
#         return q_values
#
#

#
#
# # ------------------ SANITY CHECK ------------------ #
# if __name__ == "__main__":
#     # a = torch.tensor([[0, 0], [9, 9]]).long()
#     # b = torch.tensor([[9, 6], [0, 0]]).long()
#     # print(periodic_distance(a, b, H=10, W=10))
#     B = 2
#     state = torch.randn(B, C, K, K)
#     print(state.shape)
#     q_model = QInsect01()
#     q_values = q_model(state)
#     print(q_values.shape)
