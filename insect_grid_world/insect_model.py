# import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch import Tensor
from core.q_model import MetazoanQModel, DuelingQHead


class InsectQModel(MetazoanQModel):
    def __init__(self, state_shape: tuple[int, ...], num_actions: int,
                 hid_channels: int = 32, hid_features: int = 16):

        super().__init__(state_shape=state_shape, num_actions=num_actions)
        # self.state_shape = state_shape  # (B, C, K, K)

        B, C, K, _ = self.state_shape
        # hid_channels = 8
        # hid_features = 16
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=hid_channels, kernel_size=3, padding=1),  # (B, hid, K, K)
            nn.ReLU(),
            nn.Conv2d(in_channels=hid_channels, out_channels=1, kernel_size=3, padding=0),  # (B, 1, K-2, K-2)
            nn.Flatten()                                                                    # (B, (K-2) * (K-2))
        )
        self.duel = DuelingQHead(in_features=(K-2)*(K-2), hid_features=hid_features, num_actions=num_actions)

    def forward(self, state: Tensor) -> Tensor:
        # C, H, W = self._input_shape
        # assert state.shape[1:] == (C, H, W), \
        #     f"Expected input shape (B, {C}, {H}, {W}), got {state.shape}"
        x = self.trunk(state)
        q_values = self.duel(x)
        return q_values


# === SOME MODEL IDEAS FOR LATER ===
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
