# import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch import Tensor
from core.q_model import MetazoanQModel


class InsectQModel(MetazoanQModel):
    def __init__(self, input_shape: tuple[int, int, int], num_actions: int):
        super().__init__()
        self._input_shape = input_shape  # (C, K, K)

        C, K, _ = input_shape
        hid_channels = 8
        hid_features = 16
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=hid_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=hid_channels * K * K, out_features=hid_features),
            nn.ReLU(),
            nn.Linear(in_features=hid_features, out_features=num_actions)
        )

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    def forward(self, state: Tensor) -> Tensor:
        C, H, W = self._input_shape
        assert state.shape[1:] == (C, H, W), \
            f"Expected input shape (B, {C}, {H}, {W}), got {state.shape}"
        q_values = self.trunk(state)
        return q_values
