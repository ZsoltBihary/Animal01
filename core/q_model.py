import torch
from torch import Tensor
import torch.nn as nn
from abc import ABC, abstractmethod
from core.tensordict_helper import Schema, State, Observation, Action


class MetazoanQModel(nn.Module, ABC):
    def __init__(self, state_schema: Schema,  num_actions: int):
        super().__init__()
        self.state_schema = state_schema
        self.num_actions = num_actions

    @abstractmethod
    def forward(self, state: State) -> Tensor: ...


class SDQNModel(nn.Module, ABC):
    @abstractmethod
    def SDQN(self, observation: Observation,  state: State) -> Tensor: ...

    @abstractmethod
    def forward(self, state: State) -> Tensor: ...


# class VertebrateQModel(MetazoanQModel):
#     @abstractmethod
#     def Q_and_update(self, state: State) -> tuple[Tensor, State]: ...
#
#     def forward(self, state: State) -> Tensor:
#         q_values, _ = self.Q_and_update(state)
#         return q_values


# class DuelingQHead(nn.Module):
#     def __init__(self, in_features, hid_features, num_actions):
#         """
#         Args:
#             in_features:  Dimension of the input features (output from trunk).
#             hid_features:  Dimension of the hidden features.
#             num_actions:  Number of discrete actions.
#         """
#         super().__init__()
#
#         # Value stream: outputs a scalar V(s)
#         self.value_stream = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(in_features=in_features, out_features=hid_features),
#             nn.ReLU(),
#             nn.Linear(in_features=hid_features, out_features=1)
#         )
#
#         # Advantage stream: outputs a vector A(s, a)
#         self.advantage_stream = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(in_features=in_features, out_features=hid_features),
#             nn.ReLU(),
#             nn.Linear(in_features=hid_features, out_features=num_actions)
#         )
#
#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (B, feature_dim), the output from the trunk.
#         Returns:
#             q_values: Tensor of shape (B, num_actions)
#         """
#         V = self.value_stream(x)                   # (B, 1)
#         A = self.advantage_stream(x)               # (B, A)
#         A_mean = A.mean(dim=1, keepdim=True)       # (B, 1)
#
#         # Combine streams: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
#         Q = V + (A - A_mean)                       # (B, A)
#         return Q
