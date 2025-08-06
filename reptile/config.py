from dataclasses import dataclass
import torch
from torch import Tensor

Observation = Tensor
Action = Tensor
Encoded = Tensor
BrainState = Tensor
PredQ = Tensor
PredReward = Tensor
PredObservation = Tensor

CUDA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    batch_size: int = 256
    num_actions: int = 5
    obs_channels: int = 3
    obs_size: int = 7

    brain_state_layers: int = 3
    brain_state_channels: int = 32
    brain_state_device: torch.device = CUDA_DEVICE

    epsilon0: float = 0.1
    epsilon1: float = 0.02
    temperature0: float = 0.1
    temperature1: float = 0.02

    model_device: torch.device = CUDA_DEVICE
