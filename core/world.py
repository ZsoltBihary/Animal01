# world.py
# import torch
from torch import Tensor
# from core.animal import Animal
# from core.terrain import Terrain
# from utils.helper import print_grid, ACTION_STR
from abc import ABC, abstractmethod
from core.tensordict_helper import Observation, Action


class World(ABC):
    def __init__(self, B: int):
        self.B = B

    @abstractmethod
    def resolve_action(self, action: Action) -> tuple[Observation, Tensor]: ...
    # Input
    #   action: Action: Tensor(B, )
    # Output
    #   observation: Observation: TensorDict(B, ...)
    #   reward: Tensor (B, )

    @abstractmethod
    def get_state(self) -> dict: ...

    @abstractmethod
    def zero_action(self) -> Action: ...

    @abstractmethod
    def print_world(self) -> None: ...
    # Print world state

    @abstractmethod
    def print_action_reward(self, action: Action, reward: Tensor) -> None: ...
    # Print action and reward
