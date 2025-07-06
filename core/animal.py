# animal.py
# from core.policy import BasePolicy

from abc import ABC, abstractmethod
import torch
from torch import Tensor
from utils.helper import N_ACTIONS
from core.brain import Brain


# Base abstract class
class Animal(ABC):
    @abstractmethod
    def policy(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Takes observation tensor of shape (B, K, K).
        Returns:
            action: Tensor of shape (B,)
            cost: Tensor of shape (B,)
        """
        pass


class Amoeba(Animal):
    # Random policy — no brain used
    # def __init__(self):
    #     self.n_actions = N_ACTIONS

    def policy(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        B = obs.shape[0]
        action = torch.randint(low=0, high=N_ACTIONS, size=(B,))
        cost = torch.zeros(B)
        return action, cost


class Mammal(Animal):
    # Brain-based policy
    """
    Greedy Brain-based policy
    This is used for simulation, so BrainModel runs in 'think' mode
    """
    def __init__(self, brain: Brain):
        self.brain = brain

    def policy(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode observation into brain
        self.brain.encode_observation(obs)
        # Compute Q-values and energy cost of thinking
        q_values, cost = self.brain.think()             # shape: (B, A), (B, )
        print("q_values:", q_values[0, :])
        # Pick action greedily
        action = torch.argmax(q_values, dim=1)   # shape: (B, )
        return action, cost


# class Animal:
#     def __init__(self, policy: BasePolicy):
#         """
#         Args:
#             policy: An instance of BasePolicy (e.g., RandomPolicy, BrainPolicy)
#         """
#         self.policy = policy

    # def act(self, obs: Tensor) -> tuple[Tensor, Tensor]:
    #     """
    #     Choose action based on observation.
    #     Args:
    #         obs: Tensor of shape (B, n_connections, n_connections)
    #     Returns:
    #         actions: Tensor of shape (B,)
    #         brain_cost: Tensor of shape (B,)  (0.0 if no brain used)
    #     """
    #     return self.policy(obs)
