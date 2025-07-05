from abc import ABC, abstractmethod
import torch
from torch import Tensor
from utils.helper import N_ACTIONS


# Base abstract class
class BasePolicy(ABC):
    @abstractmethod
    def __call__(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Takes observation tensor of shape (B, K, K).
        Returns:
            actions: Tensor of shape (B,)
            brain_cost: Tensor of shape (B,)
        """
        pass


# Random policy — no brain used
class RandomPolicy(BasePolicy):
    def __init__(self):
        self.n_actions = N_ACTIONS

    def __call__(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        B = obs.shape[0]
        actions = torch.randint(low=0, high=self.n_actions, size=(B,))
        brain_cost = torch.zeros(B)
        return actions, brain_cost


# TODO: Connect this to brain modelling ...
# # Brain-based policy
# class BrainPolicy(BasePolicy):
#     def __init__(self, brain):
#         self.brain = brain  # Must have .forward() and access to internal activity
#
#     def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         # Encode observation into brain
#         self.brain.encode_observation(obs)
#
#         # Forward pass to compute Q-values or logits
#         q_values = self.brain.forward()  # shape: (B, A)
#
#         # Pick action greedily
#         actions = torch.argmax(q_values, dim=1)
#
#         # Compute energy cost — e.g., L1 norm of perceptor or whole state
#         brain_cost = self.brain.compute_energy_cost()
#
#         return actions, brain_cost
