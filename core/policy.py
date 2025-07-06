from abc import ABC, abstractmethod
import torch
from torch import Tensor
from utils.helper import N_ACTIONS
from core.brain import Brain


# Base abstract class
class BasePolicy(ABC):
    @abstractmethod
    def __call__(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Takes observation tensor of shape (B, n_connections, n_connections).
        Returns:
            actions: Tensor of shape (B,)
            cost: Tensor of shape (B,)
        """
        pass


# Random policy — no brain used
class RandomPolicy(BasePolicy):
    def __init__(self):
        self.n_actions = N_ACTIONS

    def __call__(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        B = obs.shape[0]
        actions = torch.randint(low=0, high=self.n_actions, size=(B,))
        cost = torch.zeros(B)
        return actions, cost


# Brain-based policy
class BrainPolicy(BasePolicy):
    """
    Greedy Brain-based policy
    This is used for simulation, so BrainModel runs in 'think' mode
    """
    def __init__(self, brain: Brain):
        self.brain = brain

    def __call__(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode observation into brain
        self.brain.encode_observation(obs)
        # Compute Q-values and energy cost of thinking
        q_values, cost = self.brain.think()             # shape: (B, A), (B, )
        # Pick action greedily
        action = torch.argmax(q_values, dim=1)   # shape: (B, )
        return action, cost
