# animal.py
from torch import Tensor
from core.policy import BasePolicy
from core.brain import Brain


class Animal:
    def __init__(self, policy: BasePolicy, obs_radius: int, brain: Brain = None):
        """
        Args:
            policy: An instance of BasePolicy (e.g., RandomPolicy, BrainPolicy)
            obs_radius: Observation Radius (OR)
                Size of the local observation window: K = 2 * OR + 1
        """
        self.policy = policy
        self.obs_radius = obs_radius
        self.brain = brain

    def act(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Choose action based on observation.
        Args:
            obs: Tensor of shape (B, K, K)
        Returns:
            actions: Tensor of shape (B,)
            brain_cost: Tensor of shape (B,)  (0.0 if no brain used)
        """
        return self.policy(obs)
