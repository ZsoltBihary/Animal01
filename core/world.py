# import torch
# from torch import Tensor
# import torch.nn.functional as F
# from utils.helper import EMPTY  # Includes EMPTY=0
from core.animal import Animal
from core.terrain import Terrain
from utils.helper import print_grid, ACTION_STR


class World:
    def __init__(self, terrain: Terrain, animal: Animal):
        """
        Args:
            terrain: The environment in our world.
            animal: The agent in our world.
        """
        self.terrain = terrain
        self.animal = animal

    def step(self, verbose=1):

        obs = self.terrain.get_observation(self.animal.obs_radius)          # (B, K, K)
        if verbose >= 2:
            print("Observed Window:")
            print_grid(grid=obs[0], frame=True)
        action, brain_cost = self.animal.policy(obs)                        # (B,), (B,)
        action_reward = self.terrain.resolve_action(action)                 # (B,)
        reward = action_reward - brain_cost                                 # (B,)
        if verbose >= 1:
            print("Action:", ACTION_STR[action[0].item()], "  Reward:", reward[0].item())
            self.terrain.print()

        return reward

    def simulate(self, n_steps, verbose=1):
        print("Start:")
        self.terrain.print()
        for t in range(n_steps):
            self.step(verbose=verbose)
