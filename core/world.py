# world.py
# import torch
# from torch import Tensor
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

        obs = self.terrain.get_observation()          # (B, K, K)
        if verbose >= 2:
            print("Observed Window:")
            print_grid(grid=obs[0], frame=True)
        action = self.animal.policy(obs)                             # (B, )
        reward = self.terrain.resolve_action(action)                 # (B, )
        self.terrain.step()
        if verbose >= 1:
            print("Action:", ACTION_STR[action[0].item()], "  Reward:", reward[0].item())
            self.terrain.print()

        return reward

    def simulate(self, n_steps, verbose=1):
        print("Start:")
        self.terrain.print()
        for t in range(n_steps):
            self.step(verbose=verbose)

    # TODO: factory method is abandoned here, use explicit construction externally ... To be deleted ???
    # @classmethod
    # def factory(cls, B: int, H: int, W: int, R: int, food_dens: float, poison_dens: float,
    #             animal_cls: type[Animal],
    #             animal_kwargs: dict = None,  # optional kwargs
    #             device='cpu') -> Self:
    #
    #     terrain = Terrain.random(B=B, H=H, W=W, R=R,
    #                              food_density=food_dens, poison_density=poison_dens,
    #                              device=device)
    #     # animal = animal_cls()
    #     animal = animal_cls(**(animal_kwargs or {}))
    #     world = World(terrain=terrain, animal=animal)
    #     return world
