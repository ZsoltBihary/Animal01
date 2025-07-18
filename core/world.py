# world.py
import torch
from torch import Tensor
# from core.animal import Animal
# from core.terrain import Terrain
# from utils.helper import print_grid, ACTION_STR
from abc import ABC, abstractmethod


class World(ABC):
    ACTION_STR: None

    @abstractmethod
    def get_observation(self) -> Tensor: ...
    # Output observation: Tensor (B, ...)

    @abstractmethod
    def resolve_action(self, action: Tensor) -> Tensor: ...
    # Input action: Tensor (B, )
    # Output reward: Tensor (B, )

    @abstractmethod
    def print(self) -> None: ...
    # Print world state

    # def simulate(self, n_steps: int, save_history: bool = False) -> Tensor | None:
    #
    #     if save_history:
    #         # TODO: set up history
    #         pass
    #     if self.verbose >= 2:
    #         print("World at start:")
    #         self.print()
    #     if self.verbose >= 1:
    #         print("Simulation is starting ...")
    #
    #     for t in range(n_steps):
    #         if self.verbose >= 2:
    #             print(t+1, "/", n_steps)
    #         observation = self.get_observation()
    #         action = self.animal.act(observation)
    #         reward = self.resolve_action(action)
    #         if self.verbose >= 2:
    #             print("Action:", ACTION_STR[action[0].item()], "  Reward:", reward[0].item())
    #             self.print()
    #
    #     if self.verbose >= 1:
    #         print("Simulation ended.")

# class World:
#     def __init__(self, terrain: Terrain, animal: Animal):
#         """
#         Args:
#             terrain: The environment in our world.
#             animal: The agent in our world.
#         """
#         self.terrain = terrain
#         self.animal = animal
#         self.grid_history = None
#         self.grid_history = None
#
#     def simulate_step(self, verbose=1):
#
#         obs = self.terrain.get_observation()          # (B, K, K)
#         if verbose >= 2:
#             print("Observed Window:")
#             print_grid(grid=obs[0], frame=True)
#         action = self.animal.policy(obs)                             # (B, )
#         reward = self.terrain.resolve_action(action)                 # (B, )
#         self.terrain.step()
#         if verbose >= 1:
#             print("Action:", ACTION_STR[action[0].item()], "  Reward:", reward[0].item())
#             self.terrain.print()
#
#         return reward
#
#     def simulate(self, n_steps, verbose=1, save_history=False):
#         if save_history:
#             B, H, W = self.terrain.grid.shape
#             self.history = (torch.zeros(n_steps+1, H, W), torch.zeros(n_steps+1, H, W, dtype=torch.bool))
#             self.history[0][0, :, :] = self.terrain.grid[0, :, :]
#             self.history[1][0, :, :] = self.terrain.get_visible_mask()
#         print("Start:")
#         self.terrain.print()
#         for t in range(n_steps):
#             self.simulate_step(verbose=verbose)
#             if save_history:
#                 self.history[0][t+1, :, :] = self.terrain.grid[0, :, :]
#                 self.history[1][t+1, :, :] = self.terrain.get_visible_mask()
#
#     # TODO: factory method is abandoned here, use explicit construction externally ... To be deleted ???
#     # @classmethod
#     # def factory(cls, B: int, H: int, W: int, R: int, food_dens: float, poison_dens: float,
#     #             animal_cls: type[Animal],
#     #             animal_kwargs: dict = None,  # optional kwargs
#     #             device='cpu') -> Self:
#     #
#     #     terrain = Terrain.random(B=B, H=H, W=W, R=R,
#     #                              food_density=food_dens, poison_density=poison_dens,
#     #                              device=device)
#     #     # animal = animal_cls()
#     #     animal = animal_cls(**(animal_kwargs or {}))
#     #     world = World(terrain=terrain, animal=animal)
#     #     return world
