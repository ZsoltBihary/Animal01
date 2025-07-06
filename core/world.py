
# import torch
# from torch import Tensor
from typing import Self
from nn_model.brain_model import BrainModel
from core.brain import Brain
# from core.policy import RandomPolicy, BrainPolicy
from core.animal import Animal, Amoeba, Mammal
from core.terrain import Terrain
from utils.helper import print_grid, ACTION_STR, N_ACTIONS, N_CELL_TYPES


class World:
    def __init__(self, terrain: Terrain, animal: Animal):
        """
        Args:
            terrain: The environment in our world.
            animal: The agent in our world.
        """
        self.terrain = terrain
        self.animal = animal

    @classmethod
    def factory(cls, B: int, H: int, W: int, R: int, wall_dens: float, food_dens: float,
                with_brain=False,
                n_neurons=0, n_motors=0, n_connections=0,
                device='cpu') -> Self:

        terrain = Terrain.random(B=B, H=H, W=W, R=R,
                                 wall_density=wall_dens, food_density=food_dens,
                                 device=device)

        if with_brain:
            K = R * 2 + 1
            n_perceptors = K * K * (N_CELL_TYPES - 1)
            model = BrainModel(n_neurons=n_neurons,
                               n_perceptors=n_perceptors, n_motors=n_motors, n_connections=n_connections,
                               n_actions=N_ACTIONS)

            brain = Brain(B=B, model=model, device=device)
            animal = Mammal(brain=brain)
            # policy = BrainPolicy(brain=brain)
        else:
            animal = Amoeba()
            # policy = RandomPolicy()

        # animal = Animal(policy=policy)

        world = World(terrain=terrain, animal=animal)

        return world

    def step(self, verbose=1):

        obs = self.terrain.get_observation()          # (B, K, K)
        if verbose >= 2:
            print("Observed Window:")
            print_grid(grid=obs[0], frame=True)
        action, brain_cost = self.animal.policy(obs)                        # (B,), (B,)
        action_reward = self.terrain.resolve_action(action)                 # (B,)
        reward = action_reward + brain_cost                                 # (B,)
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
