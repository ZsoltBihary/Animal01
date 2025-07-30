from __future__ import annotations
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from core.world import World
from core.animal import Animal


class Simulator:

    def __init__(self, world: World, animal: Animal, result: SimulationResult, verbose=0):
        self.world = world
        self.animal = animal
        self.verbose = verbose
        self.result = result

    def run(self, n_steps: int) -> None:
        # action = self.world.zero_action()
        # observation, _ = self.world.resolve_action(action)

        if self.verbose >= 2:
            print("World at start:")
            self.world.print_world()
        if self.verbose >= 1:
            print("Simulation is starting ...")

        for t in range(n_steps):

            observation, reward = self.world.resolve_action()

            if self.verbose >= 2:
                print(t + 1, "/", n_steps)
            if self.verbose >= 2:
                self.world.print_action_reward(reward)
                self.world.print_world()

            avg_reward = torch.mean(reward)
            world_state = self.world.get_state()
            self.result.append(action=self.world.last_action[0],
                               world_state=world_state,
                               avg_reward=avg_reward)

            action = self.animal.act(observation)
            self.world.last_action = action

        if self.verbose >= 1:
            print("Simulation ended.")


class SimulationResult(Dataset, ABC):
    def __init__(self, capacity: int):
        self.capacity = capacity

    @abstractmethod
    def append(self, action: torch.Tensor, world_state: dict, avg_reward: torch.Tensor): ...
