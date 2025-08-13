from __future__ import annotations
import torch
from torch import Tensor
from torch.utils.data import Dataset
from reptile_world.config import Config, Action, Reward
from reptile_world.grid_world import GridWorld
from reptile_world.sdqn_model import SDQNModel
from reptile_world.reptile import Reptile


class Simulator:

    def __init__(self, conf: Config, world: GridWorld, animal: Reptile, result: SimulationResult,
                 save_result=True, verbose=0):
        # def __init__(self, world: GridWorld, animal: Reptile, result: SimulationResult, verbose=0):
        self.conf = conf
        self.world = world
        self.animal = animal
        self.result = result
        self.save_result = save_result
        self.verbose = verbose

    def run(self, n_steps: int) -> None:
        # action = self.world.zero_action()
        # observation, _ = self.world.resolve_action(action)

        if self.verbose >= 2:
            print("World at start:")
            self.world.print_color_world()
        if self.verbose >= 1:
            print("Simulation is starting ...")

        for t in range(n_steps):

            observation, reward = self.world.resolve_action(self.world.last_action)

            if self.verbose >= 2:
                print(t + 1, "/", n_steps)
            if self.verbose >= 2:
                self.world.print_action_reward(reward)
                self.world.print_color_world()

            if self.save_result:
                self.result.append(grid=self.world.grid,
                                   animal_pos=self.world.animal_pos,
                                   last_action=self.world.last_action,
                                   reward=reward)

            action = self.animal.act(observation, self.world.last_action)
            self.world.last_action = action

        if self.verbose >= 1:
            print("Simulation ended.")


class SimulationResult(Dataset):
    """
    Simple buffer for storing grid-world simulation trajectories and statistics.

    This dataset stores both raw per-step data and aggregated statistics for a single simulation run.
    It is designed for animation, replay, or inspection of an agent's behavior.

    Args:
        conf (Config): Configuration object containing grid dimensions.
        capacity (int): Maximum number of timesteps to store.

    Attributes for the first sample in the batch:
        grids (Tensor[capacity, H, W], long): The discrete grid state at each timestep,
            where each element is a cell type ID.
        animal_positions (Tensor[capacity, 2], long): The agent's (row, col) position at each timestep.
        last_actions (Tensor[capacity], long): The last action taken by the agent at each timestep.
        rewards (Tensor[capacity], float32): The reward received at each timestep.
    Batch-averaged attribute:
        avg_rewards (Tensor[capacity], float32): The average reward across the batch at each timestep.
    """

    def __init__(self, conf: Config, capacity: int):
        self.conf = conf
        # === Consume configuration parameters ===
        self.H, self.W = conf.grid_height, conf.grid_width

        # === Initialize class attributes ===
        self.capacity = capacity
        self.size = 0

        self.grids = torch.empty((self.capacity, self.H, self.W), dtype=torch.long)
        self.animal_positions = torch.empty((self.capacity, 2), dtype=torch.long)
        self.last_actions = torch.empty((self.capacity,), dtype=torch.long)
        self.rewards = torch.empty((self.capacity,), dtype=torch.float32)
        self.avg_rewards = torch.empty((self.capacity,), dtype=torch.float32)

    def append(self, grid: Tensor,
               animal_pos: Tensor,
               last_action: Action,
               reward: Reward):
        if self.size >= self.capacity:
            raise IndexError("SimulationResult capacity exceeded.")

        self.grids[self.size] = grid[0]
        self.animal_positions[self.size] = animal_pos[0]
        self.last_actions[self.size] = last_action[0]
        self.rewards[self.size] = reward[0]
        self.avg_rewards[self.size] = reward.mean()

        self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx >= self.size:
            raise IndexError("Index out of range")
        return (
            self.grids[idx],  # Tensor[H, W]
            self.animal_positions[idx],  # Tensor[2]
            self.last_actions[idx],  # Tensor[]
            self.rewards[idx],  # Tensor[]
            self.avg_rewards[idx],  # Tensor[]
        )


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    config = Config()
    world = GridWorld(config)
    model = SDQNModel(config)
    animal = Reptile(conf=config, model=model)
    n_step1, n_step2 = 10, 12
    result = SimulationResult(conf=config, capacity=n_step1+n_step2)
    sim = Simulator(conf=config,
                    world=world,
                    animal=animal,
                    result=result,
                    verbose=3)
    sim.run(n_steps=n_step1)
    print(sim.result.last_actions)
    sim.run(n_steps=n_step2)
    print(sim.result.last_actions)
