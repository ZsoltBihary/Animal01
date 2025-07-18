# from torch import Tensor
from core.world import World
from core.animal import Animal


class Simulator:

    def __init__(self, world: World, animal: Animal, verbose=0):
        self.world = world
        self.animal = animal
        self.verbose = verbose
        self.history = None

    def run(self, n_steps: int, save_history: bool = False) -> None:
        if save_history:
            # TODO: set up history
            pass
        if self.verbose >= 2:
            print("World at start:")
            self.world.print()
        if self.verbose >= 1:
            print("Simulation is starting ...")

        for t in range(n_steps):
            if self.verbose >= 2:
                print(t + 1, "/", n_steps)
            observation = self.world.get_observation()
            action = self.animal.act(observation)
            reward = self.world.resolve_action(action)
            if self.verbose >= 2:
                print("Action:", self.world.ACTION_STR[action[0].item()], "  Reward:", reward[0].item())
                self.world.print()

        if self.verbose >= 1:
            print("Simulation ended.")
