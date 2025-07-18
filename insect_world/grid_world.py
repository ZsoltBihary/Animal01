import torch
from torch import Tensor
# from typing import Self
# from utils.helper import EMPTY, FOOD, POISON, ANIMAL, FOOD_REWARD, POISON_REWARD, MOVE_REWARD
# from utils.helper import STAY, C
# from utils.helper import ResolveMove, print_grid, periodic_distance
from core.world import World
import torch.nn.functional as F


class GridWorld(World):
    # Cell types
    EMPTY, FOOD, POISON, ANIMAL = 0, 1, 2, 3
    C = 3  # number of observable cell types (w/o ANIMAL), used for shapes like (B, C, K, K)
    CELL_STR = {EMPTY: '   ', FOOD: ' o ', POISON: ' x ', ANIMAL: '(A)'}

    # Actions
    STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
    A = 5  # number of actions, used for shapes like (B, A)
    delta_pos = torch.tensor([
        [0, 0],  # STAY
        [-1, 0],  # UP
        [1, 0],  # DOWN
        [0, -1],  # LEFT
        [0, 1],  # RIGHT
    ])
    ACTION_STR = {STAY: 'STAY', UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}

    def __init__(self, B: int, H: int, W: int, R: int,
                 food_reward: float, poison_reward: float, move_reward: float,
                 food_density: float, poison_density: float):
        self.B, self.H, self.W, self.R = B, H, W, R  # batch, height, width, observation radius
        self.K = 2 * R + 1  # observation window size

        # Rewards depend on cell animal lands on:
        # EMPTY: move_reward, FOOD: food_reward, POISON: poison_reward, ANIMAL (i.e., if action=STAY): 0.0
        self.rewards = torch.tensor([move_reward, food_reward, poison_reward, 0.0])
        self.food_density, self.poison_density = food_density, poison_density
        empty_density = 1.0 - food_density - poison_density
        self.target_count = torch.tensor([empty_density, food_density, poison_density]) * H * W

        self.grid, self.animal_pos = self.random_start()

    def random_start(self) -> tuple[Tensor, Tensor]:
        """ Return batched random grids, batched animal_positions """
        # Initialize with EMPTY
        grid = torch.full((self.B, self.H, self.W), self.EMPTY, dtype=torch.long)
        # Assign FOOD and POISON
        rand_vals = torch.rand((self.B, self.H, self.W))
        grid[rand_vals < self.poison_density] = self.POISON
        grid[(rand_vals >= self.poison_density) & (rand_vals < self.poison_density + self.food_density)] = self.FOOD
        # Sample random animal positions uniformly
        y = torch.randint(0, self.H, (self.B,))
        x = torch.randint(0, self.W, (self.B,))
        animal_pos = torch.stack([y, x], dim=1)  # (B, 2)
        # Mark animal position in grid (overwrites whatever was there)
        grid[torch.arange(self.B), y, x] = self.ANIMAL
        return grid, animal_pos

    def get_observation(self) -> Tensor:
        """ Returns observation window around the animal: observation (B, K, K) """
        # Pad grid periodically
        padded = F.pad(self.grid, (self.R, self.R, self.R, self.R), mode="circular")  # (B, H + 2R, W + 2R)
        # Build index tensors
        b = torch.arange(self.B).view(self.B, 1, 1)
        y = self.animal_pos[:, 0].view(self.B, 1, 1) + torch.arange(self.K).view(1, self.K, 1)
        x = self.animal_pos[:, 1].view(self.B, 1, 1) + torch.arange(self.K).view(1, 1, self.K)
        # Use batched indexing to extract observation windows
        observation = padded[b, y, x]  # (B, K, K)
        return observation

    def resolve_action(self, action: Tensor) -> Tensor:
        """
        Args:
            action: Tensor of shape (B,)
        Returns:
            reward: Tensor of shape (B,)
        """
        # Determine animal's new position
        batch_idx = torch.arange(self.B)
        old_pos = self.animal_pos                   # (B, 2)
        new_pos = old_pos + self.delta_pos[action]  # (B, 2)
        new_pos[:, 0] %= self.H  # wrap y
        new_pos[:, 1] %= self.W  # wrap x
        # Gather cell types at new position, give reward
        cell_at_new = self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1]]  # (B, )
        reward = self.rewards[cell_at_new]  # (B, )
        # Remove animal from old position
        self.grid[batch_idx, old_pos[:, 0], old_pos[:, 1]] = self.EMPTY
        # Set animal on new position
        self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1]] = self.ANIMAL
        # Update position
        self.animal_pos = new_pos
        self.step()
        return reward

    def step(self):
        """
        Stochastically change one cell towards type that is farthest from its density target,
        if cell is outside of observation window.
        """
        # B, H, W = self.grid.shape
        # Compute current counts:
        count = F.one_hot(self.grid, num_classes=self.C+1).sum(dim=(1, 2))[:, :-1]  # (B, C)
        # Compute diff and select new cell per batch
        diff_count = (self.target_count[None, :] - count) / (self.target_count[None, :] + 1.0)  # (B, C)
        new_cell = diff_count.argmax(dim=1)  # (B,)
        # Sample random (ys, xs) locations for each grid in batch
        ys = torch.randint(0, self.H, (self.B,))  # (B,)
        xs = torch.randint(0, self.W, (self.B,))  # (B,)
        pos = torch.stack([ys, xs], dim=1)  # (B, 2)
        # Distance mask: avoid change within observation window
        d = self.periodic_distance(pos, self.animal_pos)  # (B,)
        change = (d > self.R).long()
        # Update cells using arithmetic masking (GPU-friendly)
        batch_idx = torch.arange(self.B)
        current_cell = self.grid[batch_idx, ys, xs]
        updated_cell = change * new_cell + (1 - change) * current_cell
        self.grid[batch_idx, ys, xs] = updated_cell

    # def get_visible_mask(self):
    #     # mask = self.grid[0]
    #     B, H, W = self.grid.shape
    #     xy = torch.arange(H*W)
    #     y = xy // W
    #     x = xy % W
    #     pos = torch.stack((y, x), dim=1)
    #     anim_pos = self.animal_pos[0][None, :]
    #     d = periodic_distance(pos, anim_pos, H, W)
    #     mask = (d <= self.R).reshape(H, W)
    #     return mask

    def print(self):
        horizontal = " "
        for _ in range(self.W * 3):
            horizontal = horizontal + "─"
        print(horizontal)
        # Iterate through each row of the grid
        for row in self.grid[0]:
            row_chars = [self.CELL_STR[value.item()] for value in row]
            row_string = ''.join(row_chars)
            print("│" + row_string + "│")
        print(horizontal)

    def periodic_distance(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Computes Chebyshev (max) distance with periodic boundary conditions on a grid.
        Args:
            a: Tensor of shape (B, 2) — positions (y, x)
            b: Tensor of shape (B, 2) — positions (y, x)
        Returns:
            Tensor of shape (B,) — periodic max-distance between a and b
        """
        dy = torch.abs(a[:, 0] - b[:, 0])
        dx = torch.abs(a[:, 1] - b[:, 1])
        # Account for wrap-around (toroidal distance)
        dy = torch.minimum(dy, self.H - dy)
        dx = torch.minimum(dx, self.W - dx)
        return torch.maximum(dy, dx)  # Chebyshev (L∞) distance


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    world = GridWorld(B=3, H=6, W=11, R=2,
                      food_reward=100.0, poison_reward=-100.0, move_reward=-1.0,
                      food_density=0.4, poison_density=0.2)
    # kitchen = Terrain.random(B=3, H=7, W=11, R=2,
    #                          food_density=0.1, poison_density=0.1)
    world.print()
    for _ in range(10):
        action = torch.ones(world.B, dtype=torch.long) * 3
        world.resolve_action(action)
        world.print()

    # mask = kitchen.get_visible_mask()
    # print(mask.int())
