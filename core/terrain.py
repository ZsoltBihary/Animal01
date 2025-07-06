import torch
from torch import Tensor
from typing import Self
from utils.helper import EMPTY, WALL, FOOD, ANIMAL, FOOD_REWARD, WALL_PENALTY, MOVE_COST, STAY
from utils.helper import ResolveMove, print_grid, periodic_distance
import torch.nn.functional as F
# from utils.helper import N_CELL_TYPES  # Includes EMPTY=0


class Terrain:
    def __init__(self, grid: Tensor, R: int, food_density: float, animal_pos: Tensor):
        """
        Args:
            grid: Tensor of shape (B, H, W), with cell type codes.
            animal_pos: Tensor of shape (B, 2), animal positions (y, x)
        """
        self.grid = grid                # (B, H, W)
        self.animal_pos = animal_pos    # (B, 2)
        self.R = R
        self.food_density = food_density
        B, H, W = grid.shape
        self.move_fn = ResolveMove(H, W)

    @classmethod
    def random(cls, B: int, H: int, W: int, R: int,
               wall_density: float, food_density: float,
               device='cpu') -> Self:
        """
        Create random Terrain with batch of grids and animal positions.
        Returns:
            Terrain instance with randomly initialized grids and animal_pos.
        """
        # Initialize grid with EMPTY
        grid = torch.full((B, H, W), EMPTY, dtype=torch.long, device=device)
        # Assign WALL and FOOD
        rand_vals = torch.rand((B, H, W), device=device)
        grid[rand_vals < wall_density] = WALL
        grid[(rand_vals >= wall_density) & (rand_vals < wall_density + food_density)] = FOOD
        # Sample random animal positions uniformly
        y = torch.randint(0, H, (B,), device=device)
        x = torch.randint(0, W, (B,), device=device)
        animal_pos = torch.stack([y, x], dim=1)  # (B, 2)
        # Mark animal position in grid (overwrites whatever was there)
        grid[torch.arange(B), y, x] = ANIMAL
        return cls(grid, R, food_density, animal_pos)

    def get_observation(self) -> Tensor:
        """
        Returns local window around the animal:
        Output shape: (B, K, K), with K = 2*R + 1
        """
        B, H, W = self.grid.shape
        K = 2 * self.R + 1
        # Pad grid toroidally
        padded = F.pad(self.grid, (self.R, self.R, self.R, self.R), mode="circular")  # (B, H + 2R, W + 2R)

        # pos = self.animal_pos  # (B, 2)
        # Build index tensors
        b = torch.arange(B).view(B, 1, 1)
        y = self.animal_pos[:, 0].view(B, 1, 1) + torch.arange(K).view(1, K, 1)
        x = self.animal_pos[:, 1].view(B, 1, 1) + torch.arange(K).view(1, 1, K)

        # Use batched indexing to extract local windows
        local = padded[b, y, x]  # (B, K, K)
        # # One-hot encode (skip EMPTY = 0)
        # onehot = F.one_hot(local, num_classes=N_CELL_TYPES)[..., 1:]  # (B, n_connections, n_connections, C)
        # obs = onehot.permute(0, 3, 1, 2).float()  # (B, C, n_connections, n_connections)
        return local

    def resolve_action(self, action: Tensor) -> Tensor:
        """
        Args:
            action: Tensor of shape (B,)
        Returns:
            reward: Tensor of shape (B,)
        """
        B, H, W = self.grid.shape
        old_pos = self.animal_pos                # (B, 2)
        new_pos = self.move_fn(old_pos, action)  # (B, 2)
        # Gather cell types at new position
        y, x = new_pos[:, 0], new_pos[:, 1]  # (B,)
        batch_idx = torch.arange(B, device=action.device)
        cell_at_new = self.grid[batch_idx, y, x]  # (B,)

        # Movement cost for non-STAY actions
        reward = (action != STAY) * MOVE_COST
        # Add food reward where applicable
        reward += (cell_at_new == FOOD) * FOOD_REWARD
        # Subtract wall penalty where applicable
        reward += (cell_at_new == WALL) * WALL_PENALTY

        # Update grid and position (only if not wall)
        move_mask = (cell_at_new != WALL).long().unsqueeze(-1)  # (B, 1)
        # Remove animal from old position
        oy, ox = old_pos[:, 0], old_pos[:, 1]
        self.grid[batch_idx, oy, ox] = EMPTY
        # Update position
        self.animal_pos = move_mask * new_pos + (1-move_mask) * old_pos
        # Set animal on new position
        ny, nx = self.animal_pos[:, 0], self.animal_pos[:, 1]
        self.grid[batch_idx, ny, nx] = ANIMAL

        return reward

    def step(self):
        """
        Stochastically regrow food, if below self.food_density threshold AND outside of observation radius self.R
        """
        B, H, W = self.grid.shape
        target_food = int(H * W * self.food_density)
        # 1. Count current food cells in each grid (B,)
        food_count = (self.grid == FOOD).sum(dim=(1, 2))  # (B,)
        # 2. Determine which grids may grow new food (B,)
        may_grow = (food_count < target_food).long()  # (B,)
        # 3. Sample random (ys, xs) locations for each grid in batch
        ys = torch.randint(0, H, (B,), device=self.grid.device)
        xs = torch.randint(0, W, (B,), device=self.grid.device)
        pos = torch.stack([ys, xs], dim=1)  # (B, 2)
        # 3. Distance mask: avoid placing food within observation window
        d = periodic_distance(pos, self.animal_pos, H, W)  # (B,)
        may_grow *= (d > self.R).long()
        # 4. Look up cell type at those locations (B,)
        batch_idx = torch.arange(B, device=self.grid.device)
        cell_type = self.grid[batch_idx, ys, xs]  # (B,)
        # 5. Refine the growth mask: must be EMPTY
        may_grow = may_grow * (cell_type == EMPTY).long()  # (B,)
        # 6. Update cells using arithmetic masking (GPU-friendly)
        current = self.grid[batch_idx, ys, xs]
        updated = may_grow * FOOD + (1 - may_grow) * current
        self.grid[batch_idx, ys, xs] = updated

    def print(self, b=0, frame=True):
        print_grid(grid=self.grid[b], frame=frame)


# # ------------------ SANITY CHECK ------------------ #
# if __name__ == "__main__":
#     H, W, B = 7, 10, 3
#     device = 'cpu'
#     env = Terrain(height=H, width=W, batch_size=B, wall_density=0.15, food_density=0.0,
#                   device=device)
#     env.print_board(b=0)
#     env.food_density = 0.1
#     for step in range(10):
#         actions = torch.randint(0, 5, (B,), device=device)
#         rewards = env.step_animal(actions)
#         env.step_environment()
#         print("Action:", env.printer.action_to_str.get(actions[0].item(), 'UNKNOWN'), " Reward:", rewards[0].item())
#         env.print_board(b=0)
