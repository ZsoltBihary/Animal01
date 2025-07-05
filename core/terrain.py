import torch
from torch import Tensor
from typing import Self
from utils.helper import EMPTY, WALL, FOOD, ANIMAL, FOOD_REWARD, WALL_PENALTY, MOVE_COST, STAY
from utils.helper import ResolveMove, print_grid
import torch.nn.functional as F
# from utils.helper import N_CELL_TYPES  # Includes EMPTY=0


class Terrain:
    def __init__(self, grid: Tensor, animal_pos: Tensor):
        """
        Args:
            grid: Tensor of shape (B, H, W), with cell type codes.
            animal_pos: Tensor of shape (B, 2), animal positions (y, x)
        """
        self.grid = grid                # (B, H, W)
        self.animal_pos = animal_pos    # (B, 2)
        B, H, W = grid.shape
        self.move_fn = ResolveMove(H, W)

    @classmethod
    def random(cls, B: int, H: int, W: int,
               wall_density: float = 0.2, food_density: float = 0.1,
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
        return cls(grid, animal_pos)

    def get_observation(self, R: int) -> Tensor:
        """
        Arg R: observation radius
        Returns local window around the animal:
        Output shape: (B, K, K), with K = 2*R + 1
        """
        B, H, W = self.grid.shape
        K = 2 * R + 1
        # Pad grid toroidally
        padded = F.pad(self.grid, (R, R, R, R), mode="circular")  # (B, H + 2R, W + 2R)

        pos = self.animal_pos  # (B, 2)
        # Build index tensors
        y = pos[:, 0].view(B, 1, 1) + torch.arange(K).view(1, K, 1)
        x = pos[:, 1].view(B, 1, 1) + torch.arange(K).view(1, 1, K)
        b = torch.arange(B).view(B, 1, 1)
        # Use batched indexing to extract local windows
        local = padded[b, y, x]  # (B, K, K)
        # # One-hot encode (skip EMPTY = 0)
        # onehot = F.one_hot(local, num_classes=N_CELL_TYPES)[..., 1:]  # (B, K, K, C)
        # obs = onehot.permute(0, 3, 1, 2).float()  # (B, C, K, K)
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

    def print(self, b=0, frame=True):
        print_grid(grid=self.grid[b], frame=frame)

#     def step_environment(self):
#         """
#         Vectorized environment step: adds one food to boards where food count < target.
#         Assumes periodic boundary and no masking other than EMPTY cells.
#         """
#         B, H, W = self.B, self.H, self.W
#         target_food = H * W * self.food_density
#         # Count current FOOD cells per board
#         food_counts = (self.board == FOOD).flatten(start_dim=1).sum(dim=1)  # shape: (B,)
#         need_food = food_counts < target_food  # shape: (B,)
#         if not need_food.any():
#             return  # No board needs food
#         # Mask for EMPTY cells
#         empty_mask = (self.board == EMPTY)  # shape: (B, H, W)
#         flat_mask = empty_mask.flatten(start_dim=1)  # shape: (B, H*W)
#         # Convert mask to selection weights
#         weights = flat_mask.float()  # 1.0 where empty, 0.0 otherwise
#         # weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # normalize
#         weights = weights / weights.sum(dim=1, keepdim=True)  # normalize
#         # Sample indices from each board
#         indices = torch.multinomial(weights, num_samples=1).squeeze(1)  # shape: (B,)
#         # Only update boards that need food
#         selected_indices = indices[need_food]  # flattened indices
#         boards_to_update = torch.nonzero(need_food, as_tuple=False).squeeze(1)  # board indices
#
#         y = selected_indices // W
#         x = selected_indices % W
#         self.board[boards_to_update, y, x] = FOOD
#
#     def print_board(self, b, frame=True):
#         self.printer.print_board(self.board[b, ...], frame)
#
#
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
