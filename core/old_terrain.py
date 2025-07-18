# import torch
# from torch import Tensor
# from typing import Self
# from utils.helper import EMPTY, FOOD, POISON, ANIMAL, FOOD_REWARD, POISON_REWARD, MOVE_REWARD
# from utils.helper import STAY, C
# from utils.helper import ResolveMove, print_grid, periodic_distance
# import torch.nn.functional as F
#
#
# class Terrain:
#     def __init__(self, grid: Tensor, R: int, food_density: float, poison_density: float, animal_pos: Tensor):
#         """
#         Args:
#             grid: Tensor of shape (B, H, W), with cell type codes.
#             animal_pos: Tensor of shape (B, 2), animal positions (y, x)
#         """
#         self.grid = grid                # (B, H, W)
#         B, H, W = grid.shape
#         self.animal_pos = animal_pos    # (B, 2)
#         self.R = R  # observation radius
#         empty_density = 1.0 - food_density - poison_density
#         self.target_count = torch.tensor([empty_density, food_density, poison_density]) * H * W
#         self.move_fn = ResolveMove(H, W)      # helper function to interpret actions as moves in the grid
#
#     @classmethod
#     def random(cls, B: int, H: int, W: int, R: int,
#                food_density: float, poison_density: float,
#                device='cpu') -> Self:
#         """
#         Returns:
#             Terrain instance (batched) with random grids, animal_positions, food and poison cells.
#         """
#         # Initialize grid with EMPTY
#         grid = torch.full((B, H, W), EMPTY, dtype=torch.long, device=device)
#         # Assign FOOD and POISON
#         rand_vals = torch.rand((B, H, W), device=device)
#         grid[rand_vals < poison_density] = POISON
#         grid[(rand_vals >= poison_density) & (rand_vals < poison_density + food_density)] = FOOD
#         # Sample random animal positions uniformly
#         y = torch.randint(0, H, (B,), device=device)
#         x = torch.randint(0, W, (B,), device=device)
#         animal_pos = torch.stack([y, x], dim=1)  # (B, 2)
#         # Mark animal position in grid (overwrites whatever was there)
#         grid[torch.arange(B), y, x] = ANIMAL
#         return cls(grid, R, food_density, poison_density, animal_pos)
#
#     def get_observation(self) -> Tensor:
#         """
#         Returns observation window around the animal:
#         Output shape: (B, K, K), with K = 2 * R + 1
#         """
#         B, H, W = self.grid.shape
#         K = 2 * self.R + 1
#         # Pad grid toroidally
#         padded = F.pad(self.grid, (self.R, self.R, self.R, self.R), mode="circular")  # (B, H + 2R, W + 2R)
#         # Build index tensors
#         b = torch.arange(B).view(B, 1, 1)
#         y = self.animal_pos[:, 0].view(B, 1, 1) + torch.arange(K).view(1, K, 1)
#         x = self.animal_pos[:, 1].view(B, 1, 1) + torch.arange(K).view(1, 1, K)
#         # Use batched indexing to extract observation windows
#         observation = padded[b, y, x]  # (B, K, K)
#         return observation
#
#     def resolve_action(self, action: Tensor) -> Tensor:
#         """
#         Args:
#             action: Tensor of shape (B,)
#         Returns:
#             reward: Tensor of shape (B,)
#         """
#         B, H, W = self.grid.shape
#         batch_idx = torch.arange(B, device=action.device)
#         old_pos = self.animal_pos                # (B, 2)
#         new_pos = self.move_fn(old_pos, action)  # (B, 2)
#
#         # Gather cell types at new position
#         cell_at_new = self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1]]  # (B,)
#         # Movement reward for non-STAY actions (this is actually negative, moving is costly)
#         reward = (action != STAY) * MOVE_REWARD
#         # Add food reward where applicable
#         reward += (cell_at_new == FOOD) * FOOD_REWARD
#         # Add poison reward where applicable (this is actually negative reward)
#         reward += (cell_at_new == POISON) * POISON_REWARD
#
#         # Remove animal from old position
#         self.grid[batch_idx, old_pos[:, 0], old_pos[:, 1]] = EMPTY
#         # Set animal on new position
#         self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1]] = ANIMAL
#         # Update position
#         self.animal_pos = new_pos
#
#         return reward
#
#     def step(self):
#         """
#         Stochastically change one cell towards type that is farthest from its target,
#         if cell is outside of observation window.
#         """
#         B, H, W = self.grid.shape
#         # Compute current counts:
#         count = F.one_hot(self.grid, num_classes=C+1).sum(dim=(1, 2))[:, :-1]  # (B, C)
#         # Compute diff and select new cell per batch
#         diff_count = (self.target_count[None, :] - count) / (self.target_count[None, :] + 1.0)  # (B, C)
#         new_cell = diff_count.argmax(dim=1)  # (B,)
#         # Sample random (ys, xs) locations for each grid in batch
#         ys = torch.randint(0, H, (B,), device=self.grid.device)  # (B,)
#         xs = torch.randint(0, W, (B,), device=self.grid.device)  # (B,)
#         pos = torch.stack([ys, xs], dim=1)  # (B, 2)
#         # Distance mask: avoid change within observation window
#         d = periodic_distance(pos, self.animal_pos, H, W)  # (B,)
#         change = (d > self.R).long()
#         # Update cells using arithmetic masking (GPU-friendly)
#         batch_idx = torch.arange(B, device=self.grid.device)
#         current_cell = self.grid[batch_idx, ys, xs]
#         updated_cell = change * new_cell + (1 - change) * current_cell
#         self.grid[batch_idx, ys, xs] = updated_cell
#
#     def get_visible_mask(self):
#         # mask = self.grid[0]
#         B, H, W = self.grid.shape
#         xy = torch.arange(H*W)
#         y = xy // W
#         x = xy % W
#         pos = torch.stack((y, x), dim=1)
#         anim_pos = self.animal_pos[0][None, :]
#         d = periodic_distance(pos, anim_pos, H, W)
#         mask = (d <= self.R).reshape(H, W)
#         return mask
#
#     def print(self, b=0, frame=True):
#         print_grid(grid=self.grid[b], frame=frame)
#
#
# # ------------------ SANITY CHECK ------------------ #
# if __name__ == "__main__":
#     kitchen = Terrain.random(B=3, H=7, W=11, R=2,
#                              food_density=0.1, poison_density=0.1)
#     kitchen.print()
#     for _ in range(10):
#         kitchen.step()
#     kitchen.print()
#
#     mask = kitchen.get_visible_mask()
#     print(mask.int())
