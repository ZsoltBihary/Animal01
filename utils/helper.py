import torch
from torch import Tensor

# Cell types
EMPTY, WALL, FOOD, ANIMAL = 0, 1, 2, 3
CELL_STR = {EMPTY: '   ', WALL: ' X ', FOOD: ' o ', ANIMAL: '(A)'}
N_CELL_TYPES = 4
# Actions
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
ACTION_STR = {STAY: 'STAY', UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}
N_ACTIONS = 5
# Rewards
FOOD_REWARD, WALL_PENALTY, MOVE_COST = 10.0, -10.0, -1.0


class ResolveMove:
    def __init__(self, H: int, W: int):
        self.H = H
        self.W = W
        self.delta_pos = torch.tensor([
            [0,  0],   # STAY
            [-1,  0],  # UP
            [1,  0],   # DOWN
            [0, -1],   # LEFT
            [0,  1],   # RIGHT
        ])

    def __call__(self, old_pos: Tensor, actions: Tensor) -> Tensor:
        """
        Args:
            old_pos: (B, 2) positions
            actions: (B,) action indices
        Returns:
            new_pos: (B, 2), wrapped on a toroidal grid
        """
        delta = self.delta_pos[actions]  # (B, 2)
        new_pos = old_pos + delta
        new_pos[:, 0] %= self.H  # wrap y
        new_pos[:, 1] %= self.W  # wrap x
        return new_pos


def print_grid(grid: Tensor, frame=True):
    """
    Args:
        grid: (H, W) , not batched!
        frame: boolean flag
    """
    H, W = grid.shape
    horizontal = " "
    if frame:
        for _ in range(W * 3):
            horizontal = horizontal + "-"
    print(horizontal)
    # Iterate through each row of the grid
    for row in grid:
        row_chars = [CELL_STR[value.item()] for value in row]
        row_string = ''.join(row_chars)
        if frame:
            print("|" + row_string + "|")
        else:
            print(row_string)
    print(horizontal)
