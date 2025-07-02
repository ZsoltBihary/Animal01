import torch
from helper import EMPTY, WALL, FOOD, ANIMAL  # Cell codes
from print_world import PrintWorld


class GridWorld:
    def __init__(self, batch_size=3, height=7, width=11,
                 wall_density=0.1, food_density=0.05, device='cpu'):

        self.B = batch_size
        self.H = height
        self.W = width
        self.wall_density = wall_density
        self.food_density = food_density
        self.device = torch.device(device)
        self.printer = PrintWorld()
        # Define board state and agent positions
        self.board = torch.zeros((self.B, self.H, self.W), dtype=torch.int32, device=self.device)
        self.animal_position = torch.zeros((self.B, 2), dtype=torch.long, device=self.device)

        self.reset()

    def reset(self):
        self.board.fill_(EMPTY)
        # Add walls randomly
        wall_mask = (torch.rand_like(self.board.float()) < self.wall_density)
        self.board[wall_mask] = WALL
        # Add food randomly
        food_mask = (torch.rand_like(self.board.float()) < self.food_density)
        self.board[food_mask] = FOOD

        y = torch.randint(0, self.H, (self.B,), device=self.device)
        x = torch.randint(0, self.W, (self.B,), device=self.device)
        self.animal_position = torch.stack((y, x), dim=1)

        # Overwrite target positions with AGENT
        self.board[torch.arange(self.B), y, x] = ANIMAL

    def step_animal(self, actions: torch.Tensor):
        """
        Applies animal actions and returns rewards.
        actions: (B,) tensor with values in {0=stay, 1=up, 2=down, 3=left, 4=right}
        """
        actions = actions.to(self.device)
        B, H, W = self.B, self.H, self.W

        # Direction deltas: stay, up, down, left, right
        dy = torch.tensor([0, -1, 1, 0, 0], device=self.device)
        dx = torch.tensor([0, 0, 0, -1, 1], device=self.device)

        y = self.animal_position[:, 0]
        x = self.animal_position[:, 1]
        ny = (y + dy[actions]) % H
        nx = (x + dx[actions]) % W

        # Gather target cell content
        target_cell = self.board[torch.arange(B), ny, nx]

        # Determine rewards
        reward = torch.zeros(B, device=self.device)
        moved = actions != 0
        hit_wall = (target_cell == WALL) & moved
        to_food = (target_cell == FOOD) & moved
        to_empty = (target_cell == EMPTY) & moved

        reward[hit_wall] = -1.0
        reward[to_food] = 1.0
        reward[to_empty] = -0.1
        # No reward for staying

        # Compute new positions (only if not hitting wall)
        can_move = ~hit_wall & moved
        final_y = torch.where(can_move, ny, y)
        final_x = torch.where(can_move, nx, x)

        # Update board: clear old positions
        self.board[torch.arange(B), y, x] = EMPTY

        # If moved to food, remove food
        self.board[to_food, ny[to_food], nx[to_food]] = EMPTY

        # Place animals at new positions
        self.board[torch.arange(B), final_y, final_x] = ANIMAL

        # Update internal animal position
        self.animal_position[:, 0] = final_y
        self.animal_position[:, 1] = final_x

        return reward

    def step_environment(self):
        """
        Vectorized environment step: adds one food to boards where food count < target.
        Assumes periodic boundary and no masking other than EMPTY cells.
        """
        B, H, W = self.B, self.H, self.W
        target_food = H * W * self.food_density
        # Count current FOOD cells per board
        food_counts = (self.board == FOOD).flatten(start_dim=1).sum(dim=1)  # shape: (B,)
        need_food = food_counts < target_food  # shape: (B,)
        if not need_food.any():
            return  # No board needs food
        # Mask for EMPTY cells
        empty_mask = (self.board == EMPTY)  # shape: (B, H, W)
        flat_mask = empty_mask.flatten(start_dim=1)  # shape: (B, H*W)
        # Convert mask to selection weights
        weights = flat_mask.float()  # 1.0 where empty, 0.0 otherwise
        # weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # normalize
        weights = weights / weights.sum(dim=1, keepdim=True)  # normalize
        # Sample indices from each board
        indices = torch.multinomial(weights, num_samples=1).squeeze(1)  # shape: (B,)
        # Only update boards that need food
        selected_indices = indices[need_food]  # flattened indices
        boards_to_update = torch.nonzero(need_food, as_tuple=False).squeeze(1)  # board indices

        y = selected_indices // W
        x = selected_indices % W
        self.board[boards_to_update, y, x] = FOOD

    def print_board(self, b, frame=True):
        self.printer.print_board(self.board[b, ...], frame)


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    H, W, B = 7, 10, 3
    device = 'cpu'
    env = GridWorld(height=H, width=W, batch_size=B, wall_density=0.15, food_density=0.0,
                    device=device)
    env.print_board(b=0)
    env.food_density = 0.1
    for step in range(10):
        actions = torch.randint(0, 5, (B,), device=device)
        rewards = env.step_animal(actions)
        env.step_environment()
        print("Action:", env.printer.action_to_str.get(actions[0].item(), 'UNKNOWN'), " Reward:", rewards[0].item())
        env.print_board(b=0)
