import torch
import torch.nn.functional as F
from torch import Tensor
from reptile_world.config import Config, Action, Observation, Reward
from colorama import init, Fore, Style
init(autoreset=True)  # autoreset=True automatically resets after each print


class GridWorld:
    def __init__(self, conf: Config):
        self.conf = conf
        # === Consume configuration parameters ===
        self.EMPTY, self.SEED, self.PLANT, self.FRUIT, self.BARR, self.ANIMAL \
            = conf.EMPTY, conf.SEED, conf.PLANT, conf.FRUIT, conf.BARR, conf.ANIMAL
        self.C = conf.num_cell
        self.CELL_REW = conf.CELL_REW  # Rewards depend on the cell type the animal moves to
        self.CELL_STR = conf.CELL_STR

        self.STAY, self.UP, self.DOWN, self.LEFT, self.RIGHT \
            = conf.STAY, conf.UP, conf.DOWN, conf.LEFT, conf.RIGHT
        self.A = conf.num_actions
        self.delta_pos = conf.delta_pos
        self.ACTION_STR = conf.ACTION_STR

        self.B, self.H, self.W, self.R, self.K \
            = conf.batch_size, conf.grid_height, conf.grid_width, conf.obs_radius, conf.obs_size

        # === Food growth ===
        self.p_s2p = conf.food_growth_intensity  # SEED -> PLANT
        self.p_p2f = conf.food_growth_intensity  # PLANT -> FRUIT
        self.p_f2e = conf.food_growth_intensity  # FRUIT -> EMPTY

        # === Density control ===
        self.food_density = conf.food_density
        self.barr_density = conf.barr_density
        self.empty_density = conf.empty_density
        # ANYTHING -> EMPTY
        self.p_a2e = (conf.world_reset_intensity * conf.empty_density -
                      conf.food_growth_intensity * conf.food_density / 3.0)
        # ANYTHING -> BARR (adjust based on density if desired)
        self.p_a2b = conf.world_reset_intensity * conf.barr_density
        # ANYTHING -> SEED (adjust based on density if desired)
        self.p_a2s = (conf.world_reset_intensity + conf.food_growth_intensity) * conf.food_density / 3.0

        # === Initialize class attributes ===
        self.grid, self.animal_pos, self.last_action = self.random_start()

    def random_start(self) -> tuple[Tensor, Tensor, Action]:
        """ Return batched random grids, batched animal_positions """
        # Initialize with EMPTY cells
        grid = torch.full((self.B, self.H, self.W), self.EMPTY, dtype=torch.long)  # (B, H, W)
        # Assign FOOD and BARRIER
        rand_vals = torch.rand((self.B, self.H, self.W))
        rand_food_index = torch.randint(self.SEED, self.FRUIT + 1, size=(self.B, self.H, self.W), dtype=torch.long)
        # rand_food_index = torch.randint_like(rand_vals, low=self.SEED, high=self.FRUIT+1, dtype=torch.long)
        grid[rand_vals < self.food_density] = rand_food_index[rand_vals < self.food_density]
        grid[(rand_vals >= self.food_density) & (rand_vals < self.food_density + self.barr_density)] = self.BARR
        # Assign random animal positions
        y = torch.randint(0, self.H, (self.B,))
        x = torch.randint(0, self.W, (self.B,))
        animal_pos = torch.stack([y, x], dim=1)  # (B, 2)
        # Mark animal position in grid (overwrites whatever was there)
        grid[torch.arange(self.B), y, x] = self.ANIMAL
        # last_action = torch.full((self.B, ), self.STAY, dtype=torch.long)
        last_action = self.zero_action()  # (B, )
        return grid, animal_pos, last_action

    def zero_action(self) -> Action:
        action = torch.full(size=(self.B, ), fill_value=self.STAY)
        return action

    def resolve_action(self, action: Action) -> tuple[Observation, Reward]:
        """
        Returns:
            observation: Observation: Tensor (B, K, K)
            reward: Reward: Tensor (B,)
        """
        self.last_action = action
        # Determine animal's new position
        batch_idx = torch.arange(self.B)
        old_pos = self.animal_pos                   # (B, 2)
        new_pos = old_pos + self.delta_pos[self.last_action]  # (B, 2)
        new_pos[:, 0] %= self.H  # wrap y
        new_pos[:, 1] %= self.W  # wrap x
        # Gather cell types at new position, get reward
        cell_at_new = self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1]]  # (B, )
        reward = self.CELL_REW[cell_at_new]  # (B, )
        # Remove animal from old position
        self.grid[batch_idx, old_pos[:, 0], old_pos[:, 1]] = self.EMPTY
        # Set animal on new position
        self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1]] = self.ANIMAL
        # Update position
        self.animal_pos = new_pos
        # World step (random food ripening, density control)
        self.step()
        # Observe
        observation = self.get_observation()
        return observation, reward

    def get_observation(self) -> Observation:
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

    def step(self):

        # === Food growth ===
        rand_food = torch.rand((self.B, self.H, self.W))

        # Compute masks from current grid before modifying
        mask_s2p = (self.grid == self.SEED) & (rand_food < self.p_s2p)
        mask_p2f = (self.grid == self.PLANT) & (rand_food < self.p_p2f)
        mask_f2e = (self.grid == self.FRUIT) & (rand_food < self.p_f2e)

        # Apply changes in order
        self.grid[mask_s2p] = self.PLANT
        self.grid[mask_p2f] = self.FRUIT
        self.grid[mask_f2e] = self.EMPTY

        # === Density control ===
        far_mask = self.get_far_mask(R_protect=self.R+1)  # Slightly larger protected region than obs window

        rand_empty = torch.rand((self.B, self.H, self.W))
        mask_empty = far_mask & (rand_empty < self.p_a2e)

        rand_barr = torch.rand((self.B, self.H, self.W))
        mask_barr = far_mask & (rand_barr < self.p_a2b)

        rand_seed = torch.rand((self.B, self.H, self.W))
        mask_seed = far_mask & (rand_seed < self.p_a2s)

        # Apply density control transitions
        self.grid[mask_empty] = self.EMPTY
        self.grid[mask_barr] = self.BARR
        self.grid[mask_seed] = self.SEED

    def get_far_mask(self, R_protect: int) -> torch.Tensor:
        """
        Returns a boolean mask of shape (B, H, W) where True means the cell is
        at least R_protect away (Chebyshev, periodic) from the animal position.

        Args:
            R_protect: protective radius (Chebyshev distance)

        Returns:
            far_mask: (B, H, W) boolean tensor
        """
        # Make coordinate grids for all cell positions (H, W)
        ys = torch.arange(self.H, device=self.animal_pos.device)
        xs = torch.arange(self.W, device=self.animal_pos.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        # Expand to batch dimension
        grid_y = grid_y.unsqueeze(0).expand(self.B, -1, -1)  # (B, H, W)
        grid_x = grid_x.unsqueeze(0).expand(self.B, -1, -1)  # (B, H, W)
        # Animal positions
        ay = self.animal_pos[:, 0].view(self.B, 1, 1)  # (B, 1, 1)
        ax = self.animal_pos[:, 1].view(self.B, 1, 1)  # (B, 1, 1)
        # Raw distances
        dy = torch.abs(grid_y - ay)
        dx = torch.abs(grid_x - ax)
        # Toroidal adjustment
        dy = torch.minimum(dy, self.H - dy)
        dx = torch.minimum(dx, self.W - dx)
        # Chebyshev distance
        dist = torch.maximum(dy, dx)  # (B, H, W)
        # Mask: True if distance > R_protect
        return dist > R_protect

    def print_world(self):
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

    def print_action_reward(self, reward: Reward) -> None:
        print("Action:", self.ACTION_STR[self.last_action[0].item()],
              "  Reward:", reward[0].item())

    def print_color_world(self):

        # COLOR_MAP = {
        #     EMPTY: Fore.RESET,
        #     SEED: Fore.YELLOW,
        #     PLANT: Fore.GREEN,
        #     FRUIT: Fore.MAGENTA ,
        #     BARR: Fore.WHITE + ,
        #     ANIMAL: Fore.CYAN + Style.BRIGHT,
        # }

        color_map = {
            self.EMPTY: Fore.RESET,       # no color, default terminal color
            self.SEED: Fore.YELLOW,       # yellow dot for seed
            self.PLANT: Fore.LIGHTGREEN_EX + Style.BRIGHT,       # green plant
            self.FRUIT: Fore.LIGHTCYAN_EX + Style.BRIGHT,     # magenta fruit
            self.BARR: Fore.RED + Style.DIM,         # cyan barrier/star
            self.ANIMAL: Fore.LIGHTWHITE_EX,  # bright red animal
        }

        horizontal = " "
        for _ in range(self.W * 3):
            horizontal += "─"
        print(horizontal)

        for row in self.grid[0]:
            row_chars = []
            for value in row:
                cell_str = self.CELL_STR[value.item()]
                color = color_map.get(value.item(), Fore.RESET)
                # Wrap the cell string in color codes
                colored_cell = f"{color}{cell_str}{Style.RESET_ALL}"
                row_chars.append(colored_cell)
            row_string = ''.join(row_chars)
            print("│" + row_string + "│")

        print(horizontal)


# ------------------ SANITY CHECK ------------------ #
if __name__ == "__main__":
    config = Config()
    world = GridWorld(config)
    world.print_color_world()
    for _ in range(100):
        rand_action = torch.randint(world.UP, world.RIGHT, size=(world.B, ), dtype=torch.long)
        # action = world.zero_action()
        observation, reward = world.resolve_action(rand_action)
        world.print_action_reward(reward)
        world.print_color_world()
