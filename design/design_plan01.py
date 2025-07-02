import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------
# Cell Codes (you already defined this)
EMPTY, WALL, FOOD, ANIMAL = 0, 1, 2, 3
# -------------------------------------------

# === Brain ===
class Brain(nn.Module):
    def __init__(self, obs_shape, num_actions, hidden_dim=128):
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_shape[0] * obs_shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, obs):
        """Returns Q-values for each action"""
        return self.model(obs.float())

    def select_action(self, obs, eps=0.1):
        """Epsilon-greedy action selection"""
        q_values = self.forward(obs)
        if torch.rand(1).item() < eps:
            return torch.randint(0, self.num_actions, (obs.shape[0],), device=obs.device)
        return q_values.argmax(dim=1)

# === Animal ===
class Animal:
    def __init__(self, brain, env):
        self.brain = brain
        self.env = env

    def step(self, eps=0.1):
        obs = self.env.get_obs()  # Assume shape (B, H_obs, W_obs)
        actions = self.brain.select_action(obs, eps)
        obs_next, rewards = self.env.step_animal(actions)
        return obs, actions, rewards, obs_next

# === GridWorld ===
class GridWorld:
    def __init__(self, batch_size=32, height=11, width=11, food_density=0.05, device='cpu'):
        self.B = batch_size
        self.H = height
        self.W = width
        self.device = torch.device(device)
        self.food_density = food_density

        self.board = torch.zeros((self.B, self.H, self.W), dtype=torch.int32, device=self.device)
        self.animal_position = torch.zeros((self.B, 2), dtype=torch.long, device=self.device)
        self.reset()

    def reset(self):
        self.board.fill_(EMPTY)
        wall_mask = (torch.rand_like(self.board.float()) < 0.1)
        self.board[wall_mask] = WALL
        food_mask = (torch.rand_like(self.board.float()) < self.food_density)
        self.board[food_mask] = FOOD

        y = torch.randint(0, self.H, (self.B,), device=self.device)
        x = torch.randint(0, self.W, (self.B,), device=self.device)
        self.animal_position = torch.stack((y, x), dim=1)
        self.board[torch.arange(self.B), y, x] = ANIMAL

    def get_obs(self, view_size=5):
        pad = view_size // 2
        board_padded = torch.cat([self.board, self.board, self.board], dim=1)
        board_padded = torch.cat([board_padded, board_padded, board_padded], dim=2)

        obs = torch.zeros((self.B, view_size, view_size), dtype=torch.int32, device=self.device)
        for b in range(self.B):
            y, x = self.animal_position[b]
            y_pad = y + self.H
            x_pad = x + self.W
            obs[b] = board_padded[b, y_pad - pad:y_pad + pad + 1, x_pad - pad:x_pad + pad + 1]
        return obs

    def step_animal(self, actions):
        dy = torch.tensor([-1, 1, 0, 0, 0], device=self.device)
        dx = torch.tensor([0, 0, -1, 1, 0], device=self.device)

        batch = torch.arange(self.B, device=self.device)
        y, x = self.animal_position[:, 0], self.animal_position[:, 1]
        ny = (y + dy[actions]) % self.H
        nx = (x + dx[actions]) % self.W

        target = self.board[batch, ny, nx]
        reward = torch.zeros(self.B, device=self.device)

        reward[target == WALL] = -1.0
        reward[target == EMPTY] = -0.1
        reward[target == FOOD] = 1.0

        valid = (target != WALL)

        self.board[batch, y, x] = EMPTY
        new_y = torch.where(valid, ny, y)
        new_x = torch.where(valid, nx, x)

        self.board[batch, new_y, new_x] = ANIMAL
        self.animal_position = torch.stack((new_y, new_x), dim=1)

        return self.get_obs(), reward

    def step_environment(self):
        # Count food cells
        is_food = (self.board == FOOD)
        food_counts = is_food.sum(dim=(1, 2))
        target_food_count = int(self.H * self.W * self.food_density)

        mask_empty = (self.board == EMPTY)
        num_to_spawn = (food_counts < target_food_count)

        for b in torch.where(num_to_spawn)[0]:
            mask = mask_empty[b]
            indices = mask.nonzero(as_tuple=False)
            if indices.shape[0] == 0:
                continue
            idx = torch.randint(0, indices.shape[0], (1,))
            yx = indices[idx].squeeze(0)
            self.board[b, yx[0], yx[1]] = FOOD

# === Action Utils ===
def random_actions(batch_size, device='cpu'):
    return torch.randint(0, 5, (batch_size,), device=device)

def action_name(action):
    names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    return names[action.item()] if isinstance(action, torch.Tensor) else names[action]

# === Trainer / Simulation Manager ===
class Trainer:
    def __init__(self, env, animal, brain):
        self.env = env
        self.animal = animal
        self.brain = brain
        self.buffer = []  # Placeholder; replace with replay buffer

    def run_episode(self, steps=100):
        for t in range(steps):
            obs, actions, rewards, next_obs = self.animal.step()
            self.env.step_environment()
            self.buffer.append((obs, actions, rewards, next_obs))  # simplistic
            # Optionally train here

    def train_loop(self, episodes=100):
        for e in range(episodes):
            self.env.reset()
            self.run_episode()

# === Example usage ===
if False:  # Not meant to be run directly
    env = GridWorld(batch_size=32)
    brain = Brain(obs_shape=(5, 5), num_actions=5)
    animal = Animal(brain=brain, env=env)
    trainer = Trainer(env=env, animal=animal, brain=brain)
    trainer.train_loop(episodes=10)
