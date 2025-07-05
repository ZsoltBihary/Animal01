# import torch
# from torch import Tensor
# import torch.nn.functional as F
from core.policy import RandomPolicy
from core.animal import Animal
from core.terrain import Terrain
from core.world import World


B, H, W, OR = 3, 7, 11, 2
device = 'cpu'

terrain = Terrain.random(B=B, H=H, W=W, wall_density=0.1, food_density=0.05, device=device)

policy = RandomPolicy()

animal = Animal(policy=policy, obs_radius=OR)

world = World(terrain=terrain, animal=animal)

world.simulate(n_steps=10, verbose=2)
