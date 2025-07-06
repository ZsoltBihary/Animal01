# run_simulation01.py
from core.animal import Amoeba
from core.terrain import Terrain
from core.world import World


B, H, W, R = 3, 7, 11, 2
wall_density, food_density = 0.1, 0.05
device = 'cpu'

terrain = Terrain.random(B=B, H=H, W=W, R=R,
                         wall_density=wall_density, food_density=food_density,
                         device=device)

animal = Amoeba()

world = World(terrain=terrain, animal=animal)

world.simulate(n_steps=10, verbose=2)
