# run_simulation02.py
from core.terrain import Terrain
from core.world import World
from core.animal import Amoeba, Insect


# Amoeba is hunting in a pond
pond = Terrain.random(B=3, H=7, W=7, R=1,
                      food_density=0.15, poison_density=0.05)
amoeba = Amoeba()
pond_world = World(terrain=pond, animal=amoeba)
pond_world.simulate(n_steps=10, verbose=1)


# Roach is hunting for crumbles in a kitchen
kitchen = Terrain.random(B=3, H=7, W=11, R=2,
                         food_density=0.1, poison_density=0.1)
roach = Insect(main_ch=16)
crumb_hunt = World(terrain=kitchen, animal=roach)
crumb_hunt.simulate(n_steps=10, verbose=2)
