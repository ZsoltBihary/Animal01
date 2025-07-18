from insect_model import InsectQModel
from grid_world import GridWorld
from insect import Insect
from core.simulator import Simulator

world = GridWorld(B=3, H=6, W=11, R=2,
                  food_reward=100.0, poison_reward=-100.0, move_reward=-1.0,
                  food_density=0.4, poison_density=0.2)
input_shape = (world.C, world.K, world.K)
num_actions = world.A
model = InsectQModel(input_shape=input_shape, num_actions=num_actions)
animal = Insect(model=model)
sim = Simulator(world=world, animal=animal, verbose=2)
sim.run(n_steps=100)
