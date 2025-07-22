from insect_model import InsectQModel
from grid_world import GridWorld
from insect import Insect
from core.simulator import Simulator
# from core.animal import Amoeba

# Build up a grid world, run simulation with an untrained insect.
world = GridWorld(B=3, H=6, W=11, R=2,
                  food_reward=100.0, poison_reward=-100.0, move_reward=-1.0,
                  food_density=0.4, poison_density=0.2)
input_shape = (world.C, world.K, world.K)
num_actions = world.A
model = InsectQModel(input_shape=input_shape, num_actions=num_actions)
insect = Insect(model=model)
sim_insect = Simulator(world=world, animal=insect, verbose=2)
sim_insect.run(n_steps=10)

# # For comparison, continue simulation with an amoeba.
# amoeba = Amoeba(num_actions=world.A)
# sim_amoeba = Simulator(world=world, animal=amoeba, verbose=2)
# sim_amoeba.run(n_steps=10)
