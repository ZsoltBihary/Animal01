import torch
from insect_model import InsectQModel
from grid_world import GridWorld
from insect import Insect
from core.simulator import Simulator
from core.deep_ql import DeepQLearning

# Build up a grid world, with an insect.
B = 400
H = 15
W = 15
R = 2
food_r = 100.0
poison_r = -100.0
move_r = -1.0
food_d = 0.1
poison_d = 0.2
# Other specs
num_episodes = 50
steps_per_episode = 100
trainer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

world = GridWorld(B=B, H=H, W=W, R=R,
                  food_reward=food_r, poison_reward=poison_r, move_reward=move_r,
                  food_density=food_d, poison_density=poison_d)

observation_shape = (world.B, world.K, world.K)
state_shape = (world.B, world.C, world.K, world.K)
num_actions = world.A

model = InsectQModel(state_shape=state_shape, num_actions=num_actions)
insect = Insect(observation_shape=observation_shape, state_shape=state_shape, num_actions=num_actions,
                model=model)

# Run simulation with an untrained insect.
sim_insect = Simulator(world=world, animal=insect, verbose=2)
sim_insect.run(n_steps=10)

print("Constructing DeepQLearning object ...")
dqn_insect = DeepQLearning(world=world, animal=insect,
                           gamma=0.99,
                           num_episodes=num_episodes, steps_per_episode=steps_per_episode,
                           buffer_capacity=2*steps_per_episode*B,
                           num_epochs=1, batch_size=128, learning_rate=0.001,
                           trainer_device=trainer_device)
print("DeepQLearning object constructed.")
dqn_insect.run()

# Run simulation with the trained insect.
insect.epsilon = 0.0
sim_insect.run(n_steps=10)
