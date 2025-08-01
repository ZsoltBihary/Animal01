import torch
from insect_model import InsectQModel01
from grid_world import GridWorld, GridSimulationResult
from insect import Insect
from core.simulator import Simulator
from core.deep_ql import DeepQLearning
from grid_renderer import GridRenderer
from pathlib import Path

# Define subfolder and filename
subfolder = Path("animations")
subfolder.mkdir(parents=True, exist_ok=True)  # Make sure the folder exists

video_path = subfolder / "roach02.mp4"

# Build up a grid world, with an insect.
B = 128
H = 17
W = 23
R = 3
food_r = 100.0
poison_r = -50.0
move_r = -1.0
food_d = 0.02
poison_d = 0.2
# Other specs
num_episodes = 10
steps_per_episode = 100
video_steps = 200
buffer_capacity = int(steps_per_episode * B * 1.5)
trainer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# trainer_device = torch.device("cpu")

print("Constructing GridWorld ...")  # =================================
world = GridWorld(B=B, H=H, W=W, R=R,
                  food_reward=food_r, poison_reward=poison_r, move_reward=move_r,
                  food_density=food_d, poison_density=poison_d)
observation_shape = (world.B, world.K, world.K)
state_shape = (world.B, world.C, world.K, world.K)
num_actions = world.A
print("Constructing InsectQModel01 ...")  # =================================
model = InsectQModel01(state_shape=state_shape, num_actions=num_actions)
print("Constructing Insect ...")  # =================================
insect = Insect(observation_shape=observation_shape, state_shape=state_shape, num_actions=num_actions,
                model=model)
print("Constructing DeepQLearning ...")  # =================================
dql_insect = DeepQLearning(world=world, animal=insect,
                           gamma=0.99,
                           num_episodes=num_episodes, steps_per_episode=steps_per_episode,
                           buffer_capacity=buffer_capacity,
                           num_epochs=1, batch_size=128, learning_rate=0.001,
                           trainer_device=trainer_device)
print("Deep Q Learning ...")  # =================================
dql_result = dql_insect.run()
dql_result.print()
print("Constructing Simulator ...")  # =================================
insect.epsilon = 0.0
sim_result = GridSimulationResult(capacity=video_steps, grid_shape=(H, W))
sim_insect = Simulator(world=world, animal=insect, result=sim_result, verbose=1)
print("Running simulation with the trained insect ...")  # =================================
sim_insect.run(n_steps=video_steps)
# Now we have a record of the simulation in sim_result  # TODO: Make result construction dynamic
print("Constructing GridRenderer ...")  # =================================
renderer = GridRenderer(world=world, cell_size=32)
# renderer.play_simulation(result=sim_result, delay_ms=300)
print("Animating simulation  ...")  # =================================
renderer.save_as_video(result=sim_result, video_path=video_path)
print("Animation saved in video file:", video_path)  # =================================
