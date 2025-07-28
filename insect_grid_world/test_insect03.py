import torch
from insect_model import InsectQModel02
from grid_world import GridWorld, GridSimulationResult
from insect import Insect
from core.simulator import Simulator
from core.deep_ql import DeepQLearning
from grid_renderer import GridRenderer
from pathlib import Path

# Define subfolder and filename
subfolder = Path("animations")
subfolder.mkdir(parents=True, exist_ok=True)  # Make sure the folder exists
video_path = subfolder / "roach09.mp4"

# Build up a grid world, with an insect.
B = 256
H = 15
W = 21
R = 3
food_r = 100.0
poison_r = -100.0
move_r = -1.0
food_d = 0.02
poison_d = 0.3
# Other specs
num_episodes = 40
steps_per_episode = 100
video_steps = 200
epsilon, temperature = 0.1, 0.05
buffer_capacity = int(steps_per_episode * B * 1.2)
trainer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# trainer_device = torch.device("cpu")

print("Constructing GridWorld ...")  # =================================
world = GridWorld(B=B, H=H, W=W, R=R,
                  food_reward=food_r, poison_reward=poison_r, move_reward=move_r,
                  food_density=food_d, poison_density=poison_d)
num_cell = world.num_cell
num_actions = world.num_actions
K = world.K
C = world.num_cell + world.num_actions

observation_schema = {
    "image": (torch.Size([K, K]), torch.long),
    "last_action": (torch.Size([]), torch.long)  # scalar integer tensor
}
state_schema = {
    "x": (torch.Size([C, K, K]), torch.float32)
}
print("Constructing InsectQModel02 ...")  # =================================
model = InsectQModel02(state_schema=state_schema, num_actions=num_actions)

print("Constructing Insect ...")  # =================================
insect = Insect(observation_schema=observation_schema, state_schema=state_schema,
                num_cell=num_cell, num_actions=num_actions,
                model=model,
                epsilon=epsilon, temperature=temperature)

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
# Let us prepare some text info to show in animation video ...
text_info = {
    "Animal": insect.__class__.__name__,
    "Model": model.__class__.__name__,
    "Temperature": f"{temperature:.2f}",
    "Food_reward": str(int(food_r)),
    "Poison_reward": str(int(poison_r)),
    "Move_reward": str(int(move_r)),
    "Food_density": f"{food_d:.2f}",
    "Poison_density": f"{poison_d:.2f}",
    "Mean_reward": f"{sim_result.avg_rewards.mean().item():.1f}"
}
print(text_info)

renderer = GridRenderer(world=world, text_info=text_info, cell_size=32)
# renderer.play_simulation(result=sim_result, delay_ms=300)

print("Animating simulation  ...")  # =================================

renderer.save_as_video(result=sim_result, video_path=video_path)

print("Animation saved in video file:", video_path)  # =================================
