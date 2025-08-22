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
video_path = subfolder / "roach10.mp4"
video_steps = 200
# Build up a grid world, with an insect.
B = 512
H = 29
W = 47
R = 3
food_r = 47.0
bar_r = -50.0
move_r = 0.0
food_d = 0.09
bar_d = 0.0
# Other specs
steps_per_episode = 50
num_episodes = 20

lr0, lr1 = 0.001, 0.0001
epsilon0, epsilon1 = 0.1, 0.02
temp0, temp1 = 0.04, 0.02

buffer_capacity = int(steps_per_episode * B * 10.0)
trainer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# trainer_device = torch.device("cpu")

print("Constructing GridWorld ...")  # =================================
world = GridWorld(B=B, H=H, W=W, R=R,
                  food_reward=food_r, bar_reward=bar_r, move_reward=move_r,
                  food_density=food_d, bar_density=bar_d)
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
                epsilon=epsilon0, temperature=temp0)

print("Constructing DeepQLearning ...")  # =================================
dql_insect = DeepQLearning(world=world, animal=insect,
                           gamma=0.99,
                           num_episodes=num_episodes, steps_per_episode=steps_per_episode,
                           buffer_capacity=buffer_capacity,
                           num_epochs=1, batch_size=128,
                           learning_rate0=lr0, learning_rate1=lr1,
                           epsilon0=epsilon0, epsilon1=epsilon1,
                           temp0=temp0,
                           temp1=temp1,
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
    "Temperature": f"{temp1:.2f}",
    "Food_reward": str(int(food_r)),
    "Max_bar_reward": str(int(bar_r)),
    "Move_reward": str(int(move_r)),
    "Food_density": f"{food_d:.2f}",
    "Barrier_density": f"{bar_d:.2f}",
    "Mean_reward": f"{sim_result.avg_rewards.mean().item():.1f}"
}
print(text_info)

renderer = GridRenderer(world=world, text_info=text_info, cell_size=48)
# renderer.play_simulation(result=sim_result, delay_ms=300)

print("Animating simulation  ...")  # =================================

renderer.save_as_video(result=sim_result, video_path=video_path)

print("Animation saved in video file:", video_path)  # =================================
