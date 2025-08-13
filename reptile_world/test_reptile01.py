from reptile_world.config import Config   # , Action, Reward
from reptile_world.grid_world import GridWorld
from reptile_world.sdqn_model import SDQNModel
from reptile_world.reptile import Reptile
from reptile_world.simulator import Simulator, SimulationResult
from reptile_world.grid_renderer import GridRenderer
from pathlib import Path


# Define subfolder and filename
subfolder = Path("animations")
subfolder.mkdir(parents=True, exist_ok=True)  # Make sure the folder exists
video_path = subfolder / "tortoise01.mp4"
video_steps = 200

config = Config()
world = GridWorld(config)
model = SDQNModel(config)
animal = Reptile(config=config, model=model)
# n_step1, n_step2 = 10, 120
result = SimulationResult(conf=config, capacity=video_steps)
sim = Simulator(conf=config,
                world=world,
                animal=animal,
                result=result,
                verbose=3)
sim.run(n_steps=video_steps)
# sim.run(n_steps=n_step2)
print(sim.result.last_actions)
rend = GridRenderer(conf=config)
rend.play_simulation(result=result, delay_ms=100)
rend.save_as_video(result=result, video_path=video_path)
