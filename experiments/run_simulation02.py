# run_simulation02.py
from core.world import World

# rand_world = World.factory(
#     B=3, H=7, W=11, R=2, wall_dens=0.1, food_dens=0.1
# )
# rand_world.simulate(n_steps=10, verbose=1)

brain_world = World.factory(
    B=3, H=7, W=11, R=2, wall_dens=0.1, food_dens=0.1,
    with_brain=True,
    n_neurons=500, n_motors=50, n_connections=50,
    device='cpu'
)
brain_world.simulate(n_steps=10, verbose=2)
