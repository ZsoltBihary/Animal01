import torch
from torch import Tensor

Observation = Tensor    # typically shape: (B, K, K) or similar
Action = Tensor         # shape: (B,)
Reward = Tensor         # shape: (B,)
Encoded = Tensor        # shape: (B, C+A, K, K)
BrainSlice = Tensor     # shape: (B, S, K, K)
BrainState = Tensor     # shape: (B, L, S, K, K)
QA = Tensor             # Q-values: (B, A)
RewardA = Tensor        # predicted rewards: (B, A)
ObservationA = Tensor   # predicted observations: (B, C, K, K)

CUDA_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    # === Cell type constants ===
    EMPTY: int = 0
    SEED: int = 1
    PLANT: int = 2
    FRUIT: int = 3
    BARR: int = 4
    ANIMAL: int = 5  # for rendering only

    num_cell: int = 5  # observable cell types (ANIMAL excluded)

    CELL_REW = torch.tensor([
        0.0,    # EMPTY
        10.0,   # SEED
        30.0,   # PLANT
        100.0,  # FRUIT
        -50.0,  # BARR
        0.0     # ANIMAL
    ])

    CELL_STR = {
        EMPTY: '   ',
        SEED: ' o ',
        PLANT: ' Y ',
        FRUIT: ' O ',
        BARR: ' + ',
        ANIMAL: '(A)',
    }

    # === Action constants ===
    STAY: int = 0
    UP: int = 1
    DOWN: int = 2
    LEFT: int = 3
    RIGHT: int = 4

    num_actions: int = 5

    delta_pos = torch.tensor([
        [0, 0],   # STAY
        [-1, 0],  # UP
        [1, 0],   # DOWN
        [0, -1],  # LEFT
        [0, 1],   # RIGHT
    ])

    ACTION_STR = {
        STAY: 'STAY',
        UP: 'UP',
        DOWN: 'DOWN',
        LEFT: 'LEFT',
        RIGHT: 'RIGHT',
    }

    def __init__(
            self,
            batch_size: int = 16,
            grid_height: int = 19,
            grid_width: int = 29,
            # obs_channels: int = num_cell,
            obs_radius: int = 3,

            # obs_size: int = None,
            brain_state_layers: int = 3,
            brain_state_channels: int = 30,
            head_mult: int = 2,
            model_on_GPU: bool = True,
            brain_on_GPU: bool = True,
            buffer_on_GPU: bool = False,

            buffer_reuse: float = 2.0,

            food_density: float = 0.06,
            barr_density: float = 0.2,
            food_growth_intensity: float = 0.2,
            world_reset_intensity: float = 0.02,

            gamma: float = 0.99,
            num_episodes: int = 10,
            steps_per_episode: int = 25,
            num_epochs: int = 1,
            learning_rate0: float = 0.001,
            learning_rate1: float = 0.0001,
            epsilon0: float = 0.1,
            epsilon1: float = 0.02,
            temperature0: float = 0.05,
            temperature1: float = 0.02
    ):
        self.gamma = gamma
        self.num_episodes, self.steps_per_episode = num_episodes, steps_per_episode
        self.num_epochs = num_epochs
        self.learning_rate0, self.learning_rate1 = learning_rate0, learning_rate1
        self.epsilon0, self.epsilon1 = epsilon0, epsilon1
        self.temperature0, self.temperature1 = temperature0, temperature1

        # === Density control ===
        self.food_density = food_density
        self.barr_density = barr_density
        self.empty_density = 1.0 - food_density - barr_density
        self.food_growth_intensity = food_growth_intensity
        self.world_reset_intensity = world_reset_intensity

        self.batch_size = batch_size
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.obs_channels = self.num_cell
        self.obs_radius = obs_radius
        self.obs_size = 2 * obs_radius + 1
        self.encoded_channels = self.num_actions + self.num_cell

        self.brain_state_layers = brain_state_layers
        self.brain_state_channels = brain_state_channels
        self.head_mult = head_mult

        self.model_device = CUDA_DEVICE if model_on_GPU else torch.device("cpu")
        self.trainer_device = CUDA_DEVICE if model_on_GPU else torch.device("cpu")
        self.brain_device = CUDA_DEVICE if brain_on_GPU else torch.device("cpu")
        self.buffer_device = CUDA_DEVICE if buffer_on_GPU else torch.device("cpu")

        self.buffer_reuse = buffer_reuse

    # def make_model(self):
    #     from reptile_world.sdqn_model import SDQNModel
    #     return SDQNModel(self)
    #
    # def make_reptile(self, brain_state=None):
    #     from reptile_world.reptile import Reptile
    #     model = self.make_model()
    #     return Reptile(self, model, brain_state)
    #
    # def make_world(self):
    #     from reptile_world.grid_world import GridWorld
    #     return GridWorld(self)

    # def __repr__(self):
    #     return (
    #         f"<Config batch_size={self.batch_size} obs_size={self.obs_size} "
    #         f"brain_state_layers={self.brain_state_layers} "
    #         f"brain_state_channels={self.brain_state_channels} "
    #         f"epsilon0={self.epsilon0} temperature0={self.temperature0} "
    #         f"model_device={self.model_device}>"
    #     )
