from tensordict import TensorDict
# from torch import nn, Tensor
import torch
from torch import Tensor
import torch.nn.functional as F
from core.animal import Arthropod
from core.q_model import MetazoanQModel
# from insect_grid_world.insect_model import InsectQModel01
from core.tensordict_helper import Schema, Observation, State, Action


class Insect(Arthropod):
    def __init__(self,
                 observation_schema: Schema,
                 state_schema: Schema,
                 num_cell: int,
                 num_actions: int,
                 model: MetazoanQModel,
                 epsilon=0.05, temperature=0.02):

        super().__init__(
            observation_schema=observation_schema,
            state_schema=state_schema,
            num_actions=num_actions,
            model=model,
            epsilon=epsilon, temperature=temperature
        )
        self.num_cell = num_cell

    def encode(self, observation: Observation) -> State:
        """
        observation: TensorDict with keys:
            - image: Tensor(B, K, K)
            - last_action: Tensor(B,)
        Returns:
            state: TensorDict with key:
            - x: Tensor(B, C+A, K, K)
        """
        B = observation.batch_size[0]
        K = observation["image"].shape[-1]  # Assuming square (K, K)
        # === PROCESS IMAGE ===
        image = observation["image"]                                   # (B, K, K)
        # Clamp image to valid range to avoid index errors in one_hot
        clamped = image.clamp(min=0, max=self.num_cell - 1)            # (B, K, K)
        one_hot_image = F.one_hot(clamped, num_classes=self.num_cell)  # (B, K, K, C)
        # Mask: keep only valid values in range [0, num_cell-1]
        valid = (image >= 0) & (image < self.num_cell)                 # (B, K, K)
        mask = valid.unsqueeze(-1)                                     # (B, K, K, 1)
        one_hot_image = one_hot_image * mask                           # (B, K, K, C)

        # === PROCESS LAST ACTION ===
        last_action = observation["last_action"]                               # (B, )
        one_hot_action = F.one_hot(last_action, num_classes=self.num_actions)  # (B, A)
        # Expand one_hot_action to (B, K, K, A)
        one_hot_action = one_hot_action.view(B, 1, 1, self.num_actions)  # (B, 1, 1, A)
        one_hot_action = one_hot_action.expand(-1, K, K, -1)  # (B, K, K, A)

        # === COMBINE ===
        combined = torch.cat([one_hot_image, one_hot_action], dim=-1)  # (B, K, K, C+A)
        # Permute to (B, C+A, K, K)
        combined = combined.permute(0, 3, 1, 2).float()
        state = TensorDict(source={"x": combined}, batch_size=[combined.size(0)])
        return state

    # def encode(self, observation: TensorDict) -> TensorDict:
    #     """
    #     observation: TensorDict with keys:
    #         - image: Tensor(B, 1, K, K)
    #         - last_action: Tensor(B,)
    #     Returns:
    #         state: TensorDict with key:
    #         - x: Tensor(B, C, K, K)
    #     """
    #     B = observation.batch_size[0]
    #
    #     # Extract image and pass through CNN
    #     image = observation["image"].to(next(self.encoder.parameters()).device)
    #     x = self.encoder(image)
    #
    #     # Create state TensorDict
    #     state = TensorDict(
    #         {"x": x},
    #         batch_size=[B]
    #     )
    #     return state


# from torch import Tensor
# import torch.nn.functional as F
# from core.animal import Arthropod
# from core.q_model import MetazoanQModel
# # from insect_grid_world.insect_model import InsectQModel01
#
#
# class Insect(Arthropod):
#     def __init__(self, observation_shape: tuple[int, ...], state_shape: tuple[int, ...], num_actions: int,
#                  model: MetazoanQModel, epsilon=0.1, temperature=0.02):
#
#         super().__init__(observation_shape=observation_shape, state_shape=state_shape, num_actions=num_actions,
#                          model=model, epsilon=epsilon, temperature=temperature)
#         # observation_shape: (B, K, K)
#         # state_shape: (B, C, K, K)
#         self.C = state_shape[1]
#
#     def encode(self, observation: Tensor) -> Tensor:
#         """
#         observation: (B, K, K) with ints in [0, ?], but we want to encode only [0, C-1]
#         """
#         # Mask: keep only valid values in range [0, C - 1]
#         valid = (observation >= 0) & (observation < self.C)  # shape: (B, K, K)
#         mask = valid.unsqueeze(-1)  # shape: (B, K, K, 1)
#
#         # Clamp to valid range to avoid index errors in one_hot
#         clamped = observation.clamp(min=0, max=self.C - 1)  # shape: (B, K, K)
#
#         # One-hot encode and mask invalid entries
#         one_hot = F.one_hot(clamped, num_classes=self.C)  # shape: (B, K, K, C)
#         one_hot = one_hot * mask  # zero out invalid entries
#
#         # Reorder to (B, C, K, K) for CNN input
#         state = one_hot.permute(0, 3, 1, 2).float()  # shape: (B, C, K, K)
#         return state
