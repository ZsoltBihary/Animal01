import torch
import torch.nn.functional as F
from typing import cast
from reptile.config import Config, Observation, Action, Encoded, BrainState, PredQ, PredReward, PredObservation
from reptile.sdqn_model import SDQNModel


class Reptile:
    def __init__(self, config: Config, model: SDQNModel, brain_state: BrainState = None):
        self.config = config
        self.model = model
        self.B = self.config.batch_size
        self.C = self.config.obs_channels
        self.K = self.config.obs_size
        self.A = self.config.num_actions
        self.L = self.config.brain_state_layers
        self.S = self.config.brain_state_channels
        self.epsilon = self.config.epsilon0
        self.temperature = self.config.temperature0

        # Set up brain state
        if brain_state is None:
            self.brain_state = torch.rand(self.B, self.L, self.S, self.K, self.K)  # random brain (B, L, S, K, K)
        else:
            self.brain_state = brain_state
        self.brain_state = self.brain_state.to(self.config.brain_state_device)

    def encode(self, observation: Observation, last_action: Action) -> Encoded:
        """
        Args:
            observation: Tensor(B, K, K)
            last_action: Tensor(B,)
        Returns:
            encoded: Tensor(B, C+A, K, K)
        """
        # === PROCESS OBSERVATION ===
        clamped = observation.clamp(min=0, max=self.C - 1)               # (B, K, K)
        one_hot_observation = F.one_hot(clamped, num_classes=self.C)     # (B, K, K, C)
        # Mask: keep only valid values in range [0, C-1]
        valid = (observation >= 0) & (observation < self.C)              # (B, K, K)
        mask = valid.unsqueeze(-1)                                       # (B, K, K, 1)
        one_hot_observation = one_hot_observation * mask                 # (B, K, K, C)

        # === PROCESS LAST ACTION ===
        one_hot_action = F.one_hot(last_action, num_classes=self.A)      # (B, A)
        # Expand one_hot_action to (B, K, K, A)
        one_hot_action = one_hot_action.view(self.B, 1, 1, self.A)       # (B, 1, 1, A)
        one_hot_action = one_hot_action.expand(-1, self.K, self.K, -1)   # (B, K, K, A)

        # === COMBINE ===
        encoded = torch.cat([one_hot_observation, one_hot_action], dim=-1)  # (B, K, K, C+A)
        # Permute to (B, C+A, K, K)
        encoded = encoded.permute(0, 3, 1, 2).float()    # (B, C+A, K, K)
        return encoded

    def predict(self, encoded: Encoded, brain_state: BrainState) -> (
            tuple)[PredQ, PredReward, PredObservation, BrainState]:
        # move input to model device
        encoded = encoded.to(self.config.model_device)
        brain_state = brain_state.to(self.config.model_device)
        pred_q, pred_r, pred_obs, new_brain_state = self.model.sdqn(encoded, brain_state)
        # move output, except brain_state back to cpu
        pred_q = pred_q.cpu()
        pred_r = pred_r.cpu()
        pred_obs = pred_obs.cpu()
        return pred_q, pred_r, pred_obs, new_brain_state

    def select_action(self, q_values: PredQ) -> Action:
        """
        Epsilon-soft strategy, with random policy selection based on Q(s, a) values
        """
        logits = q_values / self.temperature
        probs = F.softmax(logits, dim=-1)
        probs = (1.0 - self.epsilon) * probs + self.epsilon / probs.size(-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(1)
        return action

    def act(self, observation: Observation, last_action: Action) -> Action:
        encoded = self.encode(observation, last_action)
        q_a, pred_r, pred_obs, new_brain = self.predict(encoded, self.brain_state)
        self.brain_state = new_brain
        return self.select_action(q_a)

    # def act_dissected(self, observation: Observation, last_action: Action
    #                   ) -> tuple[Action, PredQ, PredReward, PredObservation, BrainState]:
    #     encoded = self.encode(observation, last_action)
    #     q_a, pred_r, pred_obs, new_brain = self.predict(encoded, self.brain_state)
    #     action = self.select_action(q_a)
    #     return action, q_a, pred_r, pred_obs, new_brain


# class Metazoan(Animal, ABC):
#     def __init__(self,
#                  observation_schema: Schema,
#                  state_schema: Schema,
#                  num_actions: int,
#                  model: MetazoanQModel,
#                  epsilon: float, temperature: float
#                  ):
#         super().__init__(observation_schema=observation_schema, num_actions=num_actions)
#         self.state_schema = state_schema
#         self.model = model.cuda()
#         self.epsilon = epsilon
#         self.temperature = temperature
#
#     @abstractmethod
#     def perceive(self, observation: Observation) -> State: ...
#
#     @abstractmethod
#     def estimate_q(self, state: State) -> Tensor: ...
#
#     def select_action(self, q_values: Tensor) -> Action:
#         """
#         Epsilon-soft strategy, with random policy selection based on Q(s, a) values
#         """
#         logits = q_values / self.temperature
#         probs = F.softmax(logits, dim=-1)
#         probs = (1.0 - self.epsilon) * probs + self.epsilon / probs.size(-1)
#         action = torch.multinomial(probs, num_samples=1).squeeze(1)
#         return action
#
#     def act(self, observation: Observation = None) -> Action:
#         state = self.perceive(observation).cuda()
#         q_values = self.estimate_q(state).cpu()
#         action = self.select_action(q_values)
#         return action
#
#
# # === Arthropods (can learn, no brain state) ===
# class Arthropod(Metazoan, ABC):
#     @abstractmethod
#     def encode(self, observation: Observation) -> State: ...
#
#     def perceive(self, observation: Observation) -> State:
#         state = self.encode(observation)
#         return state
#
#     def estimate_q(self, state: State) -> Tensor:
#         with torch.no_grad():
#             q_values = self.model(state.cuda()).cpu()  # simple feedforward
#         return q_values
#
#
# from tensordict import TensorDict
# # from torch import nn, Tensor
# import torch
# from torch import Tensor
# import torch.nn.functional as F
# from core.animal import Arthropod
# from core.q_model import MetazoanQModel
# # from insect_grid_world.insect_model import InsectQModel01
# from core.tensordict_helper import Schema, Observation, State, Action
#
#
# class Insect(Arthropod):
#     def __init__(self,
#                  observation_schema: Schema,
#                  state_schema: Schema,
#                  num_cell: int,
#                  num_actions: int,
#                  model: MetazoanQModel,
#                  epsilon=0.05, temperature=0.02):
#
#         super().__init__(
#             observation_schema=observation_schema,
#             state_schema=state_schema,
#             num_actions=num_actions,
#             model=model,
#             epsilon=epsilon, temperature=temperature
#         )
#         self.num_cell = num_cell
#
#     def encode(self, observation: Observation) -> State:
#         """
#         observation: TensorDict with keys:
#             - image: Tensor(B, K, K)
#             - last_action: Tensor(B,)
#         Returns:
#             state: TensorDict with key:
#             - x: Tensor(B, C+A, K, K)
#         """
#         B = observation.batch_size[0]
#         K = observation["image"].shape[-1]  # Assuming square (K, K)
#         # === PROCESS IMAGE ===
#         image = observation["image"]                                   # (B, K, K)
#         # Clamp image to valid range to avoid index errors in one_hot
#         clamped = image.clamp(min=0, max=self.num_cell - 1)            # (B, K, K)
#         one_hot_image = F.one_hot(clamped, num_classes=self.num_cell)  # (B, K, K, C)
#         # Mask: keep only valid values in range [0, num_cell-1]
#         valid = (image >= 0) & (image < self.num_cell)                 # (B, K, K)
#         mask = valid.unsqueeze(-1)                                     # (B, K, K, 1)
#         one_hot_image = one_hot_image * mask                           # (B, K, K, C)
#
#         # === PROCESS LAST ACTION ===
#         last_action = observation["last_action"]                               # (B, )
#         one_hot_action = F.one_hot(last_action, num_classes=self.num_actions)  # (B, A)
#         # Expand one_hot_action to (B, K, K, A)
#         one_hot_action = one_hot_action.view(B, 1, 1, self.num_actions)  # (B, 1, 1, A)
#         one_hot_action = one_hot_action.expand(-1, K, K, -1)  # (B, K, K, A)
#
#         # === COMBINE ===
#         combined = torch.cat([one_hot_image, one_hot_action], dim=-1)  # (B, K, K, C+A)
#         # Permute to (B, C+A, K, K)
#         combined = combined.permute(0, 3, 1, 2).float()
#         state = TensorDict(source={"x": combined}, batch_size=[combined.size(0)])
#         return state
