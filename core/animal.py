# animal.py

# Animal
# └── Protozoan          ← includes agents that can act, but cannot learn
# │   ├────── * Sponge
# │   └────── * Amoeba
# └── Metazoan           ← includes all Q-learning-capable agents
#     ├── Arthropod      ← includes stateless Q-learning-capable agents
#     │   └── * Insect
#     └── Vertebrate     ← includes memory/state-based Q-learning-capable agents
#         ├── * Reptile
#         └── * Mammal

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Type, Optional
from core.q_model import MetazoanQModel, VertebrateQModel
from core.types import Observation, State, Action


# === Base Animal Class ===
class Animal(ABC):
    def __init__(self, observation_shape: tuple[int, ...], num_actions: int):
        self.observation_shape = observation_shape
        self.num_actions = num_actions

    @abstractmethod
    def act(self, observation: Observation = None) -> Action: ...

# # === Protozoans (can not learn, only act) ===
# class Protozoan(Animal, ABC):
#     def __init__(self, num_actions: int):
#
#
# class Sponge(Protozoan):
#     def act(self, observation=None):
#         # B = observation.shape[0]
#         action = torch.zeros(self.B, dtype=torch.long)
#         return action
#
#
# class Amoeba(Protozoan):
#     def act(self, observation=None):
#
#         B = observation.shape[0]
#         action = torch.randint(low=0, high=self.A, size=(B,))
#         return action


class Animala(ABC):
    def __init__(
        self,
        obs_dim: int,
        model_class: Type[nn.Module],
        model_kwargs: Optional[dict] = None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        self.model = model_class(obs_dim=obs_dim, **model_kwargs)


# === Metazoans (can learn) ===
class Metazoan(Animal, ABC):
    state_shape: tuple

    def __init__(self, observation_shape: tuple[int, ...], state_shape: tuple[int, ...], num_actions: int,
                 model: MetazoanQModel,
                 epsilon=0.1, temperature=0.1):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions)
        self.state_shape = state_shape
        self.model = model
        self.epsilon = epsilon          # used in epsilon-greedy action selection
        self.temperature = temperature  # used in logits for random policy

    @abstractmethod
    def perceive(self, observation: Observation) -> State: ...

    @abstractmethod
    def estimate_q(self, state: State) -> Tensor: ...

    def select_action(self, q_values: Tensor) -> Action:
        """
        Epsilon-soft strategy, with random policy selection based on Q(s, a) values
        """
        logits = q_values / self.temperature
        probs = F.softmax(logits, dim=-1)
        probs = (1.0 - self.epsilon) * probs + self.epsilon / probs.size(-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(1)
        return action

    def act(self, observation: Observation = None) -> Action:
        state = self.perceive(observation)
        q_values = self.estimate_q(state)
        action = self.select_action(q_values)
        return action


# === Arthropods (can learn, no brain state) ===
class Arthropod(Metazoan, ABC):
    @abstractmethod
    def encode(self, observation: Observation) -> State: ...

    def perceive(self, observation: Observation) -> State:
        state = self.encode(observation)
        return state

    def estimate_q(self, state: State) -> Tensor:
        with torch.no_grad():
            q_values = self.model(state)  # simple feedforward
        return q_values


# === Vertebrates (can learn, have brain state) ===
class Vertebrate(Metazoan, ABC):
    brain_state: State

    @abstractmethod
    def imprint(self, observation: Observation): ...
    # modifies self.brain_state in-place

    def perceive(self, observation: Observation) -> State:
        self.imprint(observation)
        state = self.brain_state
        return state

    def estimate_q(self, state: State) -> Tensor:
        q_values, new_state = self.model.Q_and_update(state)
        self.brain_state = new_state
        return q_values
