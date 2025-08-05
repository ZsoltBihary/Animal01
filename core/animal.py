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
import torch.nn.functional as F
from abc import ABC, abstractmethod
from core.q_model import MetazoanQModel  # , VertebrateQModel
from core.tensordict_helper import Schema, Observation, State, Action


# === Base Animal Class ===
class Animal(ABC):
    def __init__(self, observation_schema: Schema, num_actions: int):
        self.observation_schema = observation_schema
        self.num_actions = num_actions

    @abstractmethod
    def act(self, observation: Observation = None) -> Action: ...


# === Metazoans (can learn) ===
class Metazoan(Animal, ABC):
    def __init__(self,
                 observation_schema: Schema,
                 state_schema: Schema,
                 num_actions: int,
                 model: MetazoanQModel,
                 epsilon: float, temperature: float
                 ):
        super().__init__(observation_schema=observation_schema, num_actions=num_actions)
        self.state_schema = state_schema
        self.model = model.cuda()
        self.epsilon = epsilon
        self.temperature = temperature

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
        state = self.perceive(observation).cuda()
        q_values = self.estimate_q(state).cpu()
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
            q_values = self.model(state.cuda()).cpu()  # simple feedforward
        return q_values


# TODO: Deal with Vertebrate class later
# # === Vertebrates (can learn, have brain state) ===
# class Vertebrate(Metazoan, ABC):
#     brain_state: State
#
#     @abstractmethod
#     def imprint(self, observation: Observation): ...
#     # modifies self.brain_state in-place
#
#     def perceive(self, observation: Observation) -> State:
#         self.imprint(observation)
#         state = self.brain_state
#         return state
#
#     def estimate_q(self, state: State) -> Tensor:
#         q_values, new_state = self.model.Q_and_update(state)
#         self.brain_state = new_state
#         return q_values

# # Some other classes ...
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
