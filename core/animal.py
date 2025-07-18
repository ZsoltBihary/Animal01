# animal.py
# import torch.nn as nn

# Animal
# ├────────── * Sponge
# ├────────── * Amoeba
# └── Metazoan           ← includes all Q-learning-capable agents
#     ├── Arthropod      ← includes all stateless Q-learning-capable agents
#     │   └── * Insect
#     └── Vertebrate     ← includes memory/state-based Q-learning agents
#         ├── * Reptile
#         └── * Mammal

import torch
from torch import Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod
from core.q_model import MetazoanQModel, VertebrateQModel


# === Base Animal Class ===
class Animal(ABC):
    @abstractmethod
    def act(self, observation=None) -> Tensor: ...


# === Metazoans (can learn) ===
class Metazoan(Animal):
    def __init__(self, model: MetazoanQModel, epsilon: float = 0.1, temperature: float = 0.1):
        self.model = model
        self.epsilon = epsilon          # used in epsilon-greedy action selection
        self.temperature = temperature  # used in logits for random policy

    @abstractmethod
    def perceive(self, observation: Tensor) -> Tensor: ...

    @abstractmethod
    def estimate_q(self, state: Tensor) -> Tensor: ...

    def select_action(self, q_values: Tensor) -> Tensor:
        """
        Epsilon-soft strategy, with random policy selection based on Q(s, a) values
        """
        logits = q_values / self.temperature
        probs = F.softmax(logits, dim=-1)
        probs = (1.0 - self.epsilon) * probs + self.epsilon / probs.size(-1)
        action = torch.multinomial(probs, num_samples=1).squeeze(1)
        return action

    def act(self, observation=None) -> Tensor:
        state = self.perceive(observation)
        q_values = self.estimate_q(state)
        action = self.select_action(q_values)
        return action


# === Arthropods (can learn, no brain state) ===
class Arthropod(Metazoan):
    @abstractmethod
    def encode(self, observation: Tensor) -> Tensor: ...

    def perceive(self, observation: Tensor) -> Tensor:
        state = self.encode(observation)
        return state

    def estimate_q(self, state: Tensor) -> Tensor:
        q_values = self.model(state)  # simple feedforward
        return q_values


# === Vertebrates (can learn, have brain state) ===
class Vertebrate(Metazoan):
    def __init__(self, model: VertebrateQModel, epsilon=0.1, temperature=0.1):
        super().__init__(model, epsilon, temperature)
        self.brain_state = None

    @abstractmethod
    def imprint(self, observation: Tensor): ...
    # modifies self.brain_state in-place

    def perceive(self, observation: Tensor) -> Tensor:
        self.imprint(observation)
        state = self.brain_state
        return state

    def estimate_q(self, state: Tensor) -> Tensor:
        q_values, new_state = self.model.Q_and_update(state)
        self.brain_state = new_state
        return q_values


# # === Concrete animal classes to be implemented ===
# #   === Primitive Animals (no learning) ===
# class Sponge(Animal):
#     def act(self, observation=None):
#         # TODO: return "STAY"
#         pass
#
#
# class Amoeba(Animal):
#     def act(self, observation=None):
#         # TODO: return random.choice(["STAY, "UP", "DOWN", "LEFT", "RIGHT"])
#         pass
#
#
# class Insect(Arthropod):
#
#     def __init__(self, model: MetazoanQModel, epsilon=0.1, temperature=0.1):
#         super().__init__(model, epsilon, temperature)
#
#     def encode(self, observation: Tensor) -> Tensor:
#         state = observation
#         return state
#
# class Reptile(Vertebrate):
#     pass
#
#
# class Mammal(Vertebrate):
#     pass
#

# # class Insect(Animal):
# #     def __init__(self, main_ch: int = 16):
# #         self.q_model = QInsect01()
# #
# #     def perceive(self, observation: Tensor) -> Tensor:
# #         """
# #         Input: observation(B, K, K)
# #         Output: state(B, C, K, K) - standard CNN input
# #         """
# #         # One-hot encode (skip ANIMAL = C+1)
# #         onehot = F.one_hot(observation, num_classes=C+1)[..., :-1]        # (B, K, K, C)
# #         state = onehot.permute((0, 3, 1, 2)).to(dtype=torch.float32)      # (B, C, K, K)
# #         return state
# #
# #     def estimate_q(self, state: Tensor) -> Tensor:
# #         """
# #         Input: state(B, C, K, K)
# #         Output: q_values(B, A)
# #         """
# #         with torch.no_grad():
# #             q_values = self.q_model(state)
# #         return q_values
# #
# #     def select_action(self, q_values: Tensor, epsilon: float = 0.0, temperature: float = 0.1) -> Tensor:
# #         """
# #         Input: q_values (B, A)
# #                epsilon used in epsilon-greedy selection
# #                temperature used in scaling q_values to get logits
# #         Output: action (B,)
# #         """
# #         logits = q_values / temperature  # (B, A)
# #         # Softmax to get probabilities
# #         probabilities = torch.softmax(logits, dim=-1)  # (B, A)
# #         # Blend with uniform distribution for epsilon-greedy behavior
# #         probabilities = (1.0 - epsilon) * probabilities + epsilon / A  # (B, A)
# #         # Sample actions from the categorical distribution
# #         action = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # (B,)
# #         return action
# #
# #     def policy(self, observation) -> Tensor:
# #         """
# #         Input: observation(B, K, K)
# #         Output: action(B, )
# #         """
# #         state = self.perceive(observation)
# #         q_values = self.estimate_q(state)
# #         action = self.select_action(q_values)  # (B, )
# #         return action
# #
# #
# # class Amoeba(Animal):
# #
# #     def policy(self, obs: Tensor) -> Tensor:
# #         B = obs.shape[0]
# #         action = torch.randint(low=0, high=A, size=(B,))
# #         return action
