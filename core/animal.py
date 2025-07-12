# animal.py
from abc import ABC, abstractmethod
import torch
from torch import Tensor
# import torch.nn as nn
import torch.nn.functional as F
from utils.helper import A, C
from nn_model.q_model import QInsect


# Base abstract class
class Animal(ABC):
    @abstractmethod
    def policy(self, observation: Tensor) -> Tensor:
        """
        Input: observation(B, ...)
        Output: action(B, )
        """
    pass


class Insect(Animal):
    def __init__(self, main_ch: int = 16):
        self.q_model = QInsect(main_channels=main_ch)

    def perceive(self, observation: Tensor) -> Tensor:
        """
        Input: observation(B, K, K)
        Output: state(B, C, K, K) - standard CNN input
        """
        # One-hot encode (skip ANIMAL = C+1)
        onehot = F.one_hot(observation, num_classes=C+1)[..., :-1]        # (B, K, K, C)
        state = onehot.permute((0, 3, 1, 2)).to(dtype=torch.float32)      # (B, C, K, K)
        return state

    def estimate_q(self, state: Tensor) -> Tensor:
        """
        Input: state(B, C, K, K)
        Output: q_values(B, A)
        """
        with torch.no_grad():
            q_values = self.q_model(state)
        return q_values

    def select_action(self, q_values: Tensor, epsilon: float = 0.1, temperature: float = 0.1) -> Tensor:
        """
        Input: q_values (B, A)
               epsilon used in epsilon-greedy selection
               temperature used in scaling q_values to get logits
        Output: action (B,)
        """
        logits = q_values / temperature  # (B, A)
        # Softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)  # (B, A)
        # Blend with uniform distribution for epsilon-greedy behavior
        probabilities = (1.0 - epsilon) * probabilities + epsilon / A  # (B, A)
        # Sample actions from the categorical distribution
        action = torch.multinomial(probabilities, num_samples=1).squeeze(1)  # (B,)
        return action

    def policy(self, observation) -> Tensor:
        """
        Input: observation(B, K, K)
        Output: action(B, )
        """
        state = self.perceive(observation)
        q_values = self.estimate_q(state)
        action = self.select_action(q_values)
        return action


class Amoeba(Animal):

    def policy(self, obs: Tensor) -> Tensor:
        B = obs.shape[0]
        action = torch.randint(low=0, high=A, size=(B,))
        return action
