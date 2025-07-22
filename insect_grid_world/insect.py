from torch import Tensor
import torch.nn.functional as F
from core.animal import Arthropod
from insect_grid_world.insect_model import InsectQModel


class Insect(Arthropod):
    def __init__(self, model: InsectQModel, epsilon=0.1, temperature=0.1):
        super().__init__(model, epsilon, temperature)
        self.C, _, _ = self.model.state_shape

    def encode(self, observation: Tensor) -> Tensor:
        """
        observation: (B, K, K) with ints in [0, ?], but we want to encode only [0, C-1]
        """
        # Mask: keep only valid values in range [0, C - 1]
        valid = (observation >= 0) & (observation < self.C)  # shape: (B, K, K)
        mask = valid.unsqueeze(-1)  # shape: (B, K, K, 1)

        # Clamp to valid range to avoid index errors in one_hot
        clamped = observation.clamp(0, self.C - 1)  # shape: (B, K, K)

        # One-hot encode and mask invalid entries
        one_hot = F.one_hot(clamped, num_classes=self.C)  # shape: (B, K, K, C)
        one_hot = one_hot * mask  # zero out invalid entries

        # Reorder to (B, C, K, K) for CNN input
        state = one_hot.permute(0, 3, 1, 2).float()  # shape: (B, C, K, K)
        return state
