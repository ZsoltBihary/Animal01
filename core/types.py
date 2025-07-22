# from abc import ABC, abstractmethod
# from typing import Union, Self
# import torch
from torch import Tensor

# For now, all types are simple Tensor ...
Observation = Tensor
State = Tensor
Action = Tensor

# ---------------------------
# General batching utilities
# ---------------------------

# BatchIndex = Union[slice, torch.Tensor]


# noinspection PyArgumentList
# class TensorBatchMixin(ABC):
#     @abstractmethod
#     def _tensor_fields(self) -> list[str]:
#         """List of tensor field names for slicing/indexing."""
#         ...
#
#     def __getitem__(self, index: BatchIndex) -> Self:
#         kwargs = {k: getattr(self, k)[index] for k in self._tensor_fields()}
#         return self.__class__(**kwargs)
#
#     def __setitem__(self, index: BatchIndex, value: Self) -> None:
#         for k in self._tensor_fields():
#             getattr(self, k)[index] = getattr(value, k)


# class State(TensorBatchMixin, ABC):
#     """Base class for agent's internal state."""
#     pass


# class Action(TensorBatchMixin, ABC):
#     """Base class for actions taken by agent."""
#     pass


# class Observation(TensorBatchMixin, ABC):
#     """Base class for environmental feedback to agent."""
#     pass

# ---------------------------------
# Concrete implementation examples
# ---------------------------------


# class InsectStateExample(State):
#     def __init__(self, smell: torch.Tensor, hunger: torch.Tensor):
#         self.smell = smell
#         self.hunger = hunger
#
#     def _tensor_fields(self) -> list[str]:
#         return ["smell", "hunger"]
#
#     def __repr__(self):
#         return f"InsectStateExample(smell={self.smell}, hunger={self.hunger})"


# class GridAction(Action):
#     def __init__(self, direction: torch.Tensor):
#         self.direction = direction
#
#     def _tensor_fields(self) -> list[str]:
#         return ["direction"]
#
#     def __repr__(self):
#         return f"GridAction(direction={self.direction})"


# class GridObservationExample(Observation):
#     def __init__(self, image: torch.Tensor, scent: torch.Tensor):
#         self.image = image
#         self.scent = scent
#
#     def _tensor_fields(self) -> list[str]:
#         return ["image", "scent"]
#
#     def __repr__(self):
#         return f"GridObservationExample(image={self.image}, scent={self.scent})"


# ---------------------------
# Sanity check
# ---------------------------

# def demo():
#     B = 4  # batch size
#
#     obs = GridObservationExample(
#         image=torch.rand(B, 3, 5, 5),
#         scent=torch.rand(B, 8)
#     )
#
#     act = GridAction(
#         direction=torch.tensor([0, 1, 2, 3])
#     )
#
#     state = InsectStateExample(
#         smell=torch.rand(B, 10),
#         hunger=torch.rand(B)
#     )
#
#     # Slice some entries
#     print("=== Full Batch ===")
#     print("Observation scent:\n", obs.scent)
#     print("Action direction:\n", act)
#     print("State hunger:\n", state.hunger)
#
#     obs2 = obs[1:3]
#     act2 = act[2:3]
#     state2 = state[:2]
#
#     print("\n=== Sliced Batches ===")
#     print("Sliced scent shape:", obs2.scent.shape)
#     print("Sliced action values:", act2)
#     print("Sliced hunger tensor:", state2.hunger)
#
#     # Modify in place
#     state2.hunger += 1.0
#     state[:2] = state2
#     print("\n=== Modified Hunger ===")
#     print("Updated hunger in original state:\n", state.hunger)
#
#
# if __name__ == "__main__":
#     demo()
