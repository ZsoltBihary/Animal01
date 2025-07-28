# from abc import ABC, abstractmethod
# from typing import Union, Self
import torch
from torch import Tensor
from tensordict import TensorDict
# from torch import nn
# from torchinfo import summary

Observation = TensorDict
State = TensorDict
Action = Tensor
Schema = dict[str, tuple[torch.Size, torch.dtype]]


# def tensordict_zero(schema: dict[str, torch.Size], batch_size: int, device="cpu"):
#     """Create a TensorDict of zeros based on schema."""
#     return TensorDict(
#         {k: torch.zeros((batch_size, *shape), device=device) for k, shape in schema.items()},
#         batch_size=[batch_size],
#         device=device
#     )

#
# def summary_tensordict(model: nn.Module, schema: dict[str, torch.Size], batch_size: int, device="cpu",
#                        **summary_kwargs):
#     """
#     Wraps a model that takes a TensorDict as input so torchinfo.summary can work.
#
#     Args:
#         model: nn.Module, expects TensorDict as input.
#         schema: dict mapping field names -> torch.Size (without batch dim).
#         batch_size: int, batch size to use for dummy input.
#         device: str or torch.device, device for dummy input.
#         summary_kwargs: extra kwargs passed to torchinfo.summary (e.g., depth, verbose).
#     """
#     class SummaryWrapper(nn.Module):
#         def __init__(self, model, schema, batch_size, device):
#             super().__init__()
#             self.model = model
#             self.schema = schema
#             self.batch_size = batch_size
#             self.device = device
#
#         def forward(self, **kwargs):
#             # Convert dict back to TensorDict
#             td = TensorDict(kwargs, batch_size=[self.batch_size], device=self.device)
#             return self.model(td)
#
#     wrapped_model = SummaryWrapper(model, schema, batch_size, device)
#
#     # Create dummy dict for torchinfo
#     dummy_dict = {k: torch.zeros((batch_size, *shape), device=device) for k, shape in schema.items()}
#
#     return summary(wrapped_model, input_data=dummy_dict, **summary_kwargs)
#

# ---------------------------
# General batching utilities
# ---------------------------

# BatchIndex = Union[slice, Tensor]


# # noinspection PyArgumentList
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
#
#
# class State(TensorBatchMixin, ABC):
#     """Base class for agent's internal state."""
#     pass
#
#
# class Action(TensorBatchMixin, ABC):
#     """Base class for actions taken by agent."""
#     pass
#
#
# class Observation(TensorBatchMixin, ABC):
#     """Base class for environmental feedback to agent."""
#     pass
#
# # ---------------------------------
# # Concrete implementation examples
# # ---------------------------------
#
#
# class InsectStateExample(State):
#     def __init__(self, smell: Tensor, hunger: Tensor):
#         self.smell = smell
#         self.hunger = hunger
#
#     def _tensor_fields(self) -> list[str]:
#         return ["smell", "hunger"]
#
#     def __repr__(self):
#         return f"InsectStateExample(smell={self.smell}, hunger={self.hunger})"
#
#
# class GridAction(Action):
#     def __init__(self, direction: Tensor):
#         self.direction = direction
#
#     def _tensor_fields(self) -> list[str]:
#         return ["direction"]
#
#     def __repr__(self):
#         return f"GridAction(direction={self.direction})"
#
#
# class GridObservationExample(Observation):
#     def __init__(self, image: Tensor, scent: Tensor):
#         self.image = image
#         self.scent = scent
#
#     def _tensor_fields(self) -> list[str]:
#         return ["image", "scent"]
#
#     def __repr__(self):
#         return f"GridObservationExample(image={self.image}, scent={self.scent})"
#
#
# # ---------------------------
# # Sanity check
# # ---------------------------
#
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
