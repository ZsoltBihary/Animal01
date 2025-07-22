import torch
from torch.utils.data import DataLoader
from typing import Optional, Callable


class QTrainer:
    def __init__(
        self,
        q_model: torch.nn.Module,
        buffer: torch.utils.data.Dataset,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initializes the Q-learning trainer.
        Args:
            q_model:        The Q-network (torch.nn.Module).
            buffer:         A dataset yielding (state, action, target_q) triples.
            batch_size:     Mini-batch size.
            learning_rate:  Used only if default optimizer is constructed.
            optimizer:      Optimizer for the Q-network. Defaults to Adam if None.
            loss_fn:        Loss function. Defaults to MSELoss if None.
            device:         Device to train on. Inferred from q_model if None.
        """
        self.q_model = q_model
        self.buffer = buffer
        self.batch_size = batch_size
        self.device = device or next(q_model.parameters()).device
        self.q_model.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(q_model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn or torch.nn.MSELoss()

    def train(self, epochs: int = 1):
        self.q_model.train()
        loader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            count = 0

            for state_batch, action_batch, target_q_batch in loader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                target_q_batch = target_q_batch.to(self.device)

                q_values = self.q_model(state_batch)
                q_selected = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                loss = self.loss_fn(q_selected, target_q_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(count, 1)
            print(f"[Epoch {epoch + 1}/{epochs}] Avg Loss: {avg_loss:.4f}")


# import torch
# from torch.utils.data import DataLoader
# from typing import Optional, Callable
#
#
# class QTrainer:
#     def __init__(
#         self,
#         q_model: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         buffer: torch.utils.data.Dataset,
#         loss_fn: Optional[Callable] = None,
#         device: Optional[torch.device] = None,
#         batch_size: int = 128,
#     ):
#         """
#         Initializes the Q-learning trainer.
#
#         Args:
#             q_model:    The online Q-network (torch.nn.Module).
#             optimizer:  Optimizer for the Q-network.
#             buffer:     A dataset (e.g., CircularQBuffer) yielding (state, action, target_q) triples.
#             loss_fn:    Loss function (e.g., MSELoss or SmoothL1Loss). Defaults to MSE.
#             device:     The device to train on. Inferred from q_model if None.
#             batch_size: Mini-batch size.
#         """
#         self.q_model = q_model
#         self.optimizer = optimizer
#         self.buffer = buffer
#         self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
#         self.device = device or next(q_model.parameters()).device
#         self.batch_size = batch_size
#
#     def train(self, epochs: int = 1):
#         """
#         Trains the Q-model on the experience buffer.
#
#         Args:
#             epochs: Number of full passes over the buffer.
#         """
#         self.q_model.train()
#         loader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True)
#
#         for epoch in range(epochs):
#             total_loss = 0.0
#             count = 0
#
#             for state_batch, action_batch, target_q_batch in loader:
#                 # Move to device
#                 state_batch = state_batch.to(self.device)
#                 action_batch = action_batch.to(self.device)
#                 target_q_batch = target_q_batch.to(self.device)
#
#                 # Forward
#                 q_values = self.q_model(state_batch)  # shape: (B, A)
#                 q_selected = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
#
#                 # Loss + Backprop
#                 loss = self.loss_fn(q_selected, target_q_batch)
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#                 total_loss += loss.item()
#                 count += 1
#
#             avg_loss = total_loss / max(count, 1)
#             print(f"[Epoch {epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")
#
#
# # import torch
# #
# #
# # def train_q_model(Q_online, optimizer, buffer, batch_size=128, epochs=1, loss_fn=None, device=None):
# #     """
# #     Trains the online Q model on experience replay data.
# #
# #     Args:
# #         Q_online:     The online Q-network (torch.nn.Module).
# #         optimizer:    The optimizer for Q_online.
# #         buffer:       A CircularQBuffer containing (state, action, target_q).
# #         batch_size:   Batch size for training.
# #         epochs:       Number of full passes over the buffer.
# #         loss_fn:      Loss function (MSE or Huber). Defaults to MSELoss.
# #         device:       Device to train on. If None, inferred from model.
# #     """
# #     if loss_fn is None:
# #         loss_fn = torch.nn.MSELoss()
# #
# #     if device is None:
# #         device = next(Q_online.parameters()).device
# #
# #     Q_online.train()
# #     loader = torch.utils.data.DataLoader(buffer, batch_size=batch_size, shuffle=True)
# #
# #     for epoch in range(epochs):
# #         total_loss = 0.0
# #         count = 0
# #
# #         for state_batch, action_batch, target_q_batch in loader:
# #             state_batch = state_batch.to(device)
# #             action_batch = action_batch.to(device)
# #             target_q_batch = target_q_batch.to(device)
# #
# #             predicted_q_values = Q_online(state_batch)  # (B, A)
# #             predicted_q = predicted_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
# #
# #             loss = loss_fn(predicted_q, target_q_batch)
# #
# #             optimizer.zero_grad()
# #             loss.backward()
# #             optimizer.step()
# #
# #             total_loss += loss.item()
# #             count += 1
# #
# #         avg_loss = total_loss / max(count, 1)
# #         print(f"[Epoch {epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")
