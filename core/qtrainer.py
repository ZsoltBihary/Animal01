import torch
from torch.utils.data import DataLoader
import copy
from typing import Optional, Callable
from line_profiler_pycharm import profile


class QTrainer:
    def __init__(
        self,
        q_model: torch.nn.Module,
        buffer: torch.utils.data.Dataset,
        device: torch.device,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None
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
            device:         Device to train on.
        """
        self.model = copy.deepcopy(q_model).to(device)
        self.buffer = buffer
        self.device = device
        self.batch_size = batch_size
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn or torch.nn.MSELoss()

    @profile
    def train(self, epochs: int = 1, verbose=0) -> float:
        self.model.train()  # This is the trainer Q model, needs to be always in train() mode
        loader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            count = 0

            for state_batch, action_batch, target_q_batch in loader:
                state_batch = state_batch.to(self.device)
                action_batch = action_batch.to(self.device)
                target_q_batch = target_q_batch.to(self.device)

                q_values = self.model(state_batch)
                q_selected = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

                loss = self.loss_fn(q_selected, target_q_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                count += 1

            avg_loss = total_loss / max(count, 1)
            if verbose >= 1:
                print(f"[Epoch {epoch + 1}/{epochs}] Avg Loss: {avg_loss:.4f}")
            return avg_loss
