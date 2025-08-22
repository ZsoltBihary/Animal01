import torch
from torch import nn, optim
from torch.utils.data import DataLoader
# from typing import Callable, Optional
from reptile_world.config import Config
from reptile_world.sdql_replay_buffer import SDQLReplayBuffer
from reptile_world.sdqn_model import SDQNModel


class SDQLTrainer:
    """
    Supervised trainer for SDQL models using full-buffer sweeps.

    Args:
        model (nn.Module): The SDQLModel or compatible network.
        buffer (Dataset): An SDQLBuffer instance (CPU-based).
    Settings:
        self.q_loss_fn = nn.MSELoss().
        self.r_loss_fn = nn.MSELoss().
        self.obs_loss_fn = nn.CrossEntropyLoss().
    """
    def __init__(self, conf: Config, model: SDQNModel, buffer: SDQLReplayBuffer):
        self.conf = conf
        self.model = model
        self.buffer = buffer
        # === Consume configuration parameters ===
        self.device = conf.trainer_device
        self.num_epochs = conf.num_epochs
        self.batch_size = conf.training_batch_size

        self.loss_r_weight = conf.loss_r_weight
        self.loss_obs_weight = conf.loss_obs_weight

        # Set up standard loss functions
        self.q_loss_fn = nn.MSELoss()
        self.r_loss_fn = nn.MSELoss()
        self.obs_loss_fn = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
            model.parameters(), lr=conf.learning_rate0, weight_decay=conf.weight_decay)

    def fit(self):
        """
        Train the model for a given number of epochs over the buffer.
        """
        self.model.train()

        loader = DataLoader(
            self.buffer,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )

        max_grad_norm = 1.0  # tune this value as needed

        for epoch in range(self.num_epochs):

            total_loss = torch.tensor(0.0, device=self.device)
            total_q_loss = torch.tensor(0.0, device=self.device)
            total_r_loss = torch.tensor(0.0, device=self.device)
            total_obs_loss = torch.tensor(0.0, device=self.device)
            batches = 0

            for encoded, state, action, q_target, r_target, obs_target in loader:
                encoded = encoded.to(self.device, non_blocking=True)
                state = state.to(self.device, non_blocking=True)
                action = action.to(self.device, non_blocking=True)
                q_target = q_target.to(self.device, non_blocking=True)
                r_target = r_target.to(self.device, non_blocking=True)
                obs_target = obs_target.to(self.device, non_blocking=True)

                # Forward pass
                q_a, r_a, obs_logit_a = self.model(encoded, state)
                q_pred = q_a.gather(1, action.unsqueeze(1)).squeeze(1)
                r_pred = r_a.gather(1, action.unsqueeze(1)).squeeze(1)
                # obs_logit_pred = obs_logit_a.gather(1, action.unsqueeze(1)).squeeze(1)
                # Crazy cryptic one-liner
                obs_logit_pred = obs_logit_a.gather(
                    1,
                    action.view(-1, 1, 1, 1, 1).expand(-1, 1, *obs_logit_a.shape[2:])
                ).squeeze(1)

                # Compute losses
                q_loss = self.q_loss_fn(q_pred, q_target)
                r_loss = self.r_loss_fn(r_pred, r_target)
                obs_loss = self.obs_loss_fn(obs_logit_pred, obs_target)
                loss = q_loss + self.loss_r_weight * r_loss + self.loss_obs_weight * obs_loss

                # Backprop and optimizer step with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                # Accumulate totals on GPU
                total_loss += loss
                total_q_loss += q_loss
                total_r_loss += r_loss
                total_obs_loss += obs_loss
                batches += 1

            print(
                f"[Epoch {epoch + 1}/{self.num_epochs}] "
                f"Total: {total_loss / batches:.4f} | "
                f"Q: {total_q_loss / batches:.4f} | "
                f"R: {total_r_loss / batches:.4f} | "
                f"Obs: {total_obs_loss / batches:.4f}"
            )

    def set_learning_rate(self, new_lr: float):
        """
        Set a new learning rate for the optimizer.
        Args:
            new_lr (float): The new learning rate value.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Learning rate set to {new_lr}")
