import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Trainer: class-based, multi-epoch, auto-dispatch by dataset type
# ============================================================
class SupervisedRLTrainer:
    """
    Class-based trainer that:
      - runs multi-epoch .fit()
      - auto-selects training step by inspecting dataset type
      - supports Q-only and Q+Aux, stateless or stateful
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str = "cpu",
        criterion_q: nn.Module | None = None,
        criterion_r: nn.Module | None = None,
        criterion_obs: nn.Module | None = None,
        lambda_r: float = 1.0,
        lambda_obs: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        # default to MSE if not provided
        self.criterion_q = criterion_q if criterion_q is not None else nn.MSELoss()
        self.criterion_r = criterion_r if criterion_r is not None else nn.MSELoss()
        self.criterion_obs = criterion_obs if criterion_obs is not None else nn.MSELoss()
        self.lambda_r = float(lambda_r)
        self.lambda_obs = float(lambda_obs)

    # ---- public API ----
    def fit(self, dataloader: DataLoader, epochs: int = 1, log_interval: int = 1):
        """
        Runs multiple epochs over the dataloader.
        Auto-detects which training step to use based on dataloader.dataset type.
        """
        train_step = self._select_train_step(dataloader.dataset)

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                loss_val = train_step(batch)
                running_loss += loss_val
                num_batches += 1

            avg_loss = running_loss / max(1, num_batches)
            if (epoch % log_interval) == 0:
                print(f"[Epoch {epoch}/{epochs}] avg_loss={avg_loss:.6f}")

    # ---- dispatch based on dataset type ----
    def _select_train_step(self, dataset: Dataset):
        if isinstance(dataset, DQLBuffer):
            return self._train_step_dql
        elif isinstance(dataset, DQLAuxBuffer):
            return self._train_step_dql_aux
        elif isinstance(dataset, SDQLQOnlyBuffer):
            return self._train_step_sdql
        elif isinstance(dataset, SDQLBuffer):
            return self._train_step_sdql_aux
        else:
            raise TypeError(
                f"Unsupported dataset type: {type(dataset).__name__}. "
                f"Expected one of: DQLBuffer, DQLAuxBuffer, SDQLQOnlyBuffer, SDQLBuffer."
            )

    # ---- helpers ----
    @staticmethod
    def _batch_to_device(batch, device):
        # Convert a tuple/list of tensors to device; keep non-tensors as-is
        if isinstance(batch, (list, tuple)):
            return tuple(b.to(device, non_blocking=True) if torch.is_tensor(b) else b for b in batch)
        return batch

    # ---- per-variant training steps ----
    def _train_step_dql(self, batch):
        # (encoded, action, q_target)
        batch = self._batch_to_device(batch, self.device)
        encoded, action, q_target = batch

        self.optimizer.zero_grad(set_to_none=True)

        q_a = self.model(encoded)                  # (B, A)
        batch_idx = torch.arange(action.shape[0], device=action.device)
        q_pred = q_a[batch_idx, action]            # (B,)
        loss = self.criterion_q(q_pred, q_target)  # scalar

        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def _train_step_dql_aux(self, batch):
        # (encoded, action, q_target, r_target, obs_target)
        batch = self._batch_to_device(batch, self.device)
        encoded, action, q_target, r_target, obs_target = batch

        self.optimizer.zero_grad(set_to_none=True)

        q_a, r_a, obs_a = self.model(encoded)      # each (B, A) except obs shape if different
        batch_idx = torch.arange(action.shape[0], device=action.device)

        q_pred = q_a[batch_idx, action]
        r_pred = r_a[batch_idx, action]
        obs_pred = obs_a[batch_idx, action]

        loss_q = self.criterion_q(q_pred, q_target)
        loss_r = self.criterion_r(r_pred, r_target)
        loss_o = self.criterion_obs(obs_pred, obs_target)

        loss = loss_q + self.lambda_r * loss_r + self.lambda_obs * loss_o
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def _train_step_sdql(self, batch):
        # (encoded, state, action, q_target)
        batch = self._batch_to_device(batch, self.device)
        encoded, state, action, q_target = batch

        self.optimizer.zero_grad(set_to_none=True)

        q_a = self.model(encoded, state)           # (B, A)
        batch_idx = torch.arange(action.shape[0], device=action.device)
        q_pred = q_a[batch_idx, action]
        loss = self.criterion_q(q_pred, q_target)

        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def _train_step_sdql_aux(self, batch):
        # (encoded, state, action, q_target, r_target, obs_target)
        batch = self._batch_to_device(batch, self.device)
        encoded, state, action, q_target, r_target, obs_target = batch

        self.optimizer.zero_grad(set_to_none=True)

        q_a, r_a, obs_a = self.model(encoded, state)
        batch_idx = torch.arange(action.shape[0], device=action.device)

        q_pred = q_a[batch_idx, action]
        r_pred = r_a[batch_idx, action]
        obs_pred = obs_a[batch_idx, action]

        loss_q = self.criterion_q(q_pred, q_target)
        loss_r = self.criterion_r(r_pred, r_target)
        loss_o = self.criterion_obs(obs_pred, obs_target)

        loss = loss_q + self.lambda_r * loss_r + self.lambda_obs * loss_o
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())


# ============================================================
# Minimal dummy models + sanity checks
# ============================================================

class DummyStatelessQ(nn.Module):
    """Stateless model producing Q only."""
    def __init__(self, in_dim=10, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class DummyStatelessQRA(nn.Module):
    """Stateless model producing Q, R, Obs (same action-aligned shapes)."""
    def __init__(self, in_dim=10, num_actions=4, obs_dim=6):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, num_actions))
        self.r = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, num_actions))
        # For simplicity, obs head returns (B, A, obs_dim); we’ll pick [batch, action, :]
        self.o = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, num_actions * obs_dim))
        self.num_actions = num_actions
        self.obs_dim = obs_dim

    def forward(self, x):
        q = self.q(x)
        r = self.r(x)
        o = self.o(x).view(x.shape[0], self.num_actions, self.obs_dim)
        return q, r, o


class DummyStatefulQ(nn.Module):
    """Stateful model producing Q from (encoded, state)."""
    def __init__(self, enc_dim=10, state_dim=8, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim + state_dim, 64), nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, encoded, state):
        x = torch.cat([encoded, state], dim=-1)
        return self.net(x)


class DummyStatefulQRA(nn.Module):
    """Stateful model producing Q, R, Obs from (encoded, state)."""
    def __init__(self, enc_dim=10, state_dim=8, num_actions=4, obs_dim=6):
        super().__init__()
        in_dim = enc_dim + state_dim
        self.q = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, num_actions))
        self.r = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, num_actions))
        self.o = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, num_actions * obs_dim))
        self.num_actions = num_actions
        self.obs_dim = obs_dim

    def forward(self, encoded, state):
        x = torch.cat([encoded, state], dim=-1)
        q = self.q(x)
        r = self.r(x)
        o = self.o(x).view(x.shape[0], self.num_actions, self.obs_dim)
        return q, r, o


# if __name__ == "__main__":
#     torch.manual_seed(0)
#
#     # ---------------- A1: DQLBuffer (stateless, Q-only) ----------------
#     cap = 128
#     B = 64
#     in_dim = 10
#     num_actions = 5
#
#     dql_buf = DQLBuffer(capacity=cap, encoded_shape=(in_dim,))
#     # dump two batches
#     dql_buf.add_batch(torch.randn(B, in_dim), torch.randint(0, num_actions, (B,)), torch.randn(B))
#     dql_buf.add_batch(torch.randn(B, in_dim), torch.randint(0, num_actions, (B,)), torch.randn(B))
#
#     dql_loader = DataLoader(dql_buf, batch_size=32, shuffle=True, pin_memory=True)
#     model_a1 = DummyStatelessQ(in_dim=in_dim, num_actions=num_actions)
#     opt = torch.optim.Adam(model_a1.parameters(), lr=1e-3)
#     trainer = SupervisedRLTrainer(model_a1, opt, device="cpu")
#     print("\n--- Training A1: DQL (stateless, Q-only) ---")
#     trainer.fit(dql_loader, epochs=2, log_interval=1)
#
#     # ---------------- A2: DQLAuxBuffer (stateless, Q+Aux) ----------------
#     obs_dim = 6
#     dql_aux_buf = DQLAuxBuffer(capacity=cap, encoded_shape=(in_dim,), obs_shape=(obs_dim,))
#     dql_aux_buf.add_batch(
#         torch.randn(B, in_dim),
#         torch.randint(0, num_actions, (B,)),
#         torch.randn(B),
#         torch.randn(B),
#         torch.randn(B, obs_dim),
#     )
#     dql_aux_buf.add_batch(
#         torch.randn(B, in_dim),
#         torch.randint(0, num_actions, (B,)),
#         torch.randn(B),
#         torch.randn(B),
#         torch.randn(B, obs_dim),
#     )
#     dql_aux_loader = DataLoader(dql_aux_buf, batch_size=32, shuffle=True, pin_memory=True)
#     model_a2 = DummyStatelessQRA(in_dim=in_dim, num_actions=num_actions, obs_dim=obs_dim)
#     opt2 = torch.optim.Adam(model_a2.parameters(), lr=1e-3)
#     trainer2 = SupervisedRLTrainer(model_a2, opt2, device="cpu", lambda_r=0.5, lambda_obs=0.1)
#     print("\n--- Training A2: DQL+Aux (stateless, Q+Aux) ---")
#     trainer2.fit(dql_aux_loader, epochs=2, log_interval=1)
#
#     # ---------------- B1: SDQLQOnlyBuffer (stateful, Q-only) ----------------
#     state_dim = 8
#     sdql_q_buf = SDQLQOnlyBuffer(capacity=cap, encoded_shape=(in_dim,), state_shape=(state_dim,))
#     sdql_q_buf.add_batch(
#         torch.randn(B, in_dim),
#         torch.randn(B, state_dim),
#         torch.randint(0, num_actions, (B,)),
#         torch.randn(B),
#     )
#     sdql_q_buf.add_batch(
#         torch.randn(B, in_dim),
#         torch.randn(B, state_dim),
#         torch.randint(0, num_actions, (B,)),
#         torch.randn(B),
#     )
#     sdql_q_loader = DataLoader(sdql_q_buf, batch_size=32, shuffle=True, pin_memory=True)
#     model_b1 = DummyStatefulQ(enc_dim=in_dim, state_dim=state_dim, num_actions=num_actions)
#     opt3 = torch.optim.Adam(model_b1.parameters(), lr=1e-3)
#     trainer3 = SupervisedRLTrainer(model_b1, opt3, device="cpu")
#     print("\n--- Training B1: SDQL (stateful, Q-only) ---")
#     trainer3.fit(sdql_q_loader, epochs=2, log_interval=1)
#
#     # ---------------- B2: SDQLBuffer (stateful, Q+Aux) ----------------
#     sdql_buf = SDQLBuffer(capacity=cap, encoded_shape=(in_dim,), state_shape=(state_dim,), obs_shape=(obs_dim,))
#     sdql_buf.add_batch(
#         torch.randn(B, in_dim),
#         torch.randn(B, state_dim),
#         torch.randint(0, num_actions, (B,)),
#         torch.randn(B),
#         torch.randn(B),
#         torch.randn(B, obs_dim),
#     )
#     sdql_buf.add_batch(
#         torch.randn(B, in_dim),
#         torch.randn(B, state_dim),
#         torch.randint(0, num_actions, (B,)),
#         torch.randn(B),
#         torch.randn(B),
#         torch.randn(B, obs_dim),
#     )
#     sdql_loader = DataLoader(sdql_buf, batch_size=32, shuffle=True, pin_memory=True)
#     model_b2 = DummyStatefulQRA(enc_dim=in_dim, state_dim=state_dim, num_actions=num_actions, obs_dim=obs_dim)
#     opt4 = torch.optim.Adam(model_b2.parameters(), lr=1e-3)
#     trainer4 = SupervisedRLTrainer(model_b2, opt4, device="cpu", lambda_r=0.5, lambda_obs=0.1)
#     print("\n--- Training B2: SDQL+Aux (stateful, Q+Aux) ---")
#     trainer4.fit(sdql_loader, epochs=2, log_interval=1)
#
#     print("\nAll sanity checks complete.")
