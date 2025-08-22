import torch
from torch.utils.data import Dataset
from reptile_world.config import Config


class RingBufferDataset(Dataset):
    """
    Circular buffer dataset with fixed CPU storage.
    - Stores tensors in a preallocated array of fixed capacity.
    - Overwrites oldest entries when full (ring buffer mechanics).
    - Only supports batched writes.
    """
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def handle_expansion(self, batch_size: int):
        """Hook for implementing capacity expansion (not used here)."""
        raise ValueError(
            f"Batch size {batch_size} exceeds buffer capacity {self.capacity}. "
            f"Expansion not implemented."
        )

    def _advance(self, batch_size: int):
        """Move the write index forward, wrapping around if needed."""
        if batch_size > self.capacity:
            self.handle_expansion(batch_size)
        end = self.idx + batch_size
        if end >= self.capacity:
            self.full = True
        self.idx = end % self.capacity

    def _write_batch(self, arr: torch.Tensor, batch: torch.Tensor, start_idx: int):
        """
        Write a batch into the buffer array with wrap-around handling.

        Always stores:
        - On CPU
        - Detached from autograd history
        """
        batch = batch.detach().cpu()
        n = batch.shape[0]
        end_idx = start_idx + n
        if end_idx <= self.capacity:
            arr[start_idx:end_idx] = batch
        else:
            split = self.capacity - start_idx
            arr[start_idx:] = batch[:split]
            arr[:n - split] = batch[split:]


class SDQLBuffer(RingBufferDataset):
    """
    Full-state SDQL replay buffer (CPU-only).
    Stored fields:
    - encoded:   (N, E, K, K) float32
    - state:     (N, L, S, K, K) float32
    - action:    (N,) int64
    - q_target:  (N,) float32
    - r_target:  (N,) float32
    - obs_target: (N, K, K) int64 (discrete pixel labels)
    Notes:
    - Capacity is computed from Config: steps_per_episode * batch_size * buffer_reuse
    - DataLoader or custom batching can be used for sampling.
    - Returned samples remain on CPU — move to GPU during training if needed.
    """
    def __init__(self, conf: Config):
        capacity = int(conf.steps_per_episode * conf.batch_size * conf.buffer_reuse)
        super().__init__(capacity)

        E = conf.encoded_channels
        K = conf.obs_size
        L = conf.brain_state_layers
        S = conf.brain_state_channels

        # Preallocate storage on CPU
        self.encoded = torch.empty((capacity, E, K, K), dtype=torch.float32, device="cpu")
        self.state = torch.empty((capacity, L, S, K, K), dtype=torch.float32, device="cpu")
        self.action = torch.empty((capacity,), dtype=torch.long, device="cpu")
        self.q_target = torch.empty((capacity,), dtype=torch.float32, device="cpu")
        self.r_target = torch.empty((capacity,), dtype=torch.float32, device="cpu")
        self.obs_target = torch.empty((capacity, K, K), dtype=torch.long, device="cpu")

        print(f"SDQLBuffer initialized with capacity {self.capacity}")

    def add_batch(self, encoded_batch, state_batch, action_batch, q_target_batch, r_target_batch, obs_target_batch):
        """Add a full batch of experiences to the buffer."""
        bsz = encoded_batch.shape[0]
        self._write_batch(self.encoded, encoded_batch, self.idx)
        self._write_batch(self.state, state_batch, self.idx)
        self._write_batch(self.action, action_batch, self.idx)
        self._write_batch(self.q_target, q_target_batch, self.idx)
        self._write_batch(self.r_target, r_target_batch, self.idx)
        self._write_batch(self.obs_target, obs_target_batch, self.idx)
        self._advance(bsz)

    def __getitem__(self, idx):
        """Return a single stored experience."""
        return (
            self.encoded[idx],
            self.state[idx],
            self.action[idx],
            self.q_target[idx],
            self.r_target[idx],
            self.obs_target[idx],
        )


# # ---- A1: Stateless (Residual) + Q-only ----
# class DQLBuffer(RingBufferDataset):
#     def __init__(self, capacity, encoded_shape):
#         super().__init__(capacity)
#         self.encoded = torch.empty((capacity,) + tuple(encoded_shape), dtype=torch.float32)
#         self.action = torch.empty((capacity,), dtype=torch.long)
#         self.q_target = torch.empty((capacity,), dtype=torch.float32)
#
#     def add_batch(self, encoded_batch, action_batch, q_target_batch):
#         bsz = encoded_batch.shape[0]
#         self._write_batch(self.encoded, encoded_batch, self.idx)
#         self._write_batch(self.action, action_batch, self.idx)
#         self._write_batch(self.q_target, q_target_batch, self.idx)
#         self._advance(bsz)
#
#     def __getitem__(self, idx):
#         return self.encoded[idx], self.action[idx], self.q_target[idx]
#
#
# # ---- A2: Stateless (Residual) + Q+Aux ----
# class DQLAuxBuffer(RingBufferDataset):
#     def __init__(self, capacity, encoded_shape, obs_shape):
#         super().__init__(capacity)
#         self.encoded = torch.empty((capacity,) + tuple(encoded_shape), dtype=torch.float32)
#         self.action = torch.empty((capacity,), dtype=torch.long)
#         self.q_target = torch.empty((capacity,), dtype=torch.float32)
#         self.r_target = torch.empty((capacity,), dtype=torch.float32)
#         self.obs_target = torch.empty((capacity,) + tuple(obs_shape), dtype=torch.float32)
#
#     def add_batch(self, encoded_batch, action_batch, q_target_batch, r_target_batch, obs_target_batch):
#         bsz = encoded_batch.shape[0]
#         self._write_batch(self.encoded, encoded_batch, self.idx)
#         self._write_batch(self.action, action_batch, self.idx)
#         self._write_batch(self.q_target, q_target_batch, self.idx)
#         self._write_batch(self.r_target, r_target_batch, self.idx)
#         self._write_batch(self.obs_target, obs_target_batch, self.idx)
#         self._advance(bsz)
#
#     def __getitem__(self, idx):
#         return (
#             self.encoded[idx],
#             self.action[idx],
#             self.q_target[idx],
#             self.r_target[idx],
#             self.obs_target[idx],
#         )
#
#
# # ---- B1: Stateful (Brainy) + Q-only ----
# class SDQLQOnlyBuffer(RingBufferDataset):
#     def __init__(self, capacity, encoded_shape, state_shape):
#         super().__init__(capacity)
#         self.encoded = torch.empty((capacity,) + tuple(encoded_shape), dtype=torch.float32)
#         self.state = torch.empty((capacity,) + tuple(state_shape), dtype=torch.float32)
#         self.action = torch.empty((capacity,), dtype=torch.long)
#         self.q_target = torch.empty((capacity,), dtype=torch.float32)
#
#     def add_batch(self, encoded_batch, state_batch, action_batch, q_target_batch):
#         bsz = encoded_batch.shape[0]
#         self._write_batch(self.encoded, encoded_batch, self.idx)
#         self._write_batch(self.state, state_batch, self.idx)
#         self._write_batch(self.action, action_batch, self.idx)
#         self._write_batch(self.q_target, q_target_batch, self.idx)
#         self._advance(bsz)
#
#     def __getitem__(self, idx):
#         return self.encoded[idx], self.state[idx], self.action[idx], self.q_target[idx]
