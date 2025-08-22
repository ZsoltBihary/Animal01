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

    # def __len__(self):
    #     return self.capacity if self.full else self.idx

    def handle_expansion(self, batch_size: int):
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
        """Write a batch into the buffer array with wrap-around handling."""
        batch = batch.detach().cpu()
        n = batch.shape[0]
        end_idx = start_idx + n
        if end_idx <= self.capacity:
            arr[start_idx:end_idx] = batch
        else:
            split = self.capacity - start_idx
            arr[start_idx:] = batch[:split]
            arr[:n - split] = batch[split:]

    def _write_with_shift(self, arr: torch.Tensor, batch: torch.Tensor, shift: int = 0):
        """Write batch with index shift (can be negative for back-fill)."""
        start_idx = (self.idx + shift * batch.shape[0]) % self.capacity
        self._write_batch(arr, batch, start_idx)


class SDQLReplayBuffer(RingBufferDataset):
    """
    Full-state SDQL replay buffer (CPU-only) with partial/shifted writes.
    """
    def __init__(self, conf: Config):
        # === Consume configuration parameters ===
        capacity = int(conf.steps_per_episode * conf.batch_size * conf.buffer_reuse)
        super().__init__(capacity)
        E = conf.encoded_channels
        K = conf.obs_size
        L = conf.brain_state_layers
        S = conf.brain_state_channels
        self.batch_size = conf.batch_size

        # Preallocate storage on CPU
        self.encoded     = torch.empty((capacity, E, K, K), dtype=torch.float32)
        self.state       = torch.empty((capacity, L, S, K, K), dtype=torch.float32)
        self.action      = torch.empty((capacity,), dtype=torch.long)
        self.q_target    = torch.empty((capacity,), dtype=torch.float32)
        self.r_target    = torch.empty((capacity,), dtype=torch.float32)
        self.obs_target  = torch.empty((capacity, K, K), dtype=torch.long)

        print(f"SDQLBuffer initialized with capacity {self.capacity}")

    # --- Partial write methods ---
    def add_encoded_state(self, encoded_batch, state_batch, shift: int = 0):
        assert encoded_batch.shape[0] == self.batch_size, "Batch size mismatch"
        self._write_with_shift(self.encoded, encoded_batch, shift)
        self._write_with_shift(self.state, state_batch, shift)

    def add_q_target(self, q_target_batch, shift: int = 0):
        assert q_target_batch.shape[0] == self.batch_size, "Batch size mismatch"
        self._write_with_shift(self.q_target, q_target_batch, shift)

    def add_action_reward_obs(self, action_batch, reward_batch, obs_target_batch, shift: int = 0):
        assert action_batch.shape[0] == self.batch_size, "Batch size mismatch"
        self._write_with_shift(self.action, action_batch, shift)
        self._write_with_shift(self.r_target, reward_batch, shift)
        self._write_with_shift(self.obs_target, obs_target_batch, shift)

    def advance(self):
        self._advance(batch_size=self.batch_size)

    def __getitem__(self, idx):
        # Map logical idx -> physical idx in the circular storage
        if self.full:
            start = self.idx
        else:
            start = 0
        actual_idx = (start + idx) % self.capacity
        return (
            self.encoded[actual_idx],
            self.state[actual_idx],
            self.action[actual_idx],
            self.q_target[actual_idx],
            self.r_target[actual_idx],
            self.obs_target[actual_idx],
        )

    def __len__(self):
        # Exclude the last batch_size entries (they don't have q_target yet)
        if self.full:
            return self.capacity - self.batch_size
        else:
            return max(0, self.idx - self.batch_size)


if __name__ == "__main__":
    # --- Sanity check for SDQLReplayBuffer ---
    conf = Config()  # uses your default config
    buffer = SDQLReplayBuffer(conf)

    print("Initial buffer length:", len(buffer))

    # Create dummy batches
    E, K, L, S = conf.encoded_channels, conf.obs_size, conf.brain_state_layers, conf.brain_state_channels
    batch_size = conf.batch_size

    encoded_batch = torch.randn(batch_size, E, K, K)
    state_batch   = torch.randn(batch_size, L, S, K, K)
    action_batch  = torch.randint(0, 10, (batch_size,))
    q_target_batch = torch.randn(batch_size)
    r_target_batch = torch.randn(batch_size)
    obs_target_batch = torch.randint(0, 256, (batch_size, K, K))

    # Write multiple batches
    num_batches = 5
    for i in range(num_batches):
        buffer.add_encoded_state(encoded_batch, state_batch)
        buffer.add_q_target(q_target_batch)
        buffer.add_action_reward_obs(action_batch, r_target_batch, obs_target_batch)
        buffer.advance()
        print(f"After batch {i+1}, buffer length: {len(buffer)}")

    # Check __getitem__ access
    for i in range(len(buffer)):
        e, s, a, q, r, o = buffer[i]
        assert e.shape == (E, K, K)
        assert s.shape == (L, S, K, K)
        assert a.shape == ()
        assert q.shape == ()
        assert r.shape == ()
        assert o.shape == (K, K)
    print("Sanity check passed: all shapes correct and indexing works.")

    # Check wrap-around behavior
    total_batches = (buffer.capacity // batch_size) + 2
    for _ in range(total_batches):
        buffer.add_encoded_state(encoded_batch, state_batch)
        buffer.add_q_target(q_target_batch)
        buffer.add_action_reward_obs(action_batch, r_target_batch, obs_target_batch)
        buffer.advance()
    print("Wrap-around test completed. Buffer length:", len(buffer))
