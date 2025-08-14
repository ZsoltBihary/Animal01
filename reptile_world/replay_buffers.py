import torch
from torch.utils.data import Dataset, DataLoader


class RingBufferDataset(Dataset):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def handle_expansion(self, batch_size: int):
        # Placeholder for future buffer expansion logic
        raise ValueError(
            f"Batch size {batch_size} exceeds buffer capacity {self.capacity}. Expansion not implemented."
        )

    def _advance(self, batch_size: int):
        if batch_size >= self.capacity:
            self.handle_expansion(batch_size)
        end = self.idx + batch_size
        if end >= self.capacity:
            self.full = True
        self.idx = end % self.capacity

    def _write_batch(self, arr: torch.Tensor, batch: torch.Tensor, start_idx: int):
        # Always store detached CPU tensor
        batch = batch.detach().cpu()
        n = batch.shape[0]
        end_idx = start_idx + n
        if end_idx <= self.capacity:
            arr[start_idx:end_idx] = batch
        else:
            split = self.capacity - start_idx
            arr[start_idx:] = batch[:split]
            arr[:n - split] = batch[split:]


# === 4 buffer variants ===

class DQLBuffer(RingBufferDataset):
    def __init__(self, capacity, encoded_shape):
        super().__init__(capacity)
        self.encoded = torch.empty((capacity,) + encoded_shape, dtype=torch.float32)
        self.action = torch.empty((capacity,), dtype=torch.long)
        self.q_target = torch.empty((capacity,), dtype=torch.float32)

    def add_batch(self, encoded_batch, action_batch, q_target_batch):
        bsz = encoded_batch.shape[0]
        self._write_batch(self.encoded, encoded_batch, self.idx)
        self._write_batch(self.action, action_batch, self.idx)
        self._write_batch(self.q_target, q_target_batch, self.idx)
        self._advance(bsz)

    def __getitem__(self, idx):
        return self.encoded[idx], self.action[idx], self.q_target[idx]


class DQLAuxBuffer(RingBufferDataset):
    def __init__(self, capacity, encoded_shape, obs_shape):
        super().__init__(capacity)
        self.encoded = torch.empty((capacity,) + encoded_shape, dtype=torch.float32)
        self.action = torch.empty((capacity,), dtype=torch.long)
        self.q_target = torch.empty((capacity,), dtype=torch.float32)
        self.r_target = torch.empty((capacity,), dtype=torch.float32)
        self.obs_target = torch.empty((capacity,) + obs_shape, dtype=torch.float32)

    def add_batch(self, encoded_batch, action_batch, q_target_batch, r_target_batch, obs_target_batch):
        bsz = encoded_batch.shape[0]
        self._write_batch(self.encoded, encoded_batch, self.idx)
        self._write_batch(self.action, action_batch, self.idx)
        self._write_batch(self.q_target, q_target_batch, self.idx)
        self._write_batch(self.r_target, r_target_batch, self.idx)
        self._write_batch(self.obs_target, obs_target_batch, self.idx)
        self._advance(bsz)

    def __getitem__(self, idx):
        return (
            self.encoded[idx],
            self.action[idx],
            self.q_target[idx],
            self.r_target[idx],
            self.obs_target[idx],
        )


class SDQLQOnlyBuffer(RingBufferDataset):
    def __init__(self, capacity, encoded_shape, state_shape):
        super().__init__(capacity)
        self.encoded = torch.empty((capacity,) + encoded_shape, dtype=torch.float32)
        self.state = torch.empty((capacity,) + state_shape, dtype=torch.float32)
        self.action = torch.empty((capacity,), dtype=torch.long)
        self.q_target = torch.empty((capacity,), dtype=torch.float32)

    def add_batch(self, encoded_batch, state_batch, action_batch, q_target_batch):
        bsz = encoded_batch.shape[0]
        self._write_batch(self.encoded, encoded_batch, self.idx)
        self._write_batch(self.state, state_batch, self.idx)
        self._write_batch(self.action, action_batch, self.idx)
        self._write_batch(self.q_target, q_target_batch, self.idx)
        self._advance(bsz)

    def __getitem__(self, idx):
        return self.encoded[idx], self.state[idx], self.action[idx], self.q_target[idx]


class SDQLBuffer(RingBufferDataset):
    def __init__(self, capacity, encoded_shape, state_shape, obs_shape):
        super().__init__(capacity)
        self.encoded = torch.empty((capacity,) + encoded_shape, dtype=torch.float32)
        self.state = torch.empty((capacity,) + state_shape, dtype=torch.float32)
        self.action = torch.empty((capacity,), dtype=torch.long)
        self.q_target = torch.empty((capacity,), dtype=torch.float32)
        self.r_target = torch.empty((capacity,), dtype=torch.float32)
        self.obs_target = torch.empty((capacity,) + obs_shape, dtype=torch.float32)

    def add_batch(self, encoded_batch, state_batch, action_batch, q_target_batch, r_target_batch, obs_target_batch):
        bsz = encoded_batch.shape[0]
        self._write_batch(self.encoded, encoded_batch, self.idx)
        self._write_batch(self.state, state_batch, self.idx)
        self._write_batch(self.action, action_batch, self.idx)
        self._write_batch(self.q_target, q_target_batch, self.idx)
        self._write_batch(self.r_target, r_target_batch, self.idx)
        self._write_batch(self.obs_target, obs_target_batch, self.idx)
        self._advance(bsz)

    def __getitem__(self, idx):
        return (
            self.encoded[idx],
            self.state[idx],
            self.action[idx],
            self.q_target[idx],
            self.r_target[idx],
            self.obs_target[idx],
        )


# === Sanity checks ===
if __name__ == "__main__":
    cap = 5
    enc_shape = (3,)
    state_shape = (2,)
    obs_shape = (4,)

    print("=== DQLBuffer ===")
    buf = DQLBuffer(cap, enc_shape)
    buf.add_batch(torch.randn(3, *enc_shape), torch.randint(0, 5, (3,)), torch.randn(3))
    buf.add_batch(torch.randn(4, *enc_shape), torch.randint(0, 5, (4,)), torch.randn(4))
    dl = DataLoader(buf, batch_size=2, shuffle=True)
    for batch in dl:
        print(batch)

    print("\n=== SDQLBuffer ===")
    buf2 = SDQLBuffer(cap, enc_shape, state_shape, obs_shape)
    buf2.add_batch(
        torch.randn(3, *enc_shape).cuda(),  # GPU input test
        torch.randn(3, *state_shape).cuda(),
        torch.randint(0, 5, (3,), device='cuda'),
        torch.randn(3).cuda(),
        torch.randn(3).cuda(),
        torch.randn(3, *obs_shape).cuda()
    )
    buf2.add_batch(
        torch.randn(4, *enc_shape).cuda(),
        torch.randn(4, *state_shape).cuda(),
        torch.randint(0, 5, (4,), device='cuda'),
        torch.randn(4).cuda(),
        torch.randn(4).cuda(),
        torch.randn(4, *obs_shape).cuda()
    )
    dl2 = DataLoader(buf2, batch_size=2, shuffle=True)
    for batch in dl2:
        print(batch)


# import torch
# from torch.utils.data import Dataset, DataLoader
#
#
# class RingBufferDataset(Dataset):
#     def __init__(self, capacity: int):
#         self.capacity = capacity
#         self.idx = 0
#         self.full = False
#
#     def __len__(self):
#         return self.capacity if self.full else self.idx
#
#     def _advance(self, batch_size: int):
#         if batch_size >= self.capacity:
#             self.idx = 0
#             self.full = True
#         else:
#             end = self.idx + batch_size
#             if end >= self.capacity:
#                 self.full = True
#             self.idx = end % self.capacity
#
#     def _write_batch(self, arr: torch.Tensor, batch: torch.Tensor, start_idx: int):
#         n = batch.shape[0]
#         end_idx = start_idx + n
#         if end_idx <= self.capacity:
#             arr[start_idx:end_idx] = batch
#         else:
#             split = self.capacity - start_idx
#             arr[start_idx:] = batch[:split]
#             arr[:n - split] = batch[split:]
#
#
# # === 4 buffer variants ===
#
# # A1: Stateless (Residual) + Q-only
# class DQLBuffer(RingBufferDataset):
#     def __init__(self, capacity, encoded_shape):
#         super().__init__(capacity)
#         self.encoded = torch.empty((capacity,) + encoded_shape, dtype=torch.float32)
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
# # A2: Stateless (Residual) + Q+Aux
# class DQLAuxBuffer(RingBufferDataset):
#     def __init__(self, capacity, encoded_shape, obs_shape):
#         super().__init__(capacity)
#         self.encoded = torch.empty((capacity,) + encoded_shape, dtype=torch.float32)
#         self.action = torch.empty((capacity,), dtype=torch.long)
#         self.q_target = torch.empty((capacity,), dtype=torch.float32)
#         self.r_target = torch.empty((capacity,), dtype=torch.float32)
#         self.obs_target = torch.empty((capacity,) + obs_shape, dtype=torch.float32)
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
# # B1: Stateful (Brainy) + Q-only
# class SDQLQOnlyBuffer(RingBufferDataset):
#     def __init__(self, capacity, encoded_shape, state_shape):
#         super().__init__(capacity)
#         self.encoded = torch.empty((capacity,) + encoded_shape, dtype=torch.float32)
#         self.state = torch.empty((capacity,) + state_shape, dtype=torch.float32)
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
#
#
# # B2: Stateful (Brainy) + Q+Aux
# class SDQLBuffer(RingBufferDataset):
#     def __init__(self, capacity, encoded_shape, state_shape, obs_shape):
#         super().__init__(capacity)
#         self.encoded = torch.empty((capacity,) + encoded_shape, dtype=torch.float32)
#         self.state = torch.empty((capacity,) + state_shape, dtype=torch.float32)
#         self.action = torch.empty((capacity,), dtype=torch.long)
#         self.q_target = torch.empty((capacity,), dtype=torch.float32)
#         self.r_target = torch.empty((capacity,), dtype=torch.float32)
#         self.obs_target = torch.empty((capacity,) + obs_shape, dtype=torch.float32)
#
#     def add_batch(self, encoded_batch, state_batch, action_batch, q_target_batch, r_target_batch, obs_target_batch):
#         bsz = encoded_batch.shape[0]
#         self._write_batch(self.encoded, encoded_batch, self.idx)
#         self._write_batch(self.state, state_batch, self.idx)
#         self._write_batch(self.action, action_batch, self.idx)
#         self._write_batch(self.q_target, q_target_batch, self.idx)
#         self._write_batch(self.r_target, r_target_batch, self.idx)
#         self._write_batch(self.obs_target, obs_target_batch, self.idx)
#         self._advance(bsz)
#
#     def __getitem__(self, idx):
#         return (
#             self.encoded[idx],
#             self.state[idx],
#             self.action[idx],
#             self.q_target[idx],
#             self.r_target[idx],
#             self.obs_target[idx],
#         )
#
#
# # === Sanity checks ===
# if __name__ == "__main__":
#     cap = 5
#     enc_shape = (3,)
#     state_shape = (2,)
#     obs_shape = (4,)
#
#     print("=== DQLBuffer ===")
#     buf = DQLBuffer(cap, enc_shape)
#     buf.add_batch(torch.randn(3, *enc_shape), torch.randint(0, 5, (3,)), torch.randn(3))
#     buf.add_batch(torch.randn(4, *enc_shape), torch.randint(0, 5, (4,)), torch.randn(4))
#     dl = DataLoader(buf, batch_size=2, shuffle=True)
#     for batch in dl:
#         print(batch)
#
#     print("\n=== SDQLBuffer ===")
#     buf2 = SDQLBuffer(cap, enc_shape, state_shape, obs_shape)
#     buf2.add_batch(
#         torch.randn(3, *enc_shape),
#         torch.randn(3, *state_shape),
#         torch.randint(0, 5, (3,)),
#         torch.randn(3),
#         torch.randn(3),
#         torch.randn(3, *obs_shape)
#     )
#     buf2.add_batch(
#         torch.randn(4, *enc_shape),
#         torch.randn(4, *state_shape),
#         torch.randint(0, 5, (4,)),
#         torch.randn(4),
#         torch.randn(4),
#         torch.randn(4, *obs_shape)
#     )
#     dl2 = DataLoader(buf2, batch_size=2, shuffle=True)
#     for batch in dl2:
#         print(batch)
