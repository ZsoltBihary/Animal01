import torch
from torch.utils.data import Dataset


class StateTargetReplayBuffer(Dataset):
    def __init__(self, time_size, batch_size, state_shape, target_shape, device='cpu'):
        self.T = time_size
        self.B = batch_size
        self.ptr = 0  # current write index along the time axis
        self.states = torch.zeros((self.T, self.B) + state_shape, device=device)
        self.targets = torch.zeros((self.T, self.B) + target_shape, device=device)
        self.filled = False  # becomes True after first wraparound

    def push(self, state_batch, target_batch):
        assert state_batch.shape[0] == self.B
        assert target_batch.shape[0] == self.B

        self.states[self.ptr] = state_batch
        self.targets[self.ptr] = target_batch

        self.ptr = (self.ptr + 1) % self.T
        if self.ptr == 0:
            self.filled = True

    def __len__(self):
        # Only T-1 rows are usable for (state, target) alignment
        return (self.T - 1) * self.B if self.filled else max(0, (self.ptr - 1) * self.B)

    def __getitem__(self, idx):
        assert self.filled, "Replay buffer not full yet — do not sample before buffer is fully populated."
        row = idx // self.B
        col = idx % self.B

        # Map to aligned (state, target) pair: state[t], target[t+1]
        i = (self.ptr + row) % self.T
        j = (self.ptr + row + 1) % self.T

        state = self.states[i, col]
        target = self.targets[j, col]

        return state, target
