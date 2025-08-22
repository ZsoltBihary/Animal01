import torch
from torch.utils.data import DataLoader


# --------------------------
# Shared supervised train step
# --------------------------
def train_batch_step(model, batch, optimizer, criterion_q, criterion_r=None, criterion_obs=None,
                     lambda_r=1.0, lambda_obs=1.0, stateful=False, aux=False):
    """One gradient step for a single batch."""
    optimizer.zero_grad()

    # Unpack batch depending on statefulness / aux
    if not stateful and not aux:
        encoded, action, q_target = batch
        q_a = model(encoded)
        q_pred = q_a[torch.arange(len(action)), action]
        loss = criterion_q(q_pred, q_target)

    elif not stateful and aux:
        encoded, action, q_target, r_target, obs_target = batch
        q_a, r_a, obs_a = model(encoded)
        q_pred = q_a[torch.arange(len(action)), action]
        r_pred = r_a[torch.arange(len(action)), action]
        obs_pred = obs_a[torch.arange(len(action)), action]
        loss_q = criterion_q(q_pred, q_target)
        loss_r = criterion_r(r_pred, r_target)
        loss_o = criterion_obs(obs_pred, obs_target)
        loss = loss_q + lambda_r * loss_r + lambda_obs * loss_o

    elif stateful and not aux:
        encoded, state, action, q_target = batch
        q_a = model(encoded, state)
        q_pred = q_a[torch.arange(len(action)), action]
        loss = criterion_q(q_pred, q_target)

    else:  # stateful and aux
        encoded, state, action, q_target, r_target, obs_target = batch
        q_a, r_a, obs_a = model(encoded, state)
        q_pred = q_a[torch.arange(len(action)), action]
        r_pred = r_a[torch.arange(len(action)), action]
        obs_pred = obs_a[torch.arange(len(action)), action]
        loss_q = criterion_q(q_pred, q_target)
        loss_r = criterion_r(r_pred, r_target)
        loss_o = criterion_obs(obs_pred, obs_target)
        loss = loss_q + lambda_r * loss_r + lambda_obs * loss_o

    loss.backward()
    optimizer.step()
    return loss.item()


# --------------------------
# Four thin wrappers
# --------------------------
def train_dql(model, dataloader, optimizer, criterion_q, device):
    """Stateless, Q-only"""
    model.train()
    for batch in dataloader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        train_batch_step(model, batch, optimizer, criterion_q,
                         stateful=False, aux=False)


def train_dql_aux(model, dataloader, optimizer, criterion_q, criterion_r, criterion_obs,
                  lambda_r, lambda_obs, device):
    """Stateless, Q+Aux"""
    model.train()
    for batch in dataloader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        train_batch_step(model, batch, optimizer, criterion_q, criterion_r, criterion_obs,
                         lambda_r, lambda_obs, stateful=False, aux=True)


def train_sdql(model, dataloader, optimizer, criterion_q, device):
    """Stateful, Q-only"""
    model.train()
    for batch in dataloader:
        # Important: state should be on CPU or detached to avoid autograd issues
        batch = [b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch]
        train_batch_step(model, batch, optimizer, criterion_q,
                         stateful=True, aux=False)


def train_sdql_aux(model, dataloader, optimizer, criterion_q, criterion_r, criterion_obs,
                   lambda_r, lambda_obs, device):
    """Stateful, Q+Aux"""
    model.train()
    for batch in dataloader:
        batch = [b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b for b in batch]
        train_batch_step(model, batch, optimizer, criterion_q, criterion_r, criterion_obs,
                         lambda_r, lambda_obs, stateful=True, aux=True)


# --------------------------
# Example sanity check
# --------------------------
if __name__ == "__main__":
    import torch.nn as nn

    # Dummy model examples
    class StatelessModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 4)  # q_a only

        def forward(self, x):
            return self.fc(x)

    class StatelessModelAux(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_q = nn.Linear(10, 4)
            self.fc_r = nn.Linear(10, 4)
            self.fc_o = nn.Linear(10, 4)

        def forward(self, x):
            return self.fc_q(x), self.fc_r(x), self.fc_o(x)

    # Fake data for stateless Q-only
    dataset = [(torch.randn(10), torch.tensor(1), torch.randn(())) for _ in range(32)]
    dataloader = DataLoader(dataset, batch_size=8)

    model = StatelessModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_q = nn.MSELoss()

    # Run one epoch of training
    train_dql(model, dataloader, optimizer, criterion_q, device="cpu")
    print("Stateless Q-only training step done.")
