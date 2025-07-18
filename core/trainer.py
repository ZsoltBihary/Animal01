import torch


def train_q_model(Q_online, optimizer, buffer, batch_size=128, epochs=1, loss_fn=None, device=None):
    """
    Trains the online Q model on experience replay data.

    Args:
        Q_online:     The online Q-network (torch.nn.Module).
        optimizer:    The optimizer for Q_online.
        buffer:       A CircularQBuffer containing (state, action, target_q).
        batch_size:   Batch size for training.
        epochs:       Number of full passes over the buffer.
        loss_fn:      Loss function (MSE or Huber). Defaults to MSELoss.
        device:       Device to train on. If None, inferred from model.
    """
    if loss_fn is None:
        loss_fn = torch.nn.MSELoss()

    if device is None:
        device = next(Q_online.parameters()).device

    Q_online.train()
    loader = torch.utils.data.DataLoader(buffer, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0

        for state_batch, action_batch, target_q_batch in loader:
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            target_q_batch = target_q_batch.to(device)

            predicted_q_values = Q_online(state_batch)  # (B, A)
            predicted_q = predicted_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

            loss = loss_fn(predicted_q, target_q_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        print(f"[Epoch {epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")
