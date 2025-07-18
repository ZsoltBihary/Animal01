# Deep Q-learning algorithm as in DRL_Minh_Presentation
# Double DQL
# This implements the "Insect" animal type
import torch
# import torch.nn as nn
import torch.nn.functional as F
# from torch import Tensor
from core.old_terrain import Terrain
from core.animal import Insect
from core.world import World
from core.replay_buffer import ReplayBuffer


# from utils.grid_renderer import GridRenderer


class DeepQLearning:
    def __init__(self, terrain: Terrain, insect: Insect,
                 num_episodes: int, steps_per_episode: int):
        self.terrain = terrain
        self.insect = insect
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        buff_cap = 2 * num_t * B
        self.buffer = ReplayBuffer(capacity=steps_per_episode*B)

    def run(self):
        gamma = 0.99
        observation = self.terrain.get_observation()  # (B, K, K)
        state = self.insect.perceive(observation)     # (B, C, K, K)
        q_values = self.insect.estimate_q(state)      # (B, A)
        for episode in range(self.num_episodes):
            print("episode:", episode+1, "/", self.num_episodes)
            for t in range(self.steps_per_episode):

                action = self.insect.select_action(q_values, epsilon=0.1)  # (B, )
                reward = self.terrain.resolve_action(action)  # (B, )
                self.terrain.step()

                # Get next state AFTER the environment responds
                next_observation = self.terrain.get_observation()      # (B, K, K)
                next_state = self.insect.perceive(next_observation)    # (B, C, K, K)
                next_q_values = self.insect.estimate_q(next_state)     # (B, A)
                best_next_action = torch.argmax(next_q_values, dim=1)  # (B, )
                with torch.no_grad():
                    target_q_values = Q_target(next_state)  # (B, A)
                max_next_q = target_q_values.gather(1, best_next_action.unsqueeze(1)).squeeze(1)  # shape: (B,)
                target_q = (1.0 - gamma) * reward + gamma * max_next_q
                # Store transition
                self.buffer.append(state, action, target_q)

                state = next_state
                q_values = next_q_values

            Q_target.load_state_dict(Q_online.state_dict())
            train_q_model(Q_online, insect_optimizer, buffer, batch_size=128, epochs=1)
#

# food_dens, poison_dens = 0.05, 0.35
# output_mp4_path = "roach07.mp4"
# H, W = 9, 11
# main_ch = 64
# num_episode = 10
# num_t = 100
# B = 128        # population-batch size
# buff_cap = 2 * num_t * B
# terrain = Terrain.random(B=B, H=H, W=W, R=R,
#                          food_density=food_dens, poison_density=poison_dens)
# insect = Insect(main_ch=main_ch)
# buffer = ReplayBuffer(capacity=buff_cap, state_shape=(C, K, K))
# world = World(terrain=terrain, animal=insect)
#
# # batch_index = torch.arange(B)
# Q_online = insect.q_model
# Q_target = QInsect01()
#
# insect_optimizer = torch.optim.Adam(Q_online.parameters(), lr=0.0001)
#
#
# gamma = 0.99
# observation = terrain.get_observation()  # (B, K, K)
# state = insect.perceive(observation)     # (B, C, K, K)
# q_values = insect.estimate_q(state)      # (B, A)
#
# # world.simulate(n_steps=10, verbose=1)
#
# for episode in range(num_episode):
#     print("episode:", episode+1, "/", num_episode)
#     # terrain.print()
#     for t in range(num_t):
#
#         action = insect.select_action(q_values, epsilon=0.1)  # (B, )
#         reward = terrain.resolve_action(action)  # (B, )
#         terrain.step()
#
#         # Get next state AFTER the environment responds
#         next_observation = terrain.get_observation()    # (B, K, K)
#         next_state = insect.perceive(next_observation)  # (B, C, K, K)
#         next_q_values = insect.estimate_q(next_state)   # (B, A)
#         best_next_action = torch.argmax(next_q_values, dim=1)  # (B, )
#         with torch.no_grad():
#             target_q_values = Q_target(next_state)  # (B, A)
#         max_next_q = target_q_values.gather(1, best_next_action.unsqueeze(1)).squeeze(1)  # shape: (B,)
#         target_q = (1.0 - gamma) * reward + gamma * max_next_q
#         # Store transition
#         buffer.append(state, action, target_q)
#
#         state = next_state
#         q_values = next_q_values
#
#     Q_target.load_state_dict(Q_online.state_dict())
#     train_q_model(Q_online, insect_optimizer, buffer, batch_size=128, epochs=1)
#
# world.simulate(n_steps=200, verbose=1, save_history=True)
# history = world.history

# renderer = GridRenderer(H=H, W=W, cell_size=32)
# Display animation
# renderer.play_episode(history, delay_ms=500)

# Save as GIF
# output_gif_path = "roach01.gif"
# renderer.save_episode_as_gif(history, gif_path=output_gif_path, delay_ms=1000)

# renderer.save_episode_as_mp4(history, video_path=output_mp4_path, fps=2)

# def train_q_model(Q_online, optimizer, buffer, batch_size=128, epochs=1, loss_fn=None, device=None):
#     """
#     Trains the online Q model on experience replay data.
#
#     Args:
#         Q_online:     The online Q-network (torch.nn.Module).
#         optimizer:    The optimizer for Q_online.
#         buffer:       A CircularQBuffer containing (state, action, target_q).
#         batch_size:   Batch size for training.
#         epochs:       Number of full passes over the buffer.
#         loss_fn:      Loss function (MSE or Huber). Defaults to MSELoss.
#         device:       Device to train on. If None, inferred from model.
#     """
#     if loss_fn is None:
#         loss_fn = torch.nn.MSELoss()
#
#     if device is None:
#         device = next(Q_online.parameters()).device
#
#     Q_online.train()
#     loader = torch.utils.data.DataLoader(buffer, batch_size=batch_size, shuffle=True)
#
#     for epoch in range(epochs):
#         total_loss = 0.0
#         count = 0
#
#         for state_batch, action_batch, target_q_batch in loader:
#             state_batch = state_batch.to(device)
#             action_batch = action_batch.to(device)
#             target_q_batch = target_q_batch.to(device)
#
#             predicted_q_values = Q_online(state_batch)  # (B, A)
#             predicted_q = predicted_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
#
#             loss = loss_fn(predicted_q, target_q_batch)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             count += 1
#
#         avg_loss = total_loss / max(count, 1)
#         print(f"[Epoch {epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")
