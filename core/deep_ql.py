# Double Deep Q-learning algorithm as in DRL_Minh_Presentation, with some differences:
# We calculate target_q values during rollout, and store (state, action, target_q) in replay buffer.
# Target Q model is cloned from online Q model before each training cycle.
# Replay buffer is circular, training cycles reuse data a few times.
# Trainer Q model can be on CUDA.
from __future__ import annotations
import torch
import copy
from torch.utils.data import Dataset
from core.animal import Metazoan
from core.world import World
from core.replay_buffer import ReplayBuffer
from core.qtrainer import QTrainer
from line_profiler_pycharm import profile


class DeepQLearning:
    def __init__(self, world: World, animal: Metazoan,
                 gamma: float,
                 num_episodes: int, steps_per_episode: int, buffer_capacity: int,
                 num_epochs: int, batch_size: int, learning_rate: float,
                 trainer_device: torch.device):
        self.world = world
        self.animal = animal
        self.animal.model.eval()  # This is the online Q model, needs to be always in eval() mode
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode

        self.buffer = ReplayBuffer(capacity=buffer_capacity, state_shape=animal.state_shape[1:])  # remove batch dim
        # Make a clone of animal.model for double deep learning.
        self.target_model = copy.deepcopy(self.animal.model)
        self.target_model.eval()  # This is the target Q model, needs to be always in eval() mode
        # Set up trainer
        self.num_epochs = num_epochs
        self.trainer = QTrainer(self.animal.model, buffer=self.buffer, device=trainer_device,
                                batch_size=batch_size, learning_rate=learning_rate)
        self.result = DQLResult(capacity=num_episodes)

    @profile
    def rollout(self) -> float:
        action = self.world.zero_action()
        observation, _ = self.world.resolve_action(action)  # (B, K, K)
        state = self.animal.perceive(observation)  # (B, C, K, K)
        q_values = self.animal.estimate_q(state)  # (B, A)

        sum_reward = 0.0
        for t in range(self.steps_per_episode):
            action = self.animal.select_action(q_values)  # (B, )
            observation, reward = self.world.resolve_action(action)  # (B, )
            # Get next state AFTER the environment responds
            next_state = self.animal.perceive(observation)  # (B, C, K, K)
            next_q_values = self.animal.estimate_q(next_state)  # (B, A)
            # best next action is selected based on the online model ...
            best_next_action = torch.argmax(next_q_values, dim=1)  # (B, )
            # ... but the next q values are calculated with target model
            with torch.no_grad():
                target_q_values = self.target_model(next_state)  # (B, A)
            max_next_q = target_q_values.gather(1, best_next_action.unsqueeze(1)).squeeze(1)  # shape: (B,)
            target_q = (1.0 - self.gamma) * reward + self.gamma * max_next_q
            # Store transition. We use target_q, rather than storing (state, action, reward, next_state).
            self.buffer.append(state, action, target_q)

            state = next_state
            q_values = next_q_values
            sum_reward += torch.mean(reward).item()

        avg_reward = sum_reward / self.steps_per_episode
        return avg_reward

    @profile
    def run(self) -> DQLResult:
        for episode in range(self.num_episodes):
            # === ROLLOUT ===
            avg_reward = self.rollout()
            print("episode:", episode + 1, "/", self.num_episodes, "  Avg_reward:", int(avg_reward*10) / 10.0)

            # === MODEL UPDATES and TRAINING ===
            # print("Buffer: ", len(self.buffer), "/", self.buffer.capacity)
            # We update target_model BEFORE the training cycle
            self.target_model.load_state_dict(self.animal.model.state_dict())
            # We initialize trainer model BEFORE the training cycle
            self.trainer.model.load_state_dict(self.animal.model.state_dict())
            avg_loss = self.trainer.train(epochs=self.num_epochs)
            # We update animal.model AFTER the training cycle
            self.animal.model.load_state_dict(self.trainer.model.state_dict())

            # === SAVE RESULTS ===
            self.result.append(avg_reward=avg_reward,
                               avg_Q_error=avg_loss ** 0.5)
        return self.result


class DQLResult(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0

        self.avg_rewards = torch.empty((capacity, ), dtype=torch.float32)
        self.avg_Q_errors = torch.empty((capacity, ), dtype=torch.float32)  # sqrt(loss)

    def append(self, avg_reward: float, avg_Q_error: float):
        """
        Append a batch of results to the buffer.
        Args:
            avg_reward:
            avg_Q_error:
        """
        self.avg_rewards[self.size] = avg_reward
        self.avg_Q_errors[self.size] = avg_Q_error

        self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.avg_rewards[idx],
                self.avg_Q_errors[idx])

    def print(self):
        """
        Prints the stored data in tabular format with 2 decimal places.
        """
        print("-" * 35)
        print("DQL Results")
        print(f"{'Index':<8} {'Avg Reward':<12} {'Avg Q Error':<12}")
        print("-" * 35)
        for i in range(self.size):
            reward = self.avg_rewards[i].item()
            q_error = self.avg_Q_errors[i].item()
            print(f"{i:<8} {reward:<12.2f} {q_error:<12.2f}")
