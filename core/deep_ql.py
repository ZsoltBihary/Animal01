# Double Deep Q-learning algorithm as in DRL_Minh_Presentation, with some differences:
# We calculate target_q values during rollout, and store (state, action, target_q) in replay buffer.
# Target Q model is cloned from online Q model before each training cycle.
# Replay buffer is circular, training cycles reuse data a few times.
import torch
import copy
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
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode

        self.buffer = ReplayBuffer(capacity=buffer_capacity, state_shape=animal.state_shape[1:])  # remove batch dim
        # Make a clone of animal.model for double deep learning.
        self.target_model = copy.deepcopy(self.animal.model)
        # Set up trainer
        self.num_epochs = num_epochs
        self.trainer = QTrainer(self.animal.model, buffer=self.buffer, device=trainer_device,
                                batch_size=batch_size, learning_rate=learning_rate)

    @profile
    def run(self):
        # gamma = 0.99
        action = self.world.zero_action()
        observation, _ = self.world.resolve_action(action)    # (B, K, K)
        state = self.animal.perceive(observation)     # (B, C, K, K)
        q_values = self.animal.estimate_q(state)      # (B, A)
        for episode in range(self.num_episodes):
            print("episode:", episode+1, "/", self.num_episodes)
            for t in range(self.steps_per_episode):

                action = self.animal.select_action(q_values)  # (B, )
                observation, reward = self.world.resolve_action(action)  # (B, )
                # Get next state AFTER the environment responds
                next_state = self.animal.perceive(observation)    # (B, C, K, K)
                next_q_values = self.animal.estimate_q(next_state)     # (B, A)
                # best next action is selected based on the online model ...
                best_next_action = torch.argmax(next_q_values, dim=1)  # (B, )
                # ... but the next q values are calculated with target model
                with torch.no_grad():
                    target_q_values = self.target_model(next_state)    # (B, A)
                max_next_q = target_q_values.gather(1, best_next_action.unsqueeze(1)).squeeze(1)  # shape: (B,)
                target_q = (1.0 - self.gamma) * reward + self.gamma * max_next_q
                # Store transition. We use target_q, rather than storing (state, action, reward, next_state).
                self.buffer.append(state, action, target_q)

                state = next_state
                q_values = next_q_values

            print("Buffer: ", len(self.buffer), "/", self.buffer.capacity)
            # We update target_model BEFORE the training cycle
            self.target_model.load_state_dict(self.animal.model.state_dict())
            # We initialize training model BEFORE the training cycle
            self.trainer.trainer_model.load_state_dict(self.animal.model.state_dict())
            self.trainer.train(epochs=self.num_epochs)
            # We update animal.model AFTER the training cycle
            self.animal.model.load_state_dict(self.trainer.trainer_model.state_dict())

    # def run(self):
    #     gamma = 0.99
    #     observation = self.world.get_observation()    # (B, K, K)
    #     state = self.animal.perceive(observation)     # (B, C, K, K)
    #     q_values = self.animal.estimate_q(state)      # (B, A)
    #     for episode in range(self.num_episodes):
    #         print("episode:", episode+1, "/", self.num_episodes)
    #         for t in range(self.steps_per_episode):
    #
    #             action = self.animal.select_action(q_values)  # (B, )
    #             reward = self.world.resolve_action(action)  # (B, )
    #             # self.terrain.step()
    #
    #             # Get next state AFTER the environment responds
    #             next_observation = self.world.get_observation()        # (B, K, K)
    #             next_state = self.animal.perceive(next_observation)    # (B, C, K, K)
    #             next_q_values = self.animal.estimate_q(next_state)     # (B, A)
    #             best_next_action = torch.argmax(next_q_values, dim=1)  # (B, )
    #             with torch.no_grad():
    #                 target_q_values = Q_target(next_state)  # (B, A)
    #             max_next_q = target_q_values.gather(1, best_next_action.unsqueeze(1)).squeeze(1)  # shape: (B,)
    #             target_q = (1.0 - gamma) * reward + gamma * max_next_q
    #             # Store transition
    #             self.buffer.append(state, action, target_q)
    #
    #             state = next_state
    #             q_values = next_q_values
    #
    #         Q_target.load_state_dict(Q_online.state_dict())
    #         train_q_model(Q_online, insect_optimizer, buffer, batch_size=128, epochs=1)
