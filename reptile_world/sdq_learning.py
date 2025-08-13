# Double Deep Q-learning algorithm as in DRL_Minh_Presentation, with some differences:
# We calculate target_q values during rollout, and store (state, action, target_q) in replay buffer.
# Target Q model is cloned from online Q model before each training cycle.
# Replay buffer is circular, training cycles reuse data a few times.
# Trainer Q model can be on CUDA.
from __future__ import annotations
import copy
import torch
from torch import Tensor
from torch.utils.data import Dataset
from reptile_world.config import Config, Action, Reward
from reptile_world.grid_world import GridWorld
from reptile_world.sdqn_model import SDQNModel
from reptile_world.reptile import Reptile
from reptile_world.sdql_replay_buffer import SDQLReplayBuffer
# from core.replay_buffer import SDQLReplayBuffer
# from core.qtrainer import QTrainer
# from line_profiler_pycharm import profile


class SDQLearning:
    def __init__(self, conf: Config, world: GridWorld, animal: Reptile):

        self.conf = conf
        self.world = world
        self.animal = animal
        # === Consume configuration parameters ===
        # self.EMPTY, self.SEED, self.PLANT, self.FRUIT, self.BARR, self.ANIMAL \
        #     = conf.EMPTY, conf.SEED, conf.PLANT, conf.FRUIT, conf.BARR, conf.ANIMAL
        # self.C = conf.num_cell
        # self.CELL_REW = conf.CELL_REW  # Rewards depend on the cell type the animal moves to
        # self.CELL_STR = conf.CELL_STR
        #
        # self.STAY, self.UP, self.DOWN, self.LEFT, self.RIGHT \
        #     = conf.STAY, conf.UP, conf.DOWN, conf.LEFT, conf.RIGHT
        # self.A = conf.num_actions
        # self.delta_pos = conf.delta_pos
        # self.ACTION_STR = conf.ACTION_STR
        #
        # self.B, self.H, self.W, self.R, self.K \
        #     = conf.batch_size, conf.grid_height, conf.grid_width, conf.obs_radius, conf.obs_size

        self.gamma = conf.gamma  # Discount factor for calculating return from rewards
        self.num_episodes, self.steps_per_episode = conf.num_episodes, conf.steps_per_episode
        self.num_epochs = conf.num_epochs
        self.learning_rate0, self.learning_rate1 = conf.learning_rate0, conf.learning_rate1
        self.epsilon0, self.epsilon1 = conf.epsilon0, conf.epsilon1
        self.temp0, self.temp1 = conf.temperature0, conf.temperature1

        # Set up replay buffer
        self.buffer = SDQLReplayBuffer(conf=conf)
        self.animal.model.eval()  # This is the online Q model, needs to be always in eval() mode
        # self.animal.model.cuda()
        # Make a clone of animal.model for double deep learning.
        self.target_model = copy.deepcopy(self.animal.model)
        self.target_model.eval()  # This is the target Q model, needs to be always in eval() mode
        # self.target_model.cuda()
        # TODO: Set up trainer, this is just placeholder commented out
        # self.trainer = SDQTrainer(self.animal.model, buffer=self.buffer, device=trainer_device,
        #                         batch_size=batch_size, learning_rate=learning_rate0)
        # TODO: Set up result collector buffer, this is just placeholder commented out
        # self.result = DQLResult(capacity=num_episodes)

    # @profile
    def rollout(self) -> float:
        # action = self.world.zero_action()
        observation, _ = self.world.resolve_action(self.world.last_action)  # (B, K, K)
        state = self.animal.perceive(observation)  # (B, C, K, K)
        q_a = self.animal.estimate_q(state)  # (B, A)

        sum_reward = 0.0
        for t in range(self.steps_per_episode):
            # q_a: (B, A)
            action = self.animal.select_action(q_a)  # (B, )
            # self.world.last_action = action
            observation, reward = self.world.resolve_action(action)  # (B, K, K), (B, )

            encoded = self.animal.encode(observation=observation, last_action=action)

            # Target q values are calculated with target model, brain_state is not modified.
            with torch.no_grad():
                q_a_target, _, _ = self.target_model(encoded, self.animal.brain_state)

            # We use the online model to calculate everything.
            next_q_a, r_a, obs_a, new_brain_state = self.animal.predictA(encoded, self.animal.brain_state)

            # next_q_values = self.animal.estimate_q(next_state)  # (B, A)

            # best next action is selected based on the online model ...
            best_action = torch.argmax(next_q_a, dim=1)  # (B, )

            # ... but the next q values are calculated with target model
            # with torch.no_grad():
            #     target_q_a, _, _, _ = self.target_model(encoded, self.animal.brain_state)
            #     target_q_values = self.target_model(next_state.cuda()).cpu()  # (B, A)
            best_q = q_a_target.gather(1, best_action.unsqueeze(1)).squeeze(1)  # shape: (B,)

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
            # === PARAMETER SETUP ===
            ep_ratio = episode / (self.num_episodes - 1.0)
            lr = self.learning_rate0 + (self.learning_rate1 - self.learning_rate0) * ep_ratio
            self.trainer.set_learning_rate(new_lr=lr)
            eps = self.epsilon0 + (self.epsilon1 - self.epsilon0) * ep_ratio
            self.animal.epsilon = eps
            temp = self.temp0 + (self.temp1 - self.temp0) * ep_ratio
            self.animal.temperature = temp
            # print("eps = ", eps, "temp = ", temp)

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


class SDQLResult(Dataset):
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
