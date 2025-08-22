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
from reptile_world.sdql_trainer import SDQLTrainer
# from line_profiler_pycharm import profile


class SDQLearning:
    def __init__(self, conf: Config, world: GridWorld, animal: Reptile):

        self.conf = conf
        self.world = world
        self.animal = animal

        # === Consume configuration parameters ===
        self.online_model = animal.model.to(device=conf.model_device)
        self.target_model = copy.deepcopy(animal.model).to(device=conf.model_device)
        self.target_model.eval()  # This is the target Q model, needs to be always in eval() mode

        self.gamma = conf.gamma  # Discount factor for calculating return from rewards
        self.num_episodes, self.steps_per_episode = conf.num_episodes, conf.steps_per_episode
        self.num_epochs = conf.num_epochs
        self.learning_rate0, self.learning_rate1 = conf.learning_rate0, conf.learning_rate1
        self.epsilon0, self.epsilon1 = conf.epsilon0, conf.epsilon1
        self.temp0, self.temp1 = conf.temperature0, conf.temperature1
        # Set up replay buffer
        self.buffer = SDQLReplayBuffer(conf=conf)
        # Set up trainer
        self.trainer = SDQLTrainer(conf=conf, model=self.online_model, buffer=self.buffer)

        # DONE: Set up result collector buffer, this is just placeholder commented out
        self.result = SDQLResult(capacity=conf.num_episodes)

    @torch.no_grad()
    def rollout(self) -> float:
        self.online_model.eval()
        self.target_model.eval()
        sum_reward = 0.0
        # Restart point: we know world state (with last_action) and animal.brain_state
        observation, reward = self.world.resolve_action(self.world.last_action)
        for t in range(self.steps_per_episode):
            # --- Step 1: Encode current observation ---
            encoded = self.animal.encode(observation, self.world.last_action)
            # --- Step 2: Save encoded & brain_state BEFORE any model call ---
            self.buffer.add_encoded_state(encoded_batch=encoded, state_batch=self.animal.brain_state)
            # --- Step 3: Q-values ---
            q_a_target = self.target_model.predict_q(encoded, self.animal.brain_state).cpu()  # does NOT advance brain_state
            q_a = self.online_model.advance_q(encoded, self.animal.brain_state).cpu()  # DOES advance brain_state in-place
            # --- Step 4: Best action (for THIS step) ---
            best_action = q_a.argmax(dim=1)
            best_q = q_a_target.gather(1, best_action.unsqueeze(1)).squeeze(1)
            # --- Step 5: Back-fill q_target for PREVIOUS step ---
            # DONE: Do we really want this condition? It messes up the restart ...
            # if t > 0:
            q_target = (1.0 - self.gamma) * reward + self.gamma * best_q
            self.buffer.add_q_target(q_target_batch=q_target, shift=-1)
            # --- Step 6: Action selection for environment ---
            action = self.animal.select_action(q_a)
            # --- Step 7: Step the environment ---
            observation, reward = self.world.resolve_action(action=action)  # This also sets world.last_action
            # --- Step 8: Complete this step's buffer entry ---
            # Scale down reward
            self.buffer.add_action_reward_obs(
                action_batch=action,
                reward_batch=(1.0 - self.gamma) * reward,
                obs_target_batch=observation
            )
            # --- Step 9: End of rollout step, advance buffer ---
            self.buffer.advance()
            sum_reward += reward.mean().item()
        avg_reward = sum_reward / self.steps_per_episode
        return avg_reward

    # @profile
    def run(self) -> SDQLResult:
        for episode in range(self.num_episodes):
            # === PARAMETER SETUP ===
            ep_ratio = episode / (self.num_episodes - 1.0)
            lr = self.learning_rate0 + (self.learning_rate1 - self.learning_rate0) * ep_ratio
            self.trainer.set_learning_rate(new_lr=lr)
            eps = self.epsilon0 + (self.epsilon1 - self.epsilon0) * ep_ratio
            self.animal.epsilon = eps
            temp = self.temp0 + (self.temp1 - self.temp0) * ep_ratio
            self.animal.temperature = temp
            print("eps = ", eps, "temp = ", temp)

            # === ROLLOUT ===
            avg_reward = self.rollout()
            print("episode:", episode + 1, "/", self.num_episodes, "  Avg_reward:", int(avg_reward*10) / 10.0)

            # === MODEL UPDATES and TRAINING ===
            print("Buffer: ", len(self.buffer), "/", self.buffer.capacity)
            # We update target_model BEFORE the training cycle
            self.target_model.load_state_dict(self.animal.model.state_dict())
            # We initialize trainer model BEFORE the training cycle
            # self.trainer.model.load_state_dict(self.animal.model.state_dict())
            self.trainer.fit()
            # avg_loss = self.trainer.fit()
            # We update animal.model AFTER the training cycle
            # self.animal.model.load_state_dict(self.trainer.model.state_dict())

            # === SAVE RESULTS ===
            self.result.append(avg_reward=avg_reward,
                               avg_Q_error=3.14)
        return self.result


class SDQLResult(Dataset):
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0

        self.avg_rewards = torch.empty((capacity, ), dtype=torch.float32)
        self.avg_Q_errors = torch.empty((capacity, ), dtype=torch.float32)  # sqrt(loss)

    def append(self, avg_reward: float, avg_Q_error: float):
        """
        Append a row of results to the buffer.
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


# --- Sanity check for SDQLReplayBuffer ---
if __name__ == "__main__":
    from reptile_world.model_blocks import InputBlock, InnerBlock, DuelingQHead, RewardHead, ObsHead

    config = Config()
    world = GridWorld(conf=config)

    model = SDQNModel(
        conf=config,
        has_state=False,
        return_aux=True,
        input_block_cls=InputBlock,
        inner_block_cls=InnerBlock,
        q_head_cls=DuelingQHead,
        r_head_cls=RewardHead,
        obs_head_cls=ObsHead
    )
    animal = Reptile(conf=config, model=model)

    sdql = SDQLearning(conf=config, world=world, animal=animal)
    sdql.run()
