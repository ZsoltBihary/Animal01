import torch
import torch.nn.functional as F
from reptile_world.config import Config, Observation, Action, Encoded, BrainState, QA, RewardA, ObservationA
from reptile_world.sdqn_model import SDQNModel
from torchinfo import summary


class Reptile:
    def __init__(self, conf: Config, model: SDQNModel, brain_state: BrainState | None = None):

        self.conf = conf
        # === Consume configuration parameters ===
        self.B = conf.batch_size
        self.C = conf.obs_channels
        self.K = conf.obs_size
        self.A = conf.num_actions
        self.L = conf.brain_state_layers
        self.S = conf.brain_state_channels
        self.epsilon = conf.epsilon0
        self.temperature = conf.temperature0
        self.brain_device = conf.brain_device

        # === Initialize class attributes ===
        self.model = model

        # Set up brain state
        if brain_state is None:
            self.brain_state = 0.99 * torch.rand(self.B, self.L, self.S, self.K, self.K)  # random brain (B, L, S, K, K)
        else:
            self.brain_state = brain_state
        self.brain_state = self.brain_state.to(self.brain_device)

    def encode(self, observation: Observation, last_action: Action) -> Encoded:
        """
        Args:
            observation: Tensor(B, K, K)
            last_action: Tensor(B,)
        Returns:
            encoded: Tensor(B, E=C+A, K, K)
        """
        # === PROCESS OBSERVATION ===
        clamped = observation.clamp(min=0, max=self.C - 1)               # (B, K, K)
        one_hot_observation = F.one_hot(clamped, num_classes=self.C)     # (B, K, K, C)
        # Mask: keep only valid values in range [0, C-1]
        valid = (observation >= 0) & (observation < self.C)              # (B, K, K)
        mask = valid.unsqueeze(-1)                                       # (B, K, K, 1)
        one_hot_observation = one_hot_observation * mask                 # (B, K, K, C)

        # === PROCESS LAST ACTION ===
        one_hot_action = F.one_hot(last_action, num_classes=self.A)      # (B, A)
        # Expand one_hot_action to (B, K, K, A)
        one_hot_action = one_hot_action.view(self.B, 1, 1, self.A)       # (B, 1, 1, A)
        one_hot_action = one_hot_action.expand(-1, self.K, self.K, -1)   # (B, K, K, A)

        # === COMBINE ===
        encoded = torch.cat([one_hot_observation, one_hot_action], dim=-1)  # (B, K, K, C+A)
        # Permute to (B, C+A, K, K)
        encoded = encoded.permute(0, 3, 1, 2).float()    # (B, C+A, K, K)
        return encoded

    def predictA(self, encoded: Encoded, brain_state: BrainState) -> (
            tuple)[QA, RewardA, ObservationA, BrainState]:
        """
        Return action-conditional predictions
        :param encoded:
        :param brain_state:
        :return:
        """
        # move input tensors to model device
        encoded = encoded.to(self.model.device)
        brain_state = brain_state.to(self.model.device)
        q_a, r_a, obs_a, new_brain_state = self.model.sdqn(encoded, brain_state)
        # move first three output tensors back to cpu, new_brain_state to brain_device
        q_a = q_a.cpu()
        r_a = r_a.cpu()
        obs_a = obs_a.cpu()
        new_brain_state = new_brain_state.to(self.brain_device)
        return q_a, r_a, obs_a, new_brain_state

    def select_action(self, q_a: QA) -> Action:
    #     """
    #     Epsilon-soft strategy, with random policy selection based on Q(s, a) values
    #     """
    #     logits = q_a / self.temperature
    #     probs = F.softmax(logits, dim=-1)
    #     probs = (1.0 - self.epsilon) * probs + self.epsilon / probs.size(-1)
    #     action = torch.multinomial(probs, num_samples=1).squeeze(1)
    #     return action
    #
    # def select_action(self, q_a: torch.Tensor) -> torch.Tensor:
        """
        Epsilon-soft strategy, robust version.
        Handles large logits, small temperatures, and avoids NaNs.
        """
        # --- Step 1: Scale by temperature ---
        logits = q_a / self.temperature

        # --- Step 2: Numerical stabilization ---
        # Subtract max along action dimension to prevent overflow in softmax
        logits = logits - logits.max(dim=-1, keepdim=True)[0]

        # --- Step 3: Softmax ---
        probs = F.softmax(logits, dim=-1)

        # --- Step 4: Epsilon-greedy smoothing ---
        probs = (1.0 - self.epsilon) * probs + self.epsilon / probs.size(-1)

        # --- Step 5: Clamp for safety ---
        probs = torch.clamp(probs, min=1e-8, max=1.0)

        # --- Step 6: Sample action ---
        action = torch.multinomial(probs, num_samples=1).squeeze(1)
        return action

    def act(self, observation: Observation, last_action: Action) -> Action:
        encoded = self.encode(observation, last_action)
        q_a, pred_r, pred_obs, new_brain_state = self.predictA(encoded, self.brain_state)
        self.brain_state = new_brain_state
        action = self.select_action(q_a)
        return action


if __name__ == "__main__":

    # Create config and model
    config = Config()
    model = SDQNModel(config)

    # Sanity check: summarize model
    summary(model, input_data=(torch.randn(config.batch_size,
                                           config.obs_channels + config.num_actions,
                                           config.obs_size,
                                           config.obs_size).to(config.model_device),
                               torch.randn(config.batch_size,
                                           config.brain_state_layers,
                                           config.brain_state_channels,
                                           config.obs_size,
                                           config.obs_size).to(config.brain_device)),
            col_names=["input_size", "output_size", "num_params", "kernel_size"],
            depth=3)

    # Create Reptile agent
    agent = Reptile(config, model)

    # Fake input
    observation = torch.randint(low=-1, high=config.obs_channels,
                                size=(config.batch_size, config.obs_size, config.obs_size))
    last_action = torch.randint(low=0, high=config.num_actions,
                                size=(config.batch_size,))

    # Run one step
    action = agent.act(observation, last_action)

    print("Selected action:", action)
    print("Sanity check passed.")
