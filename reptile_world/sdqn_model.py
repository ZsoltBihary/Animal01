import torch
from torch import nn
from reptile_world.config import Config, Encoded, BrainSlice, BrainState, QA, RewardA, ObservationA
from reptile_world.model_blocks import InputBlock, InnerBlock, DuelingQHead, RewardHead, ObsHead
from torchinfo import summary


class BrainBlock(nn.Module):
    """
    A single 'brain block' in the SDQN trunk.

    Wraps an inner block with a residual connection and ReLU activation.
    """
    def __init__(self, conf: Config, inner_block_cls: type[nn.Module]):
        super().__init__()
        self.inner_block = inner_block_cls(conf=conf)
        self.relu = nn.ReLU()

    def forward(self, x: BrainSlice, s_i: BrainSlice) -> BrainSlice:
        return self.relu(x + self.inner_block(s_i))


class SDQNModel(nn.Module):
    """
    Stateful Deep Q-Network (SDQN) model.

    The model consists of:
    - An input processing block
    - A stack ("brain tower") of BrainBlocks with independent parameters
    - Output heads for Q-values, rewards, and observations

    Provides multiple forward-like methods for different RL contexts:
    - forward(): training prediction without state modification
    - predict_q(): Q-values only, no state modification
    - advance_with_aux(): advance state + return Q-values and aux predictions
    - advance_q_only(): advance state + return Q-values only

    Args:
        conf (Config): Model configuration and hyperparameters.
        has_state (bool): If True, model uses a multi-layer persistent brain state.
        return_aux (bool): Whether to compute and return auxiliary outputs by default.
        input_block_cls (type[nn.Module]): Class for the input block.
        inner_block_cls (type[nn.Module]): Class for the inner block in each BrainBlock.
        q_head_cls (type[nn.Module]): Class for the Q-value head.
        r_head_cls (type[nn.Module]): Class for the reward head.
        obs_head_cls (type[nn.Module]): Class for the observation head.
    """
    def __init__(self, conf: Config,
                 has_state: bool, return_aux: bool,
                 input_block_cls: type[nn.Module],
                 inner_block_cls: type[nn.Module],
                 q_head_cls: type[nn.Module],
                 r_head_cls: type[nn.Module],
                 obs_head_cls: type[nn.Module]):
        super().__init__()
        self.conf = conf
        self.has_state = has_state
        self.return_aux = return_aux
        self.device = conf.model_device
        self.L = conf.brain_state_layers

        # Input and heads
        self.input_block = input_block_cls(conf)
        self.q_head = q_head_cls(conf)
        self.r_head = r_head_cls(conf)
        self.obs_head = obs_head_cls(conf)

        # Brain tower (no shared weights between blocks)
        self.trunk_blocks = nn.ModuleList([
            BrainBlock(conf, inner_block_cls)
            for _ in range(self.L)
        ])

        self.to(self.device)

    def _sdqn_step(self,
                   encoded: Encoded,
                   state: BrainState | None = None,
                   modify_state: bool = False,
                   return_r: bool = True,
                   return_obs: bool = True) -> tuple[QA, RewardA | None, ObservationA | None]:
        """
        Internal core step for SDQN forward logic.

        Args:
            encoded (Encoded): Encoded environment input.
            state (BrainState | None): Brain state tensor (B, L, S, K, K). Should be None if self.has_state=False.
            modify_state (bool): If True, updates the state tensor in-place with new activations.
            return_r (bool): If True, compute and return reward predictions.
            return_obs (bool): If True, compute and return observation predictions.

        Returns:
            tuple: (Q-values, reward predictions or None, observation predictions or None)
        """
        if self.has_state:
            assert state is not None, "State tensor required when has_state=True"
            assert state.shape[1] == self.L, \
                f"Expected state to have {self.L} layers, got {state.shape[1]}"

        x = self.input_block(encoded)

        for i in range(self.L):
            if self.has_state:
                x = self.trunk_blocks[i](x, state[:, i, ...])
                if modify_state:
                    state[:, i, ...] = x.detach()
            else:
                x = self.trunk_blocks[i](x, x)

        q_a = self.q_head(x)
        r_a = self.r_head(x) if return_r else None
        obs_a = self.obs_head(x) if return_obs else None
        return q_a, r_a, obs_a

    def forward(self,
                encoded: Encoded,
                state: BrainState | None = None) -> tuple[QA, RewardA, ObservationA]:
        """
        Standard PyTorch forward method.

        Used during training to compute Q-values and optional auxiliary predictions
        without modifying the brain state.

        Args:
            encoded (Encoded): Encoded environment input.
            state (BrainState | None): Brain state tensor.

        Returns:
            tuple: (Q-values, reward predictions, observation predictions)
        """
        q_a, r_a, obs_a = self._sdqn_step(encoded, state,
                                          modify_state=False,
                                          return_r=self.return_aux,
                                          return_obs=self.return_aux)
        return q_a, r_a, obs_a

    def predict_q(self,
                  encoded: Encoded,
                  state: BrainState | None = None) -> QA:
        """
        Compute Q-values without modifying the brain state and without auxiliary outputs.

        Used by target networks in Double DQN to estimate Q-values for the next state.

        Args:
            encoded (Encoded): Encoded environment input.
            state (BrainState | None): Brain state tensor.

        Returns:
            QA: Predicted Q-values.
        """
        q_a, _, _ = self._sdqn_step(encoded, state,
                                    modify_state=False,
                                    return_r=False,
                                    return_obs=False)
        return q_a

    def advance_with_aux(self,
                         encoded: Encoded,
                         state: BrainState | None = None) -> tuple[QA, RewardA, ObservationA]:
        """
        Advance the brain state and compute Q-values plus auxiliary predictions.

        Used during rollout for experience collection: updates the brain state
        and returns predictions for Q-values, rewards, and observations.

        Args:
            encoded (Encoded): Encoded environment input.
            state (BrainState | None): Brain state tensor to update in-place.

        Returns:
            tuple: (Q-values, reward predictions, observation predictions)
        """
        q_a, r_a, obs_a = self._sdqn_step(encoded, state,
                                          modify_state=True,
                                          return_r=self.return_aux,
                                          return_obs=self.return_aux)
        return q_a, r_a, obs_a

    def advance_q_only(self,
                       encoded: Encoded,
                       state: BrainState | None = None) -> QA:
        """
        Advance the brain state and compute Q-values only.

        Used during deployment when the environment supplies rewards and observations.

        Args:
            encoded (Encoded): Encoded environment input.
            state (BrainState | None): Brain state tensor to update in-place.

        Returns:
            QA: Predicted Q-values.
        """
        q_a, _, _ = self._sdqn_step(encoded, state,
                                    modify_state=True,
                                    return_r=False,
                                    return_obs=False)
        return q_a


if __name__ == "__main__":
    # --- Default config and model creation ---
    config = Config()

    model = SDQNModel(
        conf=config,
        has_state=True,
        return_aux=True,
        input_block_cls=InputBlock,
        inner_block_cls=InnerBlock,
        q_head_cls=DuelingQHead,
        r_head_cls=RewardHead,
        obs_head_cls=ObsHead
    )

    # --- Dummy inputs ---
    # Encoded input: (B, C_obs + C_action, K, K)
    encoded = torch.randn(
        config.batch_size,
        config.obs_channels + config.num_actions,
        config.obs_size,
        config.obs_size,
        device=config.model_device
    )

    # Brain state: (B, L, S, K, K)
    brain_state = torch.zeros(
        config.batch_size,
        config.brain_state_layers,
        config.brain_state_channels,
        config.obs_size,
        config.obs_size,
        device=config.brain_device
    )

    # --- 1. forward() ---
    q_a, r_a, obs_a = model(encoded, brain_state)
    print("\n[forward()] Shapes:")
    print(f"q_a   : {q_a.shape}  # Expected: (B, A)")
    print(f"r_a   : {r_a.shape}  # Expected: (B, A)")
    print(f"obs_a : {obs_a.shape}  # Expected: (B, A, C, K, K)")

    # --- 2. predict_q() ---
    q_pred = model.predict_q(encoded, brain_state)
    print("\n[predict_q()] Shapes:")
    print(f"q_pred: {q_pred.shape}  # Expected: (B, A)")

    # --- 3. advance_with_aux() ---
    state_copy = brain_state.clone()
    q_a2, r_a2, obs_a2 = model.advance_with_aux(encoded, state_copy)
    print("\n[advance_with_aux()] Shapes:")
    print(f"q_a   : {q_a2.shape}  # Expected: (B, A)")
    print(f"r_a   : {r_a2.shape}  # Expected: (B, A)")
    print(f"obs_a : {obs_a2.shape}  # Expected: (B, A, C, K, K)")
    print(f"state updated? {not torch.equal(brain_state, state_copy)}  # Should be True")

    # --- 4. advance_q_only() ---
    state_copy2 = brain_state.clone()
    q_a3 = model.advance_q_only(encoded, state_copy2)
    print("\n[advance_q_only()] Shapes:")
    print(f"q_a   : {q_a3.shape}  # Expected: (B, A)")
    print(f"state updated? {not torch.equal(brain_state, state_copy2)}  # Should be True")

    # --- Model summary ---
    print("\n[Model Summary - depth=10]")
    summary(model, input_data=(encoded, brain_state),
            col_names=["input_size", "output_size", "num_params"],
            depth=10)

    print("\n[Model Summary - depth=1]")
    summary(model, input_data=(encoded, brain_state),
            col_names=["input_size", "output_size", "num_params"],
            depth=1)
