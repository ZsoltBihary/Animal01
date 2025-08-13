import torch
from torch import Tensor
import torch.nn as nn
from reptile_world.custom_layers_sdqn import DepthwiseSeparableConv2D
from reptile_world.config import Config, Encoded, BrainState, QA, RewardA, ObservationA
from torchinfo import summary

#
# from typing import Optional, Tuple
# import torch
# import torch.nn as nn
#
# TensorOrNone = Optional[torch.Tensor]

# class SDQNModel(nn.Module):
#     def __init__(self, conf):
#         super().__init__()
#         # ... your init as before ...
#         # conv_inp, bn_inp, relu_inp, trunk, q_head, r_head, obs_head ...
#         # keep ReLU inplace=False inside your custom layers if possible
#
#     def sdqn(self,
#              encoded: torch.Tensor,
#              state: torch.Tensor,
#              compute_q: bool = True,
#              compute_r: bool = True,
#              compute_obs: bool = True,
#              return_state: bool = True
#     ) -> Tuple[TensorOrNone, TensorOrNone, TensorOrNone, Optional[torch.Tensor]]:
#         """
#         ALWAYS returns a 4-tuple: (q_a or None, r_a or None, obs_a or None, new_state or None)
#         Heads are only computed when the corresponding compute_* flag is True.
#         """
#         x = self.conv_inp(encoded)
#         x = self.bn_inp(x)
#         x = self.relu_inp(x)  # ensure not inplace
#
#         x, new_state = self.trunk(x, state)
#
#         q_a = self.q_head(x) if compute_q else None
#         r_a = self.r_head(x) if compute_r else None
#         obs_a = self.obs_head(x) if compute_obs else None
#
#         return q_a, r_a, obs_a, (new_state if return_state else None)
#
#     # Thin wrappers for the common modes
#     def forward(self, encoded: torch.Tensor, state: torch.Tensor):
#         # training mode: compute all heads (no returned state required)
#         q_a, r_a, obs_a, _ = self.sdqn(encoded, state,
#                                        compute_q=True, compute_r=True, compute_obs=True,
#                                        return_state=True)  # we receive state but ignore
#         return q_a, r_a, obs_a
#
#     def rollout(self, encoded: torch.Tensor, state: torch.Tensor):
#         # online rollout: get everything + new state
#         return self.sdqn(encoded, state,
#                          compute_q=True, compute_r=True, compute_obs=True,
#                          return_state=True)
#
    # def target_q(self, encoded: torch.Tensor, state: torch.Tensor):
    #     # target model only needs Q values; no grads, no state change
    #     q_a, _, _, _ = self.sdqn(encoded, state,
    #                              compute_q=True, compute_r=False, compute_obs=False,
    #                              return_state=False)
    #     return q_a
#
#     def simulate(self, encoded: torch.Tensor, state: torch.Tensor):
#         # simulation: need q and new_state, no other heads
#         q_a, _, _, new_state = self.sdqn(encoded, state,
#                                          compute_q=True, compute_r=False, compute_obs=False,
#                                          return_state=True)
#         return q_a, new_state
#
#     def dream(self, encoded: torch.Tensor, state: torch.Tensor):
#         # full model generation with state update
#         return self.sdqn(encoded, state,
#                          compute_q=True, compute_r=True, compute_obs=True,
#                          return_state=True)


class SDQNModel(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        self.conf = conf
        # === Consume configuration parameters ===
        self.device = conf.model_device
        # self.B = conf.batch_size
        self.C = conf.obs_channels
        self.K = conf.obs_size
        self.A = conf.num_actions
        self.L = conf.brain_state_layers
        self.S = conf.brain_state_channels
        self.head_mult = conf.head_mult

        # Convolution layer to transform encoded observation+last_action input in line with brain_state channels
        self.conv_inp = nn.Conv2d(in_channels=self.C+self.A, out_channels=self.S,
                                  kernel_size=1, padding=0, bias=False)
        self.bn_inp = nn.BatchNorm2d(num_features=self.S)
        self.relu_inp = nn.ReLU()
        # Trunk is a "recurrent stateful residual CNN tower"
        self.trunk = BrainTowerSeparable(num_layers=self.L,
                                         main_channels=self.S, hid_channels=self.S // 2,
                                         kernel_size=3)
        # Q head is a Dueling network with pointwise conv + flattening + linear
        self.q_head = DuelingHeadConv(main_channels=self.S, head_channels=self.head_mult,
                                      flat_features=self.head_mult * self.K * self.K,
                                      num_actions=self.A)
        # Reward head is similar, but w/o the Dueling complication
        self.r_head = RewardHeadConv(main_channels=self.S, head_channels=self.head_mult,
                                     flat_features=self.head_mult * self.K * self.K,
                                     num_actions=self.A)
        # Observation head is a depthwise conv + pointwise conv
        # self.obs_head = ObsHeadConv(main_channels=self.S,
        #                             num_actions=self.A,
        #                             num_classes=self.C)
        # OR ... Observation head is a depthwise conv + grouped conv
        self.obs_head = ObsHeadGroupedConv(main_channels=self.S, head_mult=self.head_mult,
                                           num_actions=self.A,
                                           num_classes=self.C)

        self.to(self.device)  # This must be at the end of __init__()

    def sdqn(self, encoded: Encoded, state: BrainState,
             return_q=True, return_r=True, return_obs=True, return_state=True):
        """
        Core computation graph.
        All other modes should call this with the appropriate flags.
        """
        x = self.conv_inp(encoded)
        x = self.bn_inp(x)
        x = self.relu_inp(x)

        x, new_state = self.trunk(x, state)

        q_a = self.q_head(x) if return_q else None
        r_a = self.r_head(x) if return_r else None
        obs_a = self.obs_head(x) if return_obs else None

        if return_state:
            return q_a, r_a, obs_a, new_state
        else:
            return q_a, r_a, obs_a

    def forward(self, encoded: Encoded, state: BrainState):
        # Training mode: full predictions, no new_state
        q_a, r_a, obs_a, _ = self.sdqn(encoded, state, return_state=True)
        return q_a, r_a, obs_a

    def rollout(self, encoded: Encoded, state: BrainState):
        # Online rollout: full predictions + state
        return self.sdqn(encoded, state)

    def target_q(self, encoded: Encoded, state: BrainState):
        # Target model for double DQN: only q_a
        q_a, _, _, _ = self.sdqn(encoded, state,
                                 return_r=False, return_obs=False)
        return q_a

    def simulate(self, encoded: Encoded, state: BrainState):
        # Simulation: q_a + state only
        q_a, _, _, new_state = self.sdqn(encoded, state,
                                         return_r=False, return_obs=False)
        return q_a, new_state

    def dream(self, encoded: Encoded, state: BrainState):
        # Dreaming/planning: full predictions + state
        return self.sdqn(encoded, state)

    # def sdqn(self, encoded: Encoded, state: BrainState
    #          ) -> tuple[QA, RewardA, ObservationA, BrainState]:
    #
    #     x = self.conv_inp(encoded)
    #     x = self.bn_inp(x)
    #     x = self.relu_inp(x)
    #
    #     x, new_state = self.trunk(x, state)
    #
    #     q_a = self.q_head(x)
    #     r_a = self.r_head(x)
    #     obs_a = self.obs_head(x)
    #
    #     return q_a, r_a, obs_a, new_state

    # def forward(self, encoded: Encoded, state: BrainState
    #             ) -> tuple[QA, RewardA, ObservationA]:
    #
    #     q_a, r_a, obs_a, _ = self.sdqn(encoded, state)
    #     return q_a, r_a, obs_a


class ObsHeadGroupedConv(nn.Module):
    def __init__(self, main_channels: int, head_mult: int, num_actions: int, num_classes: int):
        super().__init__()
        assert main_channels % num_actions == 0, \
            "main_channels must be divisible by num_actions for grouped conv."

        self.num_actions = num_actions
        self.num_classes = num_classes
        self.channels_per_action = main_channels // num_actions

        self.depthwise = nn.Conv2d(in_channels=main_channels,
                                   out_channels=main_channels * head_mult,
                                   kernel_size=3, padding=1,
                                   groups=main_channels, bias=False)

        self.relu = nn.ReLU()

        self.grouped_conv = nn.Conv2d(in_channels=main_channels * head_mult,
                                      out_channels=num_actions * num_classes,
                                      kernel_size=1,
                                      groups=num_actions, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, S, K, K) - shared spatial feature map
        Returns:
            obs_a: (B, A, C, K, K) - action-conditioned output
        """
        x = self.depthwise(x)                   # (B, S, K, K)
        x = self.relu(x)
        x = self.grouped_conv(x)                # (B, A*C, K, K)

        B = x.shape[0]
        x = x.view(B, self.num_actions, self.num_classes, *x.shape[2:])  # (B, A, C, K, K)
        return x


class ObsHeadConv(nn.Module):
    def __init__(self, main_channels: int, num_actions: int, num_classes: int):
        super().__init__()
        self.num_actions = num_actions
        self.num_classes = num_classes

        self.depthwise = nn.Conv2d(in_channels=main_channels,
                                   out_channels=main_channels,
                                   kernel_size=3, padding=1,
                                   groups=main_channels, bias=False)

        self.relu = nn.ReLU()

        self.pointwise = nn.Conv2d(in_channels=main_channels,
                                   out_channels=num_actions * num_classes,
                                   kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, S, K, K) - shared spatial feature map
        Returns:
            obs_a: (B, A, C, K, K) - action-conditioned logits or probabilities
        """
        x = self.depthwise(x)                      # (B, S, K, K)
        x = self.relu(x)
        x = self.pointwise(x)                      # (B, A*C, K, K)
        B = x.shape[0]
        x = x.view(B, self.num_actions, self.num_classes, *x.shape[2:])  # (B, A, C, K, K)
        return x


class DuelingHeadConv(nn.Module):
    def __init__(self, main_channels, head_channels, flat_features, num_actions):
        super().__init__()

        # === Value stream: outputs a scalar V ===
        self.conv_val = nn.Conv2d(in_channels=main_channels, out_channels=head_channels,
                                  kernel_size=1, bias=True)
        # self.bn_val = nn.BatchNorm2d(num_features=1)
        self.relu_val = nn.ReLU()
        self.flat_val = nn.Flatten()
        self.lin_val = nn.Linear(in_features=flat_features, out_features=1)

        # === Advantage stream: outputs a vector A ===
        self.conv_adv = nn.Conv2d(in_channels=main_channels, out_channels=head_channels,
                                  kernel_size=1, bias=True)
        # self.bn_adv = nn.BatchNorm2d(num_features=1)
        self.relu_adv = nn.ReLU()
        self.flat_adv = nn.Flatten()
        self.lin_adv = nn.Linear(in_features=flat_features, out_features=num_actions)

    def forward(self, x) -> QA:
        """
        Args:
            x: Tensor of shape (B, M, K, K), the output from the residual tower trunk.
        Returns:
            q_values: Tensor of shape (B, A)
        """

        # === Value stream: outputs a scalar V ===
        V = self.conv_val(x)                  # (B, 1, K, K)
        # V = self.bn_val(V)                    # (B, 1, K, K)
        V = self.relu_val(V)                  # (B, 1, K, K)
        V = self.flat_val(V)                  # (B, K*K)
        V = self.lin_val(V)                   # (B, 1)

        # === Advantage stream: outputs a vector A ===
        A = self.conv_adv(x)                  # (B, 1, K, K)
        # A = self.bn_adv(A)                    # (B, 1, K, K)
        A = self.relu_adv(A)                  # (B, 1, K, K)
        A = self.flat_adv(A)                  # (B, K*K)
        A = self.lin_adv(A)                   # (B, A)

        # Combine streams: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        A_mean = A.mean(dim=1, keepdim=True)  # (B, 1)
        Q = V + (A - A_mean)                  # (B, A)
        return Q


class RewardHeadConv(nn.Module):
    def __init__(self, main_channels: int, head_channels: int, flat_features: int, num_actions: int):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=main_channels, out_channels=head_channels,
                              kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=flat_features, out_features=num_actions)

    def forward(self, x: Tensor) -> RewardA:
        """
        Args:
            x: Tensor of shape (B, S, K, K)

        Returns:
            r_a: Tensor of shape (B, A) - predicted reward per action
        """
        x = self.conv(x)       # (B, head_mult, K, K)
        x = self.relu(x)
        x = self.flatten(x)    # (B, flat_features)
        r_a = self.linear(x)     # (B, A)
        return r_a


class BrainTowerSeparable(nn.Module):
    """
    Stacked recurrent residual blocks for brain state update.

    Each block receives previous activation `x_prev` and a static brain state slice `s_i`.

    Args:
        num_layers (int): Number of brain layers (L)
        main_channels (int): Channel dimension of each state (S)
        hid_channels (int): Hidden channels used inside each block
        kernel_size (int): Kernel size for the depthwise separable convs
    """

    def __init__(self, num_layers: int, main_channels: int, hid_channels: int, kernel_size: int):
        super().__init__()
        self.L = num_layers
        self.blocks = nn.ModuleList([
            BrainBlockSeparable(
                main_channels=main_channels,
                hid_channels=hid_channels,
                kernel_size=kernel_size
            )
            for _ in range(self.L)
        ])

    def forward(self, x0: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x0: Initial activation tensor (B, S, K, K)
            state: Brain state tensor (B, L, S, K, K)

        Returns:
            x_last: Final activation tensor after recurrence (B, S, K, K)
            updated_state: Full updated brain state (B, L, S, K, K)
        """
        x_prev = x0
        x_list = []

        for i in range(self.L):
            s_i = state[:, i, ...]        # Static input slice (B, S, K, K)
            x_i = self.blocks[i](x_prev, s_i)
            x_list.append(x_i)
            x_prev = x_i

        updated_state = torch.stack(x_list, dim=1)  # (B, L, S, K, K)
        x_last = x_prev
        return x_last, updated_state


class BrainBlockSeparable(nn.Module):
    """
    A recurrent residual block used in SDQN, combining the prior activation `x_prev`
    with a static state slice `s_i`.

    Args:
        main_channels (int): Number of channels for both input/output activation tensors.
        hid_channels (int): Intermediate hidden channel size.
        kernel_size (int): Kernel size for depthwise separable convolutions.
    """

    def __init__(self, main_channels: int, hid_channels: int, kernel_size: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2D(
            in_channels=main_channels,
            out_channels=hid_channels,
            kernel_size=kernel_size
        )
        self.bn1 = nn.BatchNorm2d(hid_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = DepthwiseSeparableConv2D(
            in_channels=hid_channels,
            out_channels=main_channels,
            kernel_size=kernel_size
        )
        self.bn2 = nn.BatchNorm2d(main_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x_prev: Tensor, s_i: Tensor) -> Tensor:
        """
        Forward pass for the block.

        Args:
            x_prev (Tensor): Previous activation (B, main_channels, K, K)
            s_i (Tensor): Static brain state slice (B, main_channels, K, K)

        Returns:
            x_i (Tensor): Updated activation (B, main_channels, K, K)
        """
        out = self.conv1(s_i)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + x_prev  # Skip connection
        out = self.relu2(out)

        return out


if __name__ == "__main__":

    config = Config()

    # Instantiate model
    model = SDQNModel(config)

    # Create dummy encoded observation+action (B, C + A, K, K)
    encoded = torch.randn(config.batch_size, config.obs_channels + config.num_actions,
                          config.obs_size, config.obs_size, device=config.model_device)

    # Create dummy brain state (L, B, S, K, K)
    brain_state = torch.zeros(
                              config.batch_size,
                              config.brain_state_layers,
                              config.brain_state_channels,
                              config.obs_size,
                              config.obs_size,
                              device=config.brain_device)

    # Forward pass (for training)
    q_a, r_a, obs_a = model(encoded, brain_state)

    print(f"\n[Forward] Shapes:")
    print(f"q_a     : {q_a.shape}  # Expected: (B, A)")
    print(f"r_a     : {r_a.shape}  # Expected: (B, A)")
    print(f"obs_a   : {obs_a.shape}  # Expected: (B, A, C, K, K)")

    # Full SDQN pass (returns updated brain state too)
    q_a2, r_a2, obs_a2, new_state = model.sdqn(encoded, brain_state)

    print(f"\n[SDQN] Shapes:")
    print(f"q_a     : {q_a2.shape}")
    print(f"r_a     : {r_a2.shape}")
    print(f"obs_a   : {obs_a2.shape}")
    print(f"new_state : {new_state.shape}  # Expected: (B, L, S, K, K)")

    # Model summary
    print("\n[Model Summary]")
    summary(model, input_data=(encoded, brain_state), col_names=["input_size", "output_size", "num_params"], depth=10)
    print("\n[Model Summary]")
    summary(model, input_data=(encoded, brain_state), col_names=["input_size", "output_size", "num_params"], depth=1)
