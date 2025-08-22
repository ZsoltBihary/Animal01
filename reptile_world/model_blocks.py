import torch
from torch import Tensor
import torch.nn as nn
from reptile_world.custom_layers_sdqn import DepthwiseSeparableConv2D
from reptile_world.config import Config, Encoded, BrainSlice, BrainState, QA, RewardA, ObservationA
from torchinfo import summary


class InputBlock(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        self.conf = conf
        # === Consume configuration parameters ===
        E = conf.encoded_channels
        S = conf.brain_state_channels
        ker = 1

        self.conv = nn.Conv2d(in_channels=E, out_channels=S, kernel_size=ker, padding=ker // 2, bias=False)
        self.bn = nn.BatchNorm2d(num_features=S)
        self.relu = nn.ReLU()

    def forward(self, encoded: Encoded) -> BrainSlice:
        x = self.conv(encoded)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InnerBlock(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        # === Consume configuration parameters ===
        S = conf.brain_state_channels
        hid = S
        ker = 3

        self.conv1 = DepthwiseSeparableConv2D(in_channels=S, out_channels=hid, kernel_size=ker)
        self.bn1 = nn.BatchNorm2d(hid)
        self.relu1 = nn.ReLU()
        self.conv2 = DepthwiseSeparableConv2D(in_channels=hid, out_channels=S, kernel_size=ker)
        self.bn2 = nn.BatchNorm2d(S)

    def forward(self, x: BrainSlice) -> BrainSlice:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class DuelingQHead(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        # === Consume configuration parameters ===
        # self.B = conf.batch_size
        self.A = conf.num_actions
        self.S = conf.brain_state_channels
        K = conf.obs_size
        mult = conf.head_mult
        flat = mult * K * K

        # === Value stream: outputs a scalar V ===
        self.conv_val = nn.Conv2d(in_channels=self.S, out_channels=mult,
                                  kernel_size=1, bias=False)
        self.relu_val = nn.ReLU()
        self.flat_val = nn.Flatten()
        self.lin_val = nn.Linear(in_features=flat, out_features=1)

        # === Advantage stream: outputs a vector A ===
        self.conv_adv = nn.Conv2d(in_channels=self.S, out_channels=mult,
                                  kernel_size=1, bias=False)
        # self.bn_adv = nn.BatchNorm2d(num_features=1)
        self.relu_adv = nn.ReLU()
        self.flat_adv = nn.Flatten()
        self.lin_adv = nn.Linear(in_features=flat, out_features=self.A)

    def forward(self, x: BrainSlice) -> QA:
        # === Value stream: outputs a scalar V ===
        V = self.conv_val(x)                  # (B, mult, K, K)
        V = self.relu_val(V)                  # (B, mult, K, K)
        V = self.flat_val(V)                  # (B, mult*K*K)
        V = self.lin_val(V)                   # (B, 1)

        # === Advantage stream: outputs a vector A ===
        A = self.conv_adv(x)                  # (B, mult, K, K)
        A = self.relu_adv(A)                  # (B, mult, K, K)
        A = self.flat_adv(A)                  # (B, mult*K*K)
        A = self.lin_adv(A)                   # (B, A)

        # Combine streams: Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))
        A_mean = A.mean(dim=1, keepdim=True)  # (B, 1)
        Q = V + (A - A_mean)                  # (B, A)
        return Q


class RewardHead(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        # === Consume configuration parameters ===
        # self.B = conf.batch_size
        self.C = conf.obs_channels
        self.A = conf.num_actions
        self.S = conf.brain_state_channels
        K = conf.obs_size
        mult = conf.head_mult
        flat = mult * K * K

        self.conv = nn.Conv2d(in_channels=self.S, out_channels=mult,
                              kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=flat, out_features=self.A)

    def forward(self, x: BrainSlice) -> RewardA:
        x = self.conv(x)       # (B, mult, K, K)
        x = self.relu(x)
        x = self.flatten(x)    # (B, mult*K*K)
        r_a = self.linear(x)     # (B, A)
        return r_a


class ObsHead(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        # === Consume configuration parameters ===
        # self.B = conf.batch_size
        self.C = conf.obs_channels
        self.A = conf.num_actions
        self.S = conf.brain_state_channels
        self.K = conf.obs_size
        mult = conf.head_mult

        self.depthwise = nn.Conv2d(in_channels=self.S, out_channels=self.S * mult, groups=self.S,
                                   kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.pointwise = nn.Conv2d(in_channels=self.S * mult, out_channels=self.A * self.C,
                                   kernel_size=1, groups=self.A, bias=True)

    def forward(self, x: BrainSlice) -> ObservationA:
        x = self.depthwise(x)                      # (B, S*mult, K, K)
        x = self.relu(x)
        x = self.pointwise(x)                      # (B, A*C, K, K)
        B = x.shape[0]
        x = x.view(B, self.A, self.C, self.K, self.K)  # (B, A, C, K, K)
        return x


if __name__ == "__main__":
    conf = Config()

    B = conf.batch_size
    C = conf.obs_channels
    A = conf.num_actions
    S = conf.brain_state_channels
    K = conf.obs_size
    E = conf.encoded_channels

    # ==== Test InputBlock ====
    encoded = torch.randn(B, E, K, K)
    input_block = InputBlock(conf)
    brain_slice = input_block(encoded)
    print("InputBlock output:", brain_slice.shape)  # (B, S, K, K)

    # ==== Test InnerBlock ====
    inner_block = InnerBlock(conf)
    brain_slice2 = inner_block(brain_slice)
    print("InnerBlock output:", brain_slice2.shape)  # (B, S, K, K)

    # ==== Test DuelingQHead ====
    q_head = DuelingQHead(conf)
    q_values = q_head(brain_slice2)
    print("DuelingQHead output:", q_values.shape)  # (B, A)

    # ==== Test RewardHead ====
    reward_head = RewardHead(conf)
    reward_values = reward_head(brain_slice2)
    print("RewardHead output:", reward_values.shape)  # (B, A)

    # ==== Test ObsHead ====
    obs_head = ObsHead(conf)
    obs_pred = obs_head(brain_slice2)
    print("ObsHead output:", obs_pred.shape)  # (B, A, C, K, K)


# class ObsHeadGroupedConv(nn.Module):
#     def __init__(self, conf: Config):
#         super().__init__()
#         # === Consume configuration parameters ===
#         self.B = conf.batch_size
#         self.C = conf.obs_channels
#         self.A = conf.num_actions
#         self.S = conf.brain_state_channels
#         mult = conf.head_mult
#
#         assert self.S % self.A == 0, "brain channels must be divisible by num_actions for grouped conv."
#
#         # self.num_actions = num_actions
#         # self.num_classes = num_classes
#         # self.channels_per_action = main_channels // num_actions
#
#         self.depthwise = nn.Conv2d(in_channels=main_channels,
#                                    out_channels=main_channels * head_mult,
#                                    kernel_size=3, padding=1,
#                                    groups=main_channels, bias=False)
#
#         self.relu = nn.ReLU()
#
#         self.grouped_conv = nn.Conv2d(in_channels=main_channels * head_mult,
#                                       out_channels=num_actions * num_classes,
#                                       kernel_size=1,
#                                       groups=num_actions, bias=True)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: (B, S, K, K) - shared spatial feature map
#         Returns:
#             obs_a: (B, A, C, K, K) - action-conditioned output
#         """
#         x = self.depthwise(x)                   # (B, S, K, K)
#         x = self.relu(x)
#         x = self.grouped_conv(x)                # (B, A*C, K, K)
#
#         B = x.shape[0]
#         x = x.view(B, self.num_actions, self.num_classes, *x.shape[2:])  # (B, A, C, K, K)
#         return x
