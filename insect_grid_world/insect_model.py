import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict
from core.tensordict_helper import Schema, State
from core.q_model import MetazoanQModel
from custom_layers import ResTowerSeparable


class InsectQModel02(MetazoanQModel):

    def __init__(self, state_schema: Schema, num_actions: int,
                 main_channels=16, hid_channels=8, kernel_size=3, num_blocks=5
                 ):

        super().__init__(state_schema=state_schema, num_actions=num_actions)
        C, K1, K2 = self.state_schema["x"][0]

        # Pointwise convolution to mix input channels to main channels
        self.conv_inp = nn.Conv2d(in_channels=C, out_channels=main_channels,
                                  kernel_size=1, bias=False)
        self.bn_inp = nn.BatchNorm2d(num_features=main_channels)
        self.relu_inp = nn.ReLU()
        # Residual tower with separable convolutions as the trunk of the model
        self.trunk = ResTowerSeparable(main_channels=main_channels, hid_channels=hid_channels,
                                       kernel_size=kernel_size, num_blocks=num_blocks)
        # DDQN dueling head
        self.duel = DuelingHeadConv(main_channels=main_channels,
                                    flat_features=K1*K2, num_actions=num_actions)

    def forward(self, state: State) -> Tensor:

        # x = state[0]            # (B, C, K, K)
        x = state['x']            # (B, C, K, K)
        x = self.conv_inp(x)      # (B, M, K, K)
        x = self.bn_inp(x)        # (B, M, K, K)
        x = self.relu_inp(x)      # (B, M, K, K)

        x = self.trunk(x)         # (B, M, K, K)

        q_values = self.duel(x)   # (B, A)
        return q_values


class DuelingHeadConv(nn.Module):
    def __init__(self, main_channels, flat_features, num_actions):
        """
        """
        super().__init__()
        head_channels = 2

        # === Value stream: outputs a scalar V ===
        self.conv_val = nn.Conv2d(in_channels=main_channels, out_channels=head_channels,
                                  kernel_size=1, bias=True)
        # self.bn_val = nn.BatchNorm2d(num_features=1)
        self.relu_val = nn.ReLU()
        self.flat_val = nn.Flatten()
        self.lin_val = nn.Linear(in_features=flat_features*head_channels, out_features=1)

        # === Advantage stream: outputs a vector A ===
        self.conv_adv = nn.Conv2d(in_channels=main_channels, out_channels=head_channels,
                                  kernel_size=1, bias=True)
        # self.bn_adv = nn.BatchNorm2d(num_features=1)
        self.relu_adv = nn.ReLU()
        self.flat_adv = nn.Flatten()
        self.lin_adv = nn.Linear(in_features=flat_features*head_channels, out_features=num_actions)

    def forward(self, x):
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


if __name__ == "__main__":
    import torch
    from torchinfo import summary

    # Define dummy input shape
    B = 128               # Batch size
    C = 12                 # Input channels (e.g., 4-plane board state)
    K1 = K2 = 7           # Spatial size (e.g., 7x7 board)
    A = 5                 # Number of actions (output Q-values)
    main_ch = 16
    hid_ch = 8
    kern = 3
    num_bl = 5

    state_schema = {
        "x": (torch.Size([C, K1, K2]), torch.float32)
    }

    # state_schema = {
    #     "x": torch.Size([C, K1, K2])
    # }

    # Instantiate model
    model = InsectQModel02(
        state_schema=state_schema,
        num_actions=A,
        main_channels=main_ch,
        hid_channels=hid_ch,
        kernel_size=kern,
        num_blocks=num_bl
    )

    # Dummy input
    state = TensorDict(
        source={
            "x": torch.randn((B, *state_schema["x"][0]))
        },
        batch_size=[B]
    )

    # Run model once to ensure it works
    q = model(state)
    print(f"Q-values output shape: {q.shape}")  # Should be (B, A)

    # Convert to dict
    dummy_dict = {k: v for k, v in state.items()}

    # Print model summary
    print(summary(
        model,
        input_data=[dummy_dict],
        col_names=["input_size", "output_size", "num_params"],
        depth=10,
        verbose=1
    ))

    # Print model summary
    print(summary(
        model,
        input_data=[dummy_dict],
        col_names=["input_size", "output_size", "num_params"],
        depth=1,
        verbose=1
    ))


# === SOME MODEL IDEAS FOR LATER ===


# class InsectQModel01(MetazoanQModel):
#     def __init__(self, state_shape: tuple[int, ...], num_actions: int,
#                  hid_channels: int = 32, hid_features: int = 16):
#
#         super().__init__(state_shape=state_shape, num_actions=num_actions)
#         # self.state_shape = state_shape  # (B, C, K, K)
#
#         B, C, K, _ = self.state_shape
#         # hid_channels = 8
#         # hid_features = 16
#         self.trunk = nn.Sequential(
#             nn.Conv2d(in_channels=C, out_channels=hid_channels, kernel_size=3, padding=1),  # (B, hid, K, K)
#             nn.ReLU(),
#             nn.Conv2d(in_channels=hid_channels, out_channels=1, kernel_size=3, padding=0),  # (B, 1, K-2, K-2)
#             nn.Flatten()                                                                    # (B, (K-2) * (K-2))
#         )
#         self.duel = DuelingQHead(in_features=(K-2)*(K-2), hid_features=hid_features, num_actions=num_actions)
#
#     def forward(self, state: Tensor) -> Tensor:
#         # C, H, W = self._input_shape
#         # assert state.shape[1:] == (C, H, W), \
#         #     f"Expected input shape (B, {C}, {H}, {W}), got {state.shape}"
#         x = self.trunk(state)
#         q_values = self.duel(x)
#         return q_values
