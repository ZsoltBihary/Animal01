import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, A, C=None, kernel_size=3, learnable=False,
                 use_mask_prior=False, prior=None, trainable_prior=True):
        """
        A: number of actions
        C: number of channels (needed if use_mask_prior=True)
        kernel_size: convolution kernel size (int or tuple)
        learnable: if True, kernels are trainable nn.Parameter
        use_mask_prior: if True, add mask channel and blend with prior
        prior: optional tensor (C,) with prior distribution
        trainable_prior: if True, prior is nn.Parameter (trainable)
        """
        super().__init__()
        self.A = A
        self.C = C
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.use_mask_prior = use_mask_prior

        # Kernels
        kH, kW = self.kernel_size
        if learnable:
            self.kernels = nn.Parameter(
                torch.empty(A, kH, kW).normal_(0, 0.1)
            )
        else:
            self.register_buffer("kernels", torch.empty(A, kH, kW))

        # Prior (only relevant if mask_prior enabled)
        if self.use_mask_prior:
            if prior is None:
                prior = torch.ones(C) / C  # uniform prior
            if trainable_prior:
                self.prior = nn.Parameter(prior.float())
            else:
                self.register_buffer("prior", prior.float())

    @classmethod
    def translation5(cls, C, learnable=False, use_mask_prior=False,
                     prior=None, trainable_prior=True):
        """Utility constructor: 5 actions (stay, up, down, left, right)."""
        actions = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]
        kernels = torch.zeros(len(actions), 3, 3)
        center = 1
        for i, a in enumerate(actions):
            if a == "STAY":
                kernels[i, center, center] = 1
            elif a == "UP":
                kernels[i, center-1, center] = 1
            elif a == "DOWN":
                kernels[i, center+1, center] = 1
            elif a == "LEFT":
                kernels[i, center, center-1] = 1
            elif a == "RIGHT":
                kernels[i, center, center+1] = 1
        obj = cls(len(actions), C=C, learnable=learnable,
                  use_mask_prior=use_mask_prior,
                  prior=prior, trainable_prior=trainable_prior)
        if learnable:
            obj.kernels.data.copy_(kernels)
        else:
            obj.kernels.copy_(kernels)
        return obj

    def cond_transf(self, x):
        """
        x: (B, C, H, W)
        return: (B, A, C, H, W)
        """
        B, C, H, W = x.shape
        ky, kx = self.kernel_size

        if self.use_mask_prior:
            # Step 1: build mask channel
            mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
            x_aug = torch.cat([x, mask], dim=1)  # (B, C+1, H, W)
            C_aug = C + 1
        else:
            x_aug = x
            C_aug = C

        # Kernels expanded to all channels
        kernels = self.kernels.unsqueeze(1).expand(self.A, C_aug, ky, kx)
        kernels = kernels.reshape(self.A * C_aug, 1, ky, kx)

        # Repeat channels into A groups
        x_rep = x_aug.repeat_interleave(self.A, dim=1)  # (B, A*C_aug, H, W)

        # Grouped conv
        y_aug = F.conv2d(x_rep, kernels, padding=(ky // 2, kx // 2), groups=self.A * C_aug)

        if self.use_mask_prior:
            # Split into features and mask
            y, mask_out = y_aug[:, :-self.A, :, :], y_aug[:, -self.A:, :, :]
            y = y.reshape(B, self.A, C, H, W)
            mask_out = mask_out.reshape(B, self.A, 1, H, W)

            # Blend with prior
            prior = self.prior.view(1, 1, C, 1, 1)  # broadcastable
            y = mask_out * y + (1 - mask_out) * prior
            return y
        else:
            return y_aug.reshape(B, self.A, C, H, W)

    def transf(self, x, a):
        """Apply selected action transformation"""
        x_a = self.cond_transf(x)  # (B, A, C, H, W)
        return self.select(x_a, a)

    def select(self, x_a, a):
        """Select features according to action indices"""
        B = x_a.shape[0]
        return x_a[torch.arange(B), a]

# **************************************************************************
# **************************************************************************
# **************************************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F


FeatureMap   = torch.Tensor   # (B, C, H, W)
FeatureMapA  = torch.Tensor   # (B, A, C, H, W)
Action       = torch.Tensor   # (B,) long
KernelPool   = torch.Tensor   # (A, ky, kx)


class SpatialTransformer(nn.Module):
    """
    Apply action-dependent spatial transformations (e.g. translations).
    """

    def __init__(self, A: int, ker_pool: KernelPool = None,
                 learnable: bool = False, padding: int = 1):
        """
        Core constructor.
        If ker_pool is None → initialize random kernels.
        If ker_pool is given → use provided kernels.
        """
        super().__init__()
        self.A = A
        self.padding = padding

        if ker_pool is None:
            # default: random init
            ker_pool = torch.randn(A, 3, 3)

        self.ky, self.kx = ker_pool.shape[1:]

        if learnable:
            self.ker_pool = nn.Parameter(ker_pool.clone())
        else:
            self.register_buffer("ker_pool", ker_pool.clone())

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------
    def cond_transf(self, x: FeatureMap) -> FeatureMapA:
        B, C, H, W = x.shape
        A, ky, kx = self.ker_pool.shape

        x_rep = x.unsqueeze(1).expand(B, A, C, H, W).reshape(B, A*C, H, W)
        kernels = self.ker_pool[:, None].expand(A, C, ky, kx).reshape(A*C, 1, ky, kx)

        y = F.conv2d(x_rep, kernels, padding=(ky//2, kx//2), groups=A*C)  # (B, A*C, H, W)
        return y.view(B, A, C, H, W)

    def select(self, x_a: FeatureMapA, a: Action) -> FeatureMap:
        B, A, C, H, W = x_a.shape
        idx = a.view(B, 1, 1, 1).expand(B, C, H, W).unsqueeze(1)  # (B,1,C,H,W)
        return x_a.gather(1, idx).squeeze(1)

    def transf(self, x: FeatureMap, a: Action) -> FeatureMap:
        return self.select(self.cond_transf(x), a)

    def forward(self, x: FeatureMap, a: Action = None):
        if a is None:
            return self.cond_transf(x)   # (B,A,C,H,W)
        else:
            return self.transf(x, a)     # (B,C,H,W)

    # -------------------------------------------------------------------------
    # Convenience constructors
    # -------------------------------------------------------------------------
    @classmethod
    def translation5(cls, learnable=False, padding=1):
        """
        Create transformer with 3x3 shift kernels for cardinal moves.
        """
        actions = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]
        ker_pool = cls._make_shift_kernels(actions)
        return cls(len(actions), ker_pool, learnable=learnable, padding=padding)

    @classmethod
    def translation9(cls, learnable=False, padding=1):
        """
        Create transformer with 3x3 shift kernels for all 8 directions + stay.
        (A=9)
        """
        actions = [
            "STAY",
            "UP", "DOWN", "LEFT", "RIGHT",
            "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT"
        ]
        ker_pool = cls._make_shift_kernels(actions)
        return cls(len(actions), ker_pool, learnable=learnable, padding=padding)

    # -------------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------------
    @staticmethod
    def _make_shift_kernels(actions):
        """
        Build 3x3 kernels that implement 1-pixel translations.
        """
        offsets = {
            "STAY":       (0,  0),
            "UP":         (-1, 0),
            "DOWN":       (1,  0),
            "LEFT":       (0, -1),
            "RIGHT":      (0,  1),
            "UP_LEFT":    (-1, -1),
            "UP_RIGHT":   (-1,  1),
            "DOWN_LEFT":  (1, -1),
            "DOWN_RIGHT": (1,  1),
        }
        K = torch.zeros(len(actions), 3, 3)
        for i, act in enumerate(actions):
            dy, dx = offsets[act]
            K[i, 1+dy, 1+dx] = 1.0
        return K


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def make_shift_kernels(actions=("STAY","UP","DOWN","LEFT","RIGHT")):
#     """
#     Build 3x3 kernels that implement 1-pixel translations with padding=1.
#     Convolution y(i,j) = sum_{u,v} K[u,v] * x(i+u-1, j+v-1)
#     To get y(i,j) = x(i+dy, j+dx), set K[1+dy, 1+dx] = 1.
#     """
#     A = len(actions)
#     K = torch.zeros(A, 1, 3, 3)  # (A, 1, 3, 3)
#
#     # mapping from action -> (dy, dx)
#     offsets = {
#         "STAY":  (0,  0),
#         "UP":    (-1, 0),  # content moves up
#         "DOWN":  (1,  0),
#         "LEFT":  (0, -1),
#         "RIGHT": (0,  1),
#     }
#     for i, act in enumerate(actions):
#         dy, dx = offsets[act]
#         K[i, 0, 1 + dy, 1 + dx] = 1.0
#     return K
#
#
# class AllActionConv2d(nn.Module):
#     """
#     Compute action-conditional 3x3 transforms for ALL actions at once.
#
#     Input:
#       x: (B, C, H, W)
#     Output:
#       y_all: (B, A, C, H, W)  # per-action transformed feature maps
#
#     How it works:
#       - Replicate channels across actions => (B, A*C, H, W)
#       - Use stacked per-action depthwise kernels => (A*C, 1, 3, 3)
#       - One grouped conv with groups=A*C
#     """
#     def __init__(self, actions=("STAY","UP","DOWN","LEFT","RIGHT"),
#                  padding=1, learnable=False):
#         super().__init__()
#         self.actions = tuple(actions)
#         self.A = len(self.actions)
#         self.padding = padding
#
#         base = make_shift_kernels(self.actions)  # (A,1,3,3)
#
#         if learnable:
#             # Learn per-action kernels; initialize to shifts
#             self.weight = nn.Parameter(base)  # (A,1,3,3)
#         else:
#             # Fixed (non-trainable) shift kernels
#             self.register_buffer("weight", base)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         A = self.A
#
#         # (B, C, H, W) -> (B, A, C, H, W) -> (B, A*C, H, W)
#         # expand is cheap (view with strides); reshape may materialize contiguously as needed by conv
#         x_ac = x.unsqueeze(1).expand(B, A, C, H, W).reshape(B, A * C, H, W)
#
#         # Build stacked depthwise kernels: (A*C, 1, 3, 3)
#         # Each action kernel is shared across channels
#         k = self.weight.expand(A, C, 3, 3).reshape(A * C, 1, 3, 3)
#
#         # One grouped conv over A*C groups
#         y = F.conv2d(x_ac, k, padding=self.padding, groups=A * C)  # (B, A*C, H, W)
#
#         # -> (B, A, C, H, W)
#         y = y.view(B, A, C, H, W)
#         return y
#
# #
# # B, C, H, W = 1000, 64, 9, 9
# # x = torch.randn(B, C, H, W, device="cuda")  # local maps
# #
# # layer = AllActionConv2d(
# #     actions=("STAY","UP","DOWN","LEFT","RIGHT"),
# #     padding=1,
# #     learnable=False
# # ).cuda()
# #
# # y_all = layer(x)             # (B, 5, C, H, W)
# # y_up   = y_all[:, 1]         # (B, C, H, W), for "UP"
# #
# # import torch
# # import torch.nn.functional as F
# #
# # # type aliases (for readability only, no enforcement)
# # FeatureMap   = torch.Tensor   # (B, C, H, W)
# # FeatureMapA  = torch.Tensor   # (B, A, C, H, W)
# # Action       = torch.Tensor   # (B,) long
# # KernelPool   = torch.Tensor   # (A, ky, kx)
# #
# #
# # def cond_transf(x: FeatureMap, ker_pool: KernelPool) -> FeatureMapA:
# #     """
# #     Apply all action-conditional kernels to feature map x.
# #     Args:
# #         x: (B, C, H, W)
# #         ker_pool: (A, ky, kx)  typically 3x3
# #     Returns:
# #         y: (B, A, C, H, W)
# #     """
# #     B, C, H, W = x.shape
# #     A, ky, kx = ker_pool.shape
# #
# #     # replicate x along action axis → (B, A*C, H, W)
# #     x_rep = x.unsqueeze(1).expand(B, A, C, H, W).reshape(B, A*C, H, W)
# #
# #     # make depthwise kernels, shared across channels
# #     kernels = ker_pool[:, None].expand(A, C, ky, kx).reshape(A*C, 1, ky, kx)
# #
# #     # grouped conv
# #     y = F.conv2d(x_rep, kernels, padding=(ky//2, kx//2), groups=A*C)  # (B, A*C, H, W)
# #
# #     return y.view(B, A, C, H, W)
# #
# #
# # def select(x_a: FeatureMapA, a: Action) -> FeatureMap:
# #     """
# #     Select per-sample action-conditioned map.
# #     Args:
# #         x_a: (B, A, C, H, W)
# #         a:   (B,)
# #     Returns:
# #         y: (B, C, H, W)
# #     """
# #     B, A, C, H, W = x_a.shape
# #     # expand action indices to broadcast over (C,H,W)
# #     idx = a.view(B, 1, 1, 1).expand(B, C, H, W).unsqueeze(1)  # (B,1,C,H,W)
# #     return x_a.gather(1, idx).squeeze(1)  # (B,C,H,W)
# #
# #
# # def transf(x: FeatureMap, a: Action, ker_pool: KernelPool) -> FeatureMap:
# #     """
# #     Directly apply selected actions, without materializing all A.
# #     Args:
# #         x: (B, C, H, W)
# #         a: (B,)
# #         ker_pool: (A, ky, kx)
# #     Returns:
# #         y: (B, C, H, W)
# #     """
# #     # route through cond_transf + select
# #     return select(cond_transf(x, ker_pool), a)
