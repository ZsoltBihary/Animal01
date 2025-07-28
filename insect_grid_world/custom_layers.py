import torch
import torch.nn as nn
from torchinfo import summary
from line_profiler_pycharm import profile


class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2  # Keeps output spatial size the same as input
        # Depthwise Convolution (groups=in_channels ensures separate filters per channel)
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, groups=in_channels,
                                   kernel_size=kernel_size, padding=padding, bias=False)
        # Pointwise Convolution (1x1 convolution for feature mixing)
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)  # Depthwise Convolution
        x = self.pointwise(x)  # Pointwise Convolution
        return x


class ResBlockSeparable(nn.Module):
    def __init__(self, main_channels, hid_channels, kernel_size):
        super().__init__()

        self.conv1 = DepthwiseSeparableConv2D(in_channels=main_channels,
                                              out_channels=hid_channels,
                                              kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(num_features=hid_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = DepthwiseSeparableConv2D(in_channels=hid_channels,
                                              out_channels=main_channels,
                                              kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(num_features=main_channels)
        self.relu2 = nn.ReLU()

    @profile
    def forward(self, x):
        skip = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += skip

        out = self.relu2(out)
        return out


class ResTowerSeparable(nn.Module):
    def __init__(self, main_channels, hid_channels, kernel_size, num_blocks):
        super().__init__()

        self.resi_tower = nn.ModuleList([
            ResBlockSeparable(main_channels=main_channels, hid_channels=hid_channels, kernel_size=kernel_size)
            for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.resi_tower:
            x = block(x)
        return x


if __name__ == "__main__":
    B, K = 128, 7
    main_channels, hid_channels, kernel_size, num_blocks = 32, 32, 3, 3

    # Example input: batch_size=1, channels=16, height=32, width=32
    x = torch.randn(B, main_channels, K, K)

    # Create the model
    model = ResTowerSeparable(main_channels=main_channels, hid_channels=hid_channels,
                              kernel_size=kernel_size, num_blocks=num_blocks)

    # # Print summary
    # print(summary(model, input_data=x, col_names=["input_size", "output_size", "num_params"],
    #               depth=100, verbose=1))
    # Print summary
    print(summary(model, input_data=x, col_names=["input_size", "output_size", "num_params"],
                  depth=100, verbose=1))

    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
