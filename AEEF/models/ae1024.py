import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Conv1Block(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2)
        self.cn1 = ConvBlock(in_channels, out_channels)
        self.cn2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.cn1(x)
        x = self.cn2(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cn1 = ConvBlock(3, 8)
        self.cn2 = ConvBlock(8, 8)
        self.down1 = nn.Sequential(
            Down(8, 16),
            Conv1Block(16, 8))
        self.down2 = nn.Sequential(
            Down(8, 16),
            Conv1Block(16, 8))
        self.down3 = nn.Sequential(
            Down(8, 16),
            Conv1Block(16, 8))
        self.down4 = nn.Sequential(
            Down(8, 16),
            Conv1Block(16, 8))
        self.out = nn.Sequential(
            nn.Linear(32 * 32 * 8, 32 * 32),
            nn.BatchNorm1d(32 * 32),
            nn.ReLU(),
        )

    def forward(self, input):
        x = self.cn1(input)
        x = self.cn2(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.inco = nn.Sequential(
            nn.Linear(32 * 32, 32 * 32 * 8),
            nn.BatchNorm1d(32 * 32 * 8),
            nn.ReLU(),
        )
        self.up1 = nn.Sequential(
            Conv1Block(8, 16),
            Up(16, 8))
        self.up2 = nn.Sequential(
            Conv1Block(8, 16),
            Up(16, 8))
        self.up3 = nn.Sequential(
            Conv1Block(8, 16),
            Up(16, 8))
        self.up4 = nn.Sequential(
            Conv1Block(8, 16),
            Up(16, 8))
        self.out = nn.Sequential(
            ConvBlock(8, 8),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.inco(x)
        x = x.view(-1, 8, 32, 32)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        outs = self.out(x)

        return outs


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)