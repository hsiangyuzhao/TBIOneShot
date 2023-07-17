import torch.nn as nn
import torch.nn.functional as F
from utils.ddp_utils import *


norm_dict = {'BATCH': nn.BatchNorm3d, 'INSTANCE': nn.InstanceNorm3d, 'GROUP': nn.GroupNorm}
__all__ = ['ConvNorm', 'ConvBlock', 'ConvBottleNeck', 'ResBlock', 'ResBottleneck', 'SobelEdge']


class Identity(nn.Module):
    """
    Identity mapping for building a residual connection
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ConvNorm(nn.Module):
    """
    Convolution and normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=True, norm='BATCH', activation=True):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        padding = (kernel_size - 1) // 2

        # activation, support PReLU and common ReLU
        if activation:
            self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
            # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)
        else:
            self.act = None

        # instantiate layers
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(4, in_channels)

    def basic_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        if self.act:
            x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)


class ConvBlock(nn.Module):
    """
    Convolutional blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        # activation, support PReLU and common ReLU
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True)
        self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ResBlock(nn.Module):
    """
    Residual blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1) if use_dropout else None
        # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True)
        self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

        need_map = in_channels != out_channels or stride != 1
        self.id = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False) if need_map else Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        identity = self.id(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        return out


class ConvBottleNeck(nn.Module):
    """
    Convolutional bottleneck blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        middle_channels = in_channels // 4
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, middle_channels, 1, 1, leaky, norm, True)
        self.conv2 = ConvNorm(middle_channels, middle_channels, 3, stride, leaky, norm, True)
        self.conv3 = ConvNorm(middle_channels, out_channels, 1, 1, leaky, norm, False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out


class ResBottleneck(nn.Module):
    """
    Residual bottleneck blocks
    """
    def __init__(self, in_channels, out_channels, stride=1, use_dropout=False, leaky=False, norm='BATCH'):
        super().__init__()
        self.norm_type = norm
        middle_channels = in_channels // 4
        self.act = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1) if use_dropout else None
        # self.act = nn.ELU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = ConvNorm(in_channels, middle_channels, 1, 1, leaky, norm, True)
        self.conv2 = ConvNorm(middle_channels, middle_channels, 3, stride, leaky, norm, True)
        self.conv3 = ConvNorm(middle_channels, out_channels, 1, 1, leaky, norm, False)

        need_map = in_channels != out_channels or stride != 1
        self.id = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False) if need_map else Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        identity = self.id(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        if self.dropout:
            out = self.dropout(out)

        return out


class SobelEdge(nn.Module):
    def __init__(self, input_dim, channels, kernel_size=3, stride=1):
        super().__init__()
        conv = getattr(nn, 'Conv%dd' % input_dim)
        self.filter = conv(channels, channels, kernel_size, stride, padding=(kernel_size - 1) // 2,
                           groups=channels, bias=False)
        sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        sobel_kernel = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand([channels, 1] + [kernel_size] * input_dim)
        self.filter.weight = nn.Parameter(sobel_kernel, requires_grad=False)

    def forward(self, x):
        with torch.no_grad():
            out = self.filter(x)
        return out
