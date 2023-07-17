import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_modules import *
from models.registration_modules import SpatialSampleLayer, IntensitySampleLayer


class Backbone(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=3, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
                 use_dropout=False, **kwargs):
        """
        Args:
            input_channels: the number of input channels
            channels: length-5 tuple, define the number of channels in each stage
            strides: tuple, define the stride in each stage
            use_dropout: bool, whether to use dropout
            **kwargs: other args define activation and normalization
        """
        super().__init__()
        self.nb_filter = channels
        self.strides = strides + (5 - len(strides)) * (1,)
        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, self.nb_filter[0], kernel_size=3, stride=self.strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = ResBlock(input_channels, self.nb_filter[0], self.strides[0], **kwargs)
        self.conv1_0 = res_unit(self.nb_filter[0], self.nb_filter[1], self.strides[1], **kwargs)
        self.conv2_0 = res_unit(self.nb_filter[1], self.nb_filter[2], self.strides[2], **kwargs)
        self.conv3_0 = res_unit(self.nb_filter[2], self.nb_filter[3], self.strides[3], **kwargs)
        self.conv4_0 = res_unit(self.nb_filter[3], self.nb_filter[4], self.strides[4], use_dropout=use_dropout, **kwargs)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        return x0_0, x1_0, x2_0, x3_0, x4_0


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, channels=(32, 64, 128, 256, 320),
                 strides=(2, 2, 2, 2, 2), use_dropout=False, **kwargs):
        """
        Args:
            num_classes: the number of classes in segmentation
            input_channels: the number of input channels
            channels: length-5 tuple, define the number of channels in each stage
            strides: tuple, define the stride in each stage
            use_dropout: bool, whether to use dropout
            **kwargs: other args define activation and normalization
        """
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides,
                                 use_dropout=use_dropout, **kwargs)
        nb_filter = self.backbone.nb_filter

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], use_dropout=use_dropout, **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], use_dropout=use_dropout, **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], use_dropout=use_dropout, **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], use_dropout=use_dropout, **kwargs)

        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def upsample(self, inputs, target):
        return F.interpolate(inputs, size=target.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.upsample(x4, x3)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.upsample(x3_1, x2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.upsample(x2_2, x1)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.upsample(x1_3, x0)], dim=1))

        out = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out

    def freeze(self, eval=True):
        # freeze the network
        for p in self.parameters():
            p.requires_grad = False
        if eval:
            self.eval()

    def unfreeze(self, train=True):
        # unfreeze the network to allow parameter update
        for p in self.parameters():
            p.requires_grad = True
        if train:
            self.train()


class UNet_Reg(nn.Module):
    def __init__(self, input_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2, 2),
                 input_shape=None, batch_size=None, **kwargs):
        """
        Args:
            num_classes: the number of classes in segmentation
            input_channels: the number of input channels
            channels: length-5 tuple, define the number of channels in each stage
            strides: tuple, define the stride in each stage
            use_dropout: bool, whether to use dropout
            **kwargs: other args define activation and normalization
        """
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides,
                                 use_dropout=False, **kwargs)
        nb_filter = self.backbone.nb_filter

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], use_dropout=False, **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], use_dropout=False, **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], use_dropout=False, **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], use_dropout=False, **kwargs)

        # flow layer
        self.flow = nn.Conv3d(nb_filter[0], 3, kernel_size=1)
        self.flow.weight = nn.Parameter(torch.distributions.Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # sample layer
        self.sample = SpatialSampleLayer(input_shape, batch_size)

    def upsample(self, inputs, target):
        return F.interpolate(inputs, size=target.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, moving_image, fixed_image):
        x = torch.cat([moving_image, fixed_image], dim=1)
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.upsample(x4, x3)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.upsample(x3_1, x2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.upsample(x2_2, x1)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.upsample(x1_3, x0)], dim=1))

        flow = F.interpolate(self.flow(x0_4), size=size, mode='trilinear', align_corners=False)
        sampled_flow = self.sample(flow)
        out = dict()
        out['flow'] = flow
        out['sampled_flow'] = sampled_flow
        return out

    def freeze(self, eval=True):
        # freeze the network
        for p in self.parameters():
            p.requires_grad = False
        if eval:
            self.eval()

    def unfreeze(self, train=True):
        # unfreeze the network to allow parameter update
        for p in self.parameters():
            p.requires_grad = True
        if train:
            self.train()


class TupleUNet(nn.Module):
    def __init__(self, decoder_channels, input_channels=1, channels=(32, 64, 128, 256, 320),
                 strides=(2, 2, 2, 2, 2), use_dropout=False, **kwargs):
        """
        Args:
            decoder_channels: the number of classes in segmentation
            input_channels: the number of input channels
            channels: length-5 tuple, define the number of channels in each stage
            strides: tuple, define the stride in each stage
            use_dropout: bool, whether to use dropout
            **kwargs: other args define activation and normalization
        """
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides,
                                 use_dropout=use_dropout, **kwargs)
        nb_filter = self.backbone.nb_filter

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        # decoder 1
        self.conv3_0 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], use_dropout=use_dropout, **kwargs)
        self.conv2_0 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], use_dropout=use_dropout, **kwargs)
        self.conv1_0 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], use_dropout=use_dropout, **kwargs)
        self.conv0_0 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], use_dropout=use_dropout, **kwargs)
        self.convds0 = nn.Conv3d(nb_filter[0], decoder_channels[0], kernel_size=1)
        # decoder 2
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], use_dropout=use_dropout, **kwargs)
        self.conv2_1 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], use_dropout=use_dropout, **kwargs)
        self.conv1_1 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], use_dropout=use_dropout, **kwargs)
        self.conv0_1 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], use_dropout=use_dropout, **kwargs)
        self.convds1 = nn.Conv3d(nb_filter[0], decoder_channels[1], kernel_size=1)

    def upsample(self, inputs, target):
        return F.interpolate(inputs, size=target.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_0 = self.conv3_0(torch.cat([x3, self.upsample(x4, x3)], dim=1))
        x2_0 = self.conv2_0(torch.cat([x2, self.upsample(x3_0, x2)], dim=1))
        x1_0 = self.conv1_0(torch.cat([x1, self.upsample(x2_0, x1)], dim=1))
        x0_0 = self.conv0_0(torch.cat([x0, self.upsample(x1_0, x0)], dim=1))
        out1 = F.interpolate(self.convds0(x0_0), size=size, mode='trilinear', align_corners=False)

        x3_1 = self.conv3_1(torch.cat([x3, self.upsample(x4, x3)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2, self.upsample(x3_1, x2)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1, self.upsample(x2_1, x1)], dim=1))
        x0_1 = self.conv0_1(torch.cat([x0, self.upsample(x1_1, x0)], dim=1))
        out2 = F.interpolate(self.convds1(x0_1), size=size, mode='trilinear', align_corners=False)
        return (out1, out2)

    def freeze(self, eval=True):
        # freeze the network
        for p in self.parameters():
            p.requires_grad = False
        if eval:
            self.eval()

    def unfreeze(self, train=True):
        # unfreeze the network to allow parameter update
        for p in self.parameters():
            p.requires_grad = True
        if train:
            self.train()


class AugmentationLayer(nn.Module):
    def __init__(self, input_shape=None, batch_size=None, **kwargs):
        super().__init__()
        self.spatial = SpatialSampleLayer(input_shape, batch_size)
        self.intensity = IntensitySampleLayer(input_shape, batch_size)

    def forward(self, spatial_flow, intensity_flow):
        out = dict()
        out['spatial'] = self.spatial(spatial_flow)
        out['intensity'] = self.intensity(intensity_flow)
        return out

    def freeze(self, eval=True):
        # freeze the network
        for p in self.parameters():
            p.requires_grad = False
        if eval:
            self.eval()

    def unfreeze(self, train=True):
        # unfreeze the network to allow parameter update
        for p in self.parameters():
            p.requires_grad = True
        if train:
            self.train()
