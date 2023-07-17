import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import MultiScaleLoss, DiceLoss, BendingEnergyLoss, LocalNormalizedCrossCorrelationLoss
from torchvision.models.segmentation.deeplabv3 import ASPP


__all__ = ['VecInt', 'ResizeTransform', 'SpatialTransformer', 'IntensitySampleLayer', 'DVF2DDF',
           'RegistrationCriterion']


class NCCLoss(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class GradLoss(nn.Module):
    def __init__(self, penalty='l2'):
        super().__init__()
        self.penalty = penalty

    def forward(self, dvf):
        dy = torch.abs(dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :])
        dx = torch.abs(dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :])
        dz = torch.abs(dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0
        return grad


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer()

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, mode):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class DVF2DDF(nn.Module):
    def __init__(self, down_stride=2, integration_steps=7, half_res=False):
        super().__init__()
        # configure optional resize layers (downsize)
        if not half_res and integration_steps > 0 and down_stride > 1:
            self.resize = ResizeTransform(down_stride, 'trilinear')
        else:
            self.resize = None

        # resize to full res
        if integration_steps > 0 and down_stride > 1:
            self.fullsize = ResizeTransform(1 / down_stride, 'trilinear')
        else:
            self.fullsize = None

        # configure optional integration layer for diffeomorphic warp
        self.integrate = VecInt(integration_steps) if integration_steps > 0 else None

    def forward(self, pos_flow):
        # downsample the flow
        if self.resize:
            pos_flow = self.resize(pos_flow)
        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
        return pos_flow


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def create_grid_buffer(self, flow):
        spatial_shape = flow.shape[2:]
        device = flow.device
        vectors = [torch.arange(0, s) for s in spatial_shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.to(torch.float32).to(device)  # (batch, spatial_dims, H, W, (D))
        # Registering the grid as a buffer cleanly moves it to the GPU, but it also adds it to the state dict.
        # Update: When using new PyTorch versions (>=1.7), set 'persistent=False' can solve this issue.
        # See: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict.
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, inputs, flow):
        # generate new locations according to the reference grid
        try:
            new_locs = self.grid + flow
        except:  # when the grid is not generated or the spatial shape of the grid does not match the flow's
            self.create_grid_buffer(flow)
            new_locs = self.grid + flow
        spatial_shape = flow.shape[2:]
        spatial_dims = len(spatial_shape)

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(spatial_shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (spatial_shape[i] - 1) - 0.5)

        # move channels dim to last position
        new_locs = new_locs.permute([0] + list(range(2, 2 + spatial_dims)) + [1])  # (batch, ..., spatial_dims)
        index_ordering = list(range(spatial_dims - 1, -1, -1))
        new_locs = new_locs[..., index_ordering]  # z, y, x -> x, y, z

        return F.grid_sample(inputs, new_locs, align_corners=True, mode=self.mode)


class RegistrationCriterion(nn.Module):
    def __init__(self, image_weight=1.0, label_weight=1.0, penalty=5.0, use_label_loss=True):
        super().__init__()
        # registration loss
        # 1. image loss, calculated with MSE/NCC
        self.image_loss = nn.MSELoss()
        # TODO: NCC loss cannot work with AMP
        # self.image_loss = LocalNormalizedCrossCorrelationLoss()
        # 2. label loss, used for registration
        self.label_loss = MultiScaleLoss(DiceLoss(), scales=[0, 1, 2, 4, 8, 16])
        # 3. regularization term
        self.regularization = BendingEnergyLoss()  # 0.5 penalty
        # self.regularization = GradLoss()  # 0.02 penalty
        # weights
        self.image_weight = image_weight
        self.label_weight = label_weight
        self.penalty = penalty

        self.use_label_loss = use_label_loss

    def forward(self, fixed_image, registered_image, ddf, fixed_label=None, registered_label=None):
        image_loss = self.image_weight * self.image_loss(registered_image, fixed_image)
        penalty = self.penalty * self.regularization(ddf)
        if not self.use_label_loss:
            return image_loss + penalty
        else:
            label_loss = self.label_weight * self.label_loss(registered_label, fixed_label)
            return image_loss + label_loss + penalty


class SpatialSampleLayer(nn.Module):
    def __init__(self, spatial_shape, batch_size, dtype=torch.float32):
        """
        Randomize the layer by learnable parameters to ensure adversarial augmentation
        """
        super().__init__()
        self.param = nn.Parameter(torch.rand([batch_size, 3, *spatial_shape], dtype=dtype))

    def forward(self, flow):
        augmented_flow = self.param * flow
        return augmented_flow

    def freeze(self):
        # freeze the network
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        # unfreeze the network to allow parameter update
        for p in self.parameters():
            p.requires_grad = True


class IntensitySampleLayer(nn.Module):
    def __init__(self, spatial_shape, batch_size, dtype=torch.float32):
        """
        Randomize the layer by learnable parameters to ensure adversarial augmentation
        """
        super().__init__()
        self.param = nn.Parameter(torch.rand([batch_size, 1, *spatial_shape], dtype=dtype))

    def forward(self, flow):
        augmented_flow = self.param * flow
        return augmented_flow

    def freeze(self):
        # freeze the network
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        # unfreeze the network to allow parameter update
        for p in self.parameters():
            p.requires_grad = True
