'''
Code from the 3D UNet implementation:
https://github.com/wolny/pytorch-3dunet/
'''
import importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial

from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from torch_scatter import scatter_mean, scatter_max, scatter_softmax, scatter_add

# depth = 3 # 4

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1, groups=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='crg',
                 num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        self.apply_pooling = apply_pooling
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
        self.share_channels = 8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

        # self.fc_c = nn.Linear(in_channels, out_channels)
        
        self.fc_k = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )
        self.fc_v = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 2*out_channels),
            nn.ReLU(),
            nn.Linear(2*out_channels, out_channels)
        )
        self.conv_q = conv3d(out_channels, out_channels, 1, bias=True, padding=0)
        self.fc_pos = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_channels)
        )
        self.fc_w = nn.Sequential(
            nn.Linear(out_channels, out_channels // self.share_channels),
            nn.BatchNorm1d(out_channels // self.share_channels), nn.ReLU(inplace=True))
        
        self.weight_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.conv1x1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.sample_mode = 'bilinear'
        self.padding = 0.1

    def generate_grid_features2(self, p, c, k, q, channel, reso_grid):
        
        b, n, _ = p.shape

        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(p_nor, reso_grid, coord_type='3d')
        
        pos_enc = p_nor - (p_nor * reso_grid).floor() / reso_grid
        for i, layer in enumerate(self.fc_pos):
            pos_enc = layer(pos_enc.transpose(1, 2).contiguous()).transpose(1, 2) if i == 1 else layer(pos_enc)
        
        batch_index = torch.arange(p.size(0), device=p.device).repeat_interleave(p.size(1))
        
        w = k - q.reshape(-1, channel, reso_grid**3)[batch_index, :, index.reshape(-1)].reshape(k.shape[0], k.shape[1], channel) + pos_enc
        w = w.reshape(k.shape[0]*k.shape[1], -1)
        w = self.fc_w(w)
        w = w.reshape(k.shape[0], k.shape[1], -1)
        
        
        weights_soft = scatter_softmax(w.transpose(1, 2), index).transpose(1, 2).reshape(b*n, -1)
        value = (c + pos_enc).reshape(b*n, -1)
        value_grid = (value.reshape(b*n, self.share_channels, self.out_channels // self.share_channels) * weights_soft.unsqueeze(-2)).view(b*n, -1)
        value_grid = value_grid.reshape(b, n, -1).transpose(1, 2)
        
        
        fea_plane = c.new_zeros(p.size(0), channel, reso_grid**3)
        c = c.permute(0, 2, 1) # B x 512 x T

        fea_plane = scatter_add(value_grid, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_grid, reso_grid, reso_grid) # sparce matrix (B x 512 x reso x reso)

        return fea_plane
        
    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c
        
    def forward(self, x, p, c):
        b, n, _ = c.shape
        
        if self.pooling is not None:
            x['grid'] = self.pooling(x['grid'])

        x['grid'] = self.basic_module(x['grid'])
        
        x_down = {}
        x_down['grid'] = x['grid']
        
        channel = x['grid'].shape[1]
        reso_grid = x['grid'].shape[2]

        c = c.reshape(b*n, -1)
        k = self.fc_k(c)
        v = self.fc_v(c)
        c = c.reshape(b, n, -1)
        k = k.reshape(b, n, -1)
        v = v.reshape(b, n, -1)
        q = {}
        q['grid'] = self.conv_q(x['grid'])
        
        x['grid'] = self.generate_grid_features2(p, v, k, q['grid'], channel, reso_grid)
        x['grid'] = self.weight_conv(x['grid'])
        x['grid'] = x['grid'] + self.conv1x1(x_down['grid'])

        c = v
        c += self.sample_grid_feature(p, x['grid']).transpose(1, 2)
        
        before_pool = {}
        before_pool['grid'] = x['grid']

        return x, before_pool, c


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (DoubleConv or ExtResNetBlock).
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_feature_conv1x1, in_channels, out_channels, kernel_size=3, scale_factor=(2, 2, 2), basic_module=DoubleConv,
                 conv_layer_order='crg', num_groups=8, mode='nearest', apply_pooling=False, unet_levels=4):
        super(Decoder, self).__init__()

        self.if_upsampling = apply_pooling
        self.unet_levels = unet_levels
        self.share_channels = 8
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.if_upsampling:
          if basic_module == DoubleConv:
              self.upsampling = Upsampling(transposed_conv=False, in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, scale_factor=scale_factor, mode=mode)
              self.joining = partial(self._joining, concat=True)                      
          else:
              self.upsampling = Upsampling(transposed_conv=True, in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=kernel_size, scale_factor=scale_factor, mode=mode)
              self.joining = partial(self._joining, concat=False)
              in_channels = out_channels
        else:
          self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)
        
        self.fc_k = nn.Sequential(
            nn.Linear(in_feature_conv1x1, out_channels)
        )
        self.fc_v = nn.Sequential(
            nn.Linear(in_feature_conv1x1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 2*out_channels),
            nn.ReLU(),
            nn.Linear(2*out_channels, out_channels)
        )
        self.conv_q = conv3d(out_channels, out_channels, 1, bias=True, padding=0)
        self.fc_pos = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_channels)
        )
        self.fc_w = nn.Sequential(
            nn.Linear(out_channels, out_channels // self.share_channels),
            nn.BatchNorm1d(out_channels // self.share_channels), nn.ReLU(inplace=True))
        
        self.conv1x1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.weight_conv = nn.Sequential(
            nn.Conv3d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels
            ),
            nn.Conv3d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels
            )
        )
        
        self.conv_final = nn.Sequential(
            nn.Conv3d(
                out_channels=out_channels,
                in_channels=out_channels,
                kernel_size=1,
                stride=1
            ),
            nn.BatchNorm3d(out_channels)
        )
        
        
        self.sample_mode = 'bilinear'
        self.padding = 0.1

    def generate_grid_features2(self, p, c, k, q, channel, reso_grid):
        
        b, n, _ = p.shape

        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(p_nor, reso_grid, coord_type='3d')
        
        pos_enc = p_nor - (p_nor * reso_grid).floor() / reso_grid
        for i, layer in enumerate(self.fc_pos):
            pos_enc = layer(pos_enc.transpose(1, 2).contiguous()).transpose(1, 2) if i == 1 else layer(pos_enc)
        
        batch_index = torch.arange(p.size(0), device=p.device).repeat_interleave(p.size(1))
        
        w = k - q.reshape(-1, channel, reso_grid**3)[batch_index, :, index.reshape(-1)].reshape(k.shape[0], k.shape[1], channel) + pos_enc
        w = w.reshape(k.shape[0]*k.shape[1], -1)
        w = self.fc_w(w)
        w = w.reshape(k.shape[0], k.shape[1], -1)
        
        
        weights_soft = scatter_softmax(w.transpose(1, 2), index).transpose(1, 2).reshape(b*n, -1)
        value = (c + pos_enc).reshape(b*n, -1)
        value_grid = (value.reshape(b*n, self.share_channels, self.out_channels // self.share_channels) * weights_soft.unsqueeze(-2)).view(b*n, -1)
        value_grid = value_grid.reshape(b, n, -1).transpose(1, 2)
        
        
        fea_plane = c.new_zeros(p.size(0), channel, reso_grid**3)
        c = c.permute(0, 2, 1) # B x 512 x T

        fea_plane = scatter_add(value_grid, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_grid, reso_grid, reso_grid) # sparce matrix (B x 512 x reso x reso)

        return fea_plane
        
    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, from_down, from_up, c_last, i):
        b, n, _ = c_last.shape
        
        if self.if_upsampling:
          from_up['grid'] = self.upsampling(encoder_features=from_down, x=from_up['grid'])
        
        x = {}
        x['grid'] = self.joining(from_down, from_up['grid'])
        
        x_up = {}
        x['grid'] = self.basic_module(x['grid'])
        x_up['grid'] = x['grid']
        
        channel = x['grid'].shape[1]
        reso_grid = x['grid'].shape[2]

        if i == self.unet_levels - 2:
            return x, c_last 

        c = c_last
        
        c = c.reshape(b*n, -1)
        k = self.fc_k(c)
        v = self.fc_v(c)
        c = c.reshape(b, n, -1)
        k = k.reshape(b, n, -1)
        v = v.reshape(b, n, -1)
        q = {}
        q['grid'] = self.conv_q(x['grid'])
        
        x['grid'] = self.generate_grid_features2(p, v, k, q['grid'], channel, reso_grid)
        x['grid'] = self.weight_conv(x['grid'])
        x['grid'] = x['grid'] + self.conv1x1(x_up['grid'])
        x['grid'] = self.conv_final(x['grid'])
        
        c = v
        c += self.sample_grid_feature(p, x['grid']).transpose(1, 2)
        
        return x, c 

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class Upsampling(nn.Module):
    """
    Upsamples a given multi-channel 3D data using either interpolation or learned transposed convolution.

    Args:
        transposed_conv (bool): if True uses ConvTranspose3d for upsampling, otherwise uses interpolation
        concat_joining (bool): if True uses concatenation joining between encoder and decoder features, otherwise
            uses summation joining (see Residual U-Net)
        in_channels (int): number of input channels for transposed conv
        out_channels (int): number of output channels for transpose conv
        kernel_size (int or tuple): size of the convolving kernel
        scale_factor (int or tuple): stride of the convolution
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default: 'nearest'
    """

    def __init__(self, transposed_conv, in_channels=None, out_channels=None, kernel_size=3,
                 scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=scale_factor,
                                               padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


class FinalConv(nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8):
        super(FinalConv, self).__init__()

        # conv1
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))

        # in the last layer a 1Ãƒâ€”1 convolution reduces the number of output channels to out_channels
        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)

class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        f_maps (int, tuple): if int: number of feature maps in the first conv layer of the encoder (default: 64);
            if tuple: number of feature maps at each level
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=False, testing=False, **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing
        self.unet = kwargs['is_unet']
        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i < 2:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=basic_module,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, apply_pooling=self.unet, basic_module=basic_module,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))

        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            if i > 1:
                self.unet = False
            decoder = Decoder(reversed_f_maps[i], in_feature_num, out_feature_num, basic_module=basic_module,
                              conv_layer_order=layer_order, num_groups=num_groups, apply_pooling=self.unet, unet_levels=num_levels)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)


    def forward(self, x, p, c):
        encoders_features = []
        x_out = []
        for encoder in self.encoders:
            x, before_pool, c = encoder(x, p, c)
            encoders_features.append(before_pool)
        
        for i, module in enumerate(self.decoders):
            before_pool = encoders_features[-(i + 2)]['grid']
            x, c = module(p, before_pool, x, c, i)
            x_ = {}
            x_['grid'] = x['grid']
            x_out.append(x_)
        return x_out


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_groups=num_groups, num_levels=num_levels, is_segmentation=is_segmentation,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock, f_maps=f_maps, layer_order=layer_order,
                                             num_groups=num_groups, num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             **kwargs)


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('pytorch3dunet.unet3d.model')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)
