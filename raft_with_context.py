from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchvision.ops import Conv2dNormActivation

from torchvision.transforms._presets import OpticalFlow
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import handle_legacy_interface
from torchvision.models.optical_flow._utils import grid_sample, make_coords_grid, upsample_flow

# This implementation is build by adapting the source code provided by 
# https://github.com/princeton-vl/RAFT
# https://github.com/uzh-rpg/E-RAFT/tree/main
# https://pytorch.org/vision/main/models/raft.html

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_layer, stride=1, always_project: bool = False):
        super().__init__()

        self.convnormrelu1 = Conv2dNormActivation(in_channels, out_channels, norm_layer=norm_layer, kernel_size=3, stride=stride, bias=True)
        self.convnormrelu2 = Conv2dNormActivation(out_channels, out_channels, norm_layer=norm_layer, kernel_size=3, bias=True)
        self.downsample: nn.Module

        if stride == 1 and not always_project: self.downsample = nn.Identity()
        else: self.downsample = Conv2dNormActivation(in_channels, out_channels, norm_layer=norm_layer, kernel_size=1, stride=stride, bias=True, activation_layer=None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        y = self.convnormrelu2(self.convnormrelu1(y))
        x = self.downsample(x)
        return self.relu(x + y)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm_layer, stride=1):
        super().__init__()
        self.convnormrelu1 = Conv2dNormActivation(in_channels, out_channels // 4, norm_layer=norm_layer, kernel_size=1, bias=True)
        self.convnormrelu2 = Conv2dNormActivation(out_channels // 4, out_channels // 4, norm_layer=norm_layer, kernel_size=3, stride=stride, bias=True)
        self.convnormrelu3 = Conv2dNormActivation(out_channels // 4, out_channels, norm_layer=norm_layer, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1: self.downsample = nn.Identity()
        else: self.downsample = Conv2dNormActivation(in_channels, out_channels, norm_layer=norm_layer, kernel_size=1, stride=stride, bias=True, activation_layer=None)

    def forward(self, x):
        y = x
        y = self.convnormrelu3(self.convnormrelu2(self.convnormrelu1(y)))
        x = self.downsample(x)
        return self.relu(x + y)

class FeatureEncoder(nn.Module):
    def __init__(self, *, block=ResidualBlock, layers=(64, 64, 96, 128, 256), strides=(2, 1, 2, 2), norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.convnormrelu = Conv2dNormActivation(3, layers[0], norm_layer=norm_layer, kernel_size=7, stride=strides[0], bias=True)
        self.layer1 = self._make_2_blocks(block, layers[0], layers[1], norm_layer=norm_layer, first_stride=strides[1])
        self.layer2 = self._make_2_blocks(block, layers[1], layers[2], norm_layer=norm_layer, first_stride=strides[2])
        self.layer3 = self._make_2_blocks(block, layers[2], layers[3], norm_layer=norm_layer, first_stride=strides[3])
        self.conv = nn.Conv2d(layers[3], layers[4], kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        num_downsamples = len(list(filter(lambda s: s == 2, strides)))
        self.output_dim = layers[-1]
        self.downsample_factor = 2**num_downsamples

    def _make_2_blocks(self, block, in_channels, out_channels, norm_layer, first_stride):
        block1 = block(in_channels, out_channels, norm_layer=norm_layer, stride=first_stride)
        block2 = block(out_channels, out_channels, norm_layer=norm_layer, stride=1)
        return nn.Sequential(block1, block2)

    def forward(self, x): return self.conv(self.layer3(self.layer2(self.layer1(self.convnormrelu(x)))))

class ContextEncoder(nn.Module):
    def __init__(self, *, block=ResidualBlock, layers=(64, 64, 96, 128, 256), strides=(2, 1, 2, 2), norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.convnormrelu = Conv2dNormActivation(3, layers[0], norm_layer=norm_layer, kernel_size=7, stride=strides[0], bias=True)
        self.layer1 = self._make_2_blocks(block, layers[0], layers[1], norm_layer=norm_layer, first_stride=strides[1])
        self.layer2 = self._make_2_blocks(block, layers[1], layers[2], norm_layer=norm_layer, first_stride=strides[2])
        self.layer3 = self._make_2_blocks(block, layers[2], layers[3], norm_layer=norm_layer, first_stride=strides[3])
        self.conv = nn.Conv2d(layers[3], layers[4], kernel_size=1)
        #self.conv_integ = nn.Conv2d(layers[4] * 2, layers[4], kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None: nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        num_downsamples = len(list(filter(lambda s: s == 2, strides)))
        self.output_dim = layers[-1]
        self.downsample_factor = 2**num_downsamples

    def _make_2_blocks(self, block, in_channels, out_channels, norm_layer, first_stride):
        block1 = block(in_channels, out_channels, norm_layer=norm_layer, stride=first_stride)
        block2 = block(out_channels, out_channels, norm_layer=norm_layer, stride=1)
        return nn.Sequential(block1, block2)

    def forward(self, x, precon):
        context = self.conv(self.layer3(self.layer2(self.layer1(self.convnormrelu(x)))))
        #if precon is not None: context = self.conv_integ(torch.cat([context,precon],dim=1))
        return context

class MotionEncoder(nn.Module):
    def __init__(self, *, in_channels_corr, corr_layers=(256, 192), flow_layers=(128, 64), out_channels=128):
        super().__init__()
        self.convcorr1 = Conv2dNormActivation(in_channels_corr, corr_layers[0], norm_layer=None, kernel_size=1)
        if len(corr_layers) == 2: self.convcorr2 = Conv2dNormActivation(corr_layers[0], corr_layers[1], norm_layer=None, kernel_size=3)
        else: self.convcorr2 = nn.Identity()

        self.convflow1 = Conv2dNormActivation(2, flow_layers[0], norm_layer=None, kernel_size=7)
        self.convflow2 = Conv2dNormActivation(flow_layers[0], flow_layers[1], norm_layer=None, kernel_size=3)
        self.conv = Conv2dNormActivation(corr_layers[-1] + flow_layers[-1], out_channels - 2, norm_layer=None, kernel_size=3)
        self.out_channels = out_channels

    def forward(self, flow, corr_features):
        corr = self.convcorr1(corr_features)
        corr = self.convcorr2(corr)

        flow_orig = flow
        flow = self.convflow1(flow)
        flow = self.convflow2(flow)

        corr_flow = torch.cat([corr, flow], dim=1)
        corr_flow = self.conv(corr_flow)
        return torch.cat([corr_flow, flow_orig], dim=1)

class ConvGRU(nn.Module):
    def __init__(self, *, input_size, hidden_size, kernel_size, padding):
        super().__init__()
        self.convz = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convr = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)
        self.convq = nn.Conv2d(hidden_size + input_size, hidden_size, kernel_size=kernel_size, padding=padding)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h

def _pass_through_h(h, _): return h

class RecurrentBlock(nn.Module):
    def __init__(self, *, input_size, hidden_size, kernel_size=((1, 5), (5, 1)), padding=((0, 2), (2, 0))):
        super().__init__()

        self.convgru1 = ConvGRU(input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[0], padding=padding[0])
        if len(kernel_size) == 2: self.convgru2 = ConvGRU(input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size[1], padding=padding[1])
        else: self.convgru2 = _pass_through_h
        self.hidden_size = hidden_size

    def forward(self, h, x):
        h = self.convgru1(h, x)
        h = self.convgru2(h, x)
        return h


class FlowHead(nn.Module):
    def __init__(self, *, in_channels, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class UpdateBlock(nn.Module):
    def __init__(self, *, motion_encoder, recurrent_block, flow_head):
        super().__init__()
        self.motion_encoder = motion_encoder
        self.recurrent_block = recurrent_block
        self.flow_head = flow_head
        self.hidden_state_size = recurrent_block.hidden_size

    def forward(self, hidden_state, context, corr_features, flow):
        motion_features = self.motion_encoder(flow, corr_features)
        x = torch.cat([context, motion_features], dim=1)
        hidden_state = self.recurrent_block(hidden_state, x)
        delta_flow = self.flow_head(hidden_state)
        return hidden_state, delta_flow

class MaskPredictor(nn.Module):
    def __init__(self, *, in_channels, hidden_size, multiplier=0.25):
        super().__init__()
        self.convrelu = Conv2dNormActivation(in_channels, hidden_size, norm_layer=None, kernel_size=3)
        self.conv = nn.Conv2d(hidden_size, 8 * 8 * 9, 1, padding=0)
        self.multiplier = multiplier

    def forward(self, x):
        x = self.convrelu(x)
        x = self.conv(x)
        return self.multiplier * x

class CorrBlock(nn.Module):
    def __init__(self, *, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid: List[Tensor] = [torch.tensor(0)]
        self.out_channels = num_levels * (2 * radius + 1) ** 2

    def build_pyramid(self, fmap1, fmap2):
        corr_volume = self._compute_corr_volume(fmap1, fmap2)
        batch_size, h, w, num_channels, _, _ = corr_volume.shape  # _, _ = h, w
        corr_volume = corr_volume.reshape(batch_size * h * w, num_channels, h, w)
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords):
        neighborhood_side_len = 2 * self.radius + 1
        di = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        dj = torch.linspace(-self.radius, self.radius, neighborhood_side_len)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1).to(centroids_coords.device)
        delta = delta.view(1, neighborhood_side_len, neighborhood_side_len, 2)

        batch_size, _, h, w = centroids_coords.shape  # _ = 2
        centroids_coords = centroids_coords.permute(0, 2, 3, 1).reshape(batch_size * h * w, 1, 1, 2)

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
            sampling_coords = centroids_coords + delta  # end shape is (batch_size * h * w, side_len, side_len, 2)
            indexed_corr_volume = grid_sample(corr_volume, sampling_coords, align_corners=True, mode="bilinear").view(batch_size, h, w, -1)
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords = centroids_coords / 2
        corr_features = torch.cat(indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous()
        expected_output_shape = (batch_size, self.out_channels, h, w)
        return corr_features

    def _compute_corr_volume(self, fmap1, fmap2):
        batch_size, num_channels, h, w = fmap1.shape
        fmap1 = fmap1.view(batch_size, num_channels, h * w)
        fmap2 = fmap2.view(batch_size, num_channels, h * w)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch_size, h, w, 1, h, w)
        return corr / torch.sqrt(torch.tensor(num_channels))


class RAFT(nn.Module):
    def __init__(self, *, feature_encoder, context_encoder, corr_block, update_block, mask_predictor=None):
        super().__init__()
        _log_api_usage_once(self)

        self.feature_encoder = feature_encoder
        self.context_encoder = context_encoder
        self.corr_block = corr_block
        self.update_block = update_block
        self.mask_predictor = mask_predictor

    def forward(self, image1, image2, num_flow_updates: int = 30, flow_init=None, prev_fmap=None, return_context=False):
        batch_size, _, h, w = image1.shape
        fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
        fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)
        self.corr_block.build_pyramid(fmap1, fmap2)
        context_out = self.context_encoder(image1,prev_fmap)

        hidden_state_size = self.update_block.hidden_state_size
        out_channels_context = context_out.shape[1] - hidden_state_size
        
        hidden_state, context = torch.split(context_out, [hidden_state_size, out_channels_context], dim=1)
        hidden_state = torch.tanh(hidden_state)
        context = F.relu(context)

        coords0 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
        coords1 = make_coords_grid(batch_size, h // 8, w // 8).to(fmap1.device)
        
        if flow_init is not None: coords1 = coords1 + flow_init

        for _ in range(num_flow_updates):
            coords1 = coords1.detach()
            corr_features = self.corr_block.index_pyramid(centroids_coords=coords1)

            flow = coords1 - coords0
            hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)

            coords1 = coords1 + delta_flow

            up_mask = None if self.mask_predictor is None else self.mask_predictor(hidden_state)
            upsampled_flow = upsample_flow(flow=(coords1 - coords0), up_mask=up_mask)

        if return_context == True: return coords1 - coords0
        return upsampled_flow

def _raft(*, weights=None, progress=False, feature_encoder_layers, feature_encoder_block, feature_encoder_norm_layer,
        context_encoder_layers, context_encoder_block, context_encoder_norm_layer, corr_block_num_levels,
        corr_block_radius, motion_encoder_corr_layers, motion_encoder_flow_layers, motion_encoder_out_channels,
        recurrent_block_hidden_state_size, recurrent_block_kernel_size, recurrent_block_padding,
        flow_head_hidden_size, use_mask_predictor, **kwargs):
    
    feature_encoder = kwargs.pop("feature_encoder", None) or FeatureEncoder(
        block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer)
    
    context_encoder = kwargs.pop("context_encoder", None) or ContextEncoder(
        block=context_encoder_block, layers=context_encoder_layers, norm_layer=context_encoder_norm_layer)

    corr_block = kwargs.pop("corr_block", None) or CorrBlock(num_levels=corr_block_num_levels, radius=corr_block_radius)
    
    update_block = kwargs.pop("update_block", None)
    if update_block is None:
        motion_encoder = MotionEncoder(
            in_channels_corr=corr_block.out_channels,
            corr_layers=motion_encoder_corr_layers,
            flow_layers=motion_encoder_flow_layers,
            out_channels=motion_encoder_out_channels)

        out_channels_context = context_encoder_layers[-1] - recurrent_block_hidden_state_size
        recurrent_block = RecurrentBlock(
            input_size=motion_encoder.out_channels + out_channels_context,
            hidden_size=recurrent_block_hidden_state_size,
            kernel_size=recurrent_block_kernel_size,
            padding=recurrent_block_padding)

        flow_head = FlowHead(in_channels=recurrent_block_hidden_state_size, hidden_size=flow_head_hidden_size)
        update_block = UpdateBlock(motion_encoder=motion_encoder, recurrent_block=recurrent_block, flow_head=flow_head)

    mask_predictor = kwargs.pop("mask_predictor", None)
    if mask_predictor is None and use_mask_predictor: mask_predictor = MaskPredictor(
            in_channels=recurrent_block_hidden_state_size, hidden_size=256, multiplier=0.25)

    model = RAFT(feature_encoder=feature_encoder, context_encoder=context_encoder, corr_block=corr_block,
        update_block=update_block, mask_predictor=mask_predictor, **kwargs)

    if weights is not None: model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

def raft_context_model(*, weights = None, progress=True, **kwargs) -> RAFT:

    return _raft(weights=weights, progress=progress,
        # Feature encoder
        feature_encoder_layers=(64, 64, 96, 128, 256), feature_encoder_block=ResidualBlock, feature_encoder_norm_layer=InstanceNorm2d,
        # Context encoder
        context_encoder_layers=(64, 64, 96, 128, 256), context_encoder_block=ResidualBlock, context_encoder_norm_layer=BatchNorm2d,
        # Correlation block
        corr_block_num_levels=4, corr_block_radius=4,
        # Motion encoder
        motion_encoder_corr_layers=(256, 192), motion_encoder_flow_layers=(128, 64), motion_encoder_out_channels=128,
        # Recurrent block
        recurrent_block_hidden_state_size=128, recurrent_block_kernel_size=((1, 5), (5, 1)), recurrent_block_padding=((0, 2), (2, 0)),
        # Flow head
        flow_head_hidden_size=256,
        # Mask predictor
        use_mask_predictor=True, **kwargs)
