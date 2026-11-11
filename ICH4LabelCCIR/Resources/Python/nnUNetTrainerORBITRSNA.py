"""Inference-compatible ORBIT-RSNA trainer bundled with the Slicer module.

This file reconstructs the network architecture used by method2.py. It does
not train or initialize from the original source checkpoints. During Slicer
inference nnU-Net builds this architecture and loads the already-trained
ORBIT-RSNA ``checkpoint_best.pth`` selected by the user.
"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


FUSE_LAST_N_STAGES = 4
ADAPTER_REDUCTION = 8
MAX_RESIDUAL_STRENGTH = 0.25


def group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class OrthogonalResidualFusion3D(nn.Module):
    def __init__(self, channels: int, reduction: int, max_strength: float):
        super().__init__()
        channels = int(channels)
        hidden = max(8, channels // max(1, int(reduction)))
        groups = group_count(channels)
        self.seg_norm = nn.GroupNorm(groups, channels, affine=True)
        self.rsna_norm = nn.GroupNorm(groups, channels, affine=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(2 * channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(8, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.adapter_in = nn.Conv3d(channels, hidden, 1, bias=False)
        self.adapter_norm = nn.GroupNorm(group_count(hidden), hidden, affine=True)
        self.adapter_out = nn.Conv3d(hidden, channels, 1, bias=True)
        nn.init.zeros_(self.adapter_out.weight)
        nn.init.zeros_(self.adapter_out.bias)
        self.strength_logit = nn.Parameter(torch.tensor(0.0))
        self.max_strength = float(max_strength)

    @staticmethod
    def orthogonal_component(
        rsna: torch.Tensor, segmentation: torch.Tensor
    ) -> torch.Tensor:
        with torch.autocast(device_type=segmentation.device.type, enabled=False):
            source = rsna.float()
            reference = segmentation.float()
            unit = F.normalize(reference, p=2.0, dim=1, eps=1e-6)
            parallel = (source * unit).sum(dim=1, keepdim=True) * unit
            novelty = source - parallel
        return novelty.to(dtype=segmentation.dtype)

    def forward(
        self, segmentation: torch.Tensor, rsna: torch.Tensor
    ) -> torch.Tensor:
        if rsna.shape[2:] != segmentation.shape[2:]:
            rsna = F.interpolate(
                rsna,
                size=segmentation.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        if rsna.shape[1] != segmentation.shape[1]:
            raise RuntimeError(
                f"ORBIT feature channel mismatch: seg={segmentation.shape}, "
                f"rsna={rsna.shape}"
            )
        seg = self.seg_norm(segmentation)
        source = self.rsna_norm(rsna)
        novelty = self.orthogonal_component(source, seg)
        channel = self.channel_gate(torch.cat((seg, novelty), dim=1))
        spatial = self.spatial_gate(
            torch.cat(
                (
                    seg.abs().mean(dim=1, keepdim=True),
                    novelty.abs().mean(dim=1, keepdim=True),
                ),
                dim=1,
            )
        )
        correction = self.adapter_out(
            F.silu(self.adapter_norm(self.adapter_in(novelty)), inplace=True)
        )
        strength = self.max_strength * torch.tanh(self.strength_logit)
        return segmentation + strength * channel * spatial * correction


class ORBITRSNANetwork(nn.Module):
    def __init__(self, base_network: nn.Module, fuse_last_n: int):
        super().__init__()
        if not hasattr(base_network, "encoder") or not hasattr(base_network, "decoder"):
            raise RuntimeError(
                f"Expected encoder-decoder nnU-Net, got {type(base_network).__name__}"
            )
        self.encoder = base_network.encoder
        self.decoder = base_network.decoder
        self.rsna_encoder = copy.deepcopy(base_network.encoder)
        channels = getattr(self.encoder, "output_channels", None)
        if channels is None:
            raise RuntimeError("encoder.output_channels is unavailable")
        self.encoder_channels = [int(value) for value in channels]
        number = min(len(self.encoder_channels), max(1, int(fuse_last_n)))
        self.first_fused_stage = len(self.encoder_channels) - number
        self.fusion_blocks = nn.ModuleDict(
            {
                str(index): OrthogonalResidualFusion3D(
                    self.encoder_channels[index],
                    ADAPTER_REDUCTION,
                    MAX_RESIDUAL_STRENGTH,
                )
                for index in range(self.first_fused_stage, len(self.encoder_channels))
            }
        )

        # These persistent buffers are overwritten by the trained checkpoint.
        # Dummy initialization makes architecture reconstruction independent of
        # the original training workstation and its plans JSON path.
        self.register_buffer("ct_mean", torch.tensor(0.0), persistent=True)
        self.register_buffer("ct_std", torch.tensor(1.0), persistent=True)
        self.register_buffer(
            "rsna_window", torch.tensor([-20.0, 100.0]), persistent=True
        )
        for parameter in self.rsna_encoder.parameters():
            parameter.requires_grad_(False)
        self.rsna_encoder.eval()

    def nnunet_to_rsna(self, tensor: torch.Tensor) -> torch.Tensor:
        hu = tensor * self.ct_std.to(tensor.dtype) + self.ct_mean.to(tensor.dtype)
        lower = self.rsna_window[0].to(tensor.dtype)
        upper = self.rsna_window[1].to(tensor.dtype)
        windowed = torch.minimum(torch.maximum(hu, lower), upper)
        return 2.0 * (windowed - lower) / (upper - lower) - 1.0

    def train(self, mode: bool = True):
        super().train(mode)
        self.rsna_encoder.eval()
        return self

    def forward(self, tensor: torch.Tensor):
        segmentation_features = self.encoder(tensor)
        with torch.no_grad():
            rsna_features = self.rsna_encoder(self.nnunet_to_rsna(tensor))
        if len(segmentation_features) != len(rsna_features):
            raise RuntimeError(
                f"ORBIT encoder stage mismatch: {len(segmentation_features)} "
                f"vs {len(rsna_features)}"
            )
        fused = list(segmentation_features)
        for index_text, block in self.fusion_blocks.items():
            index = int(index_text)
            fused[index] = block(fused[index], rsna_features[index])
        return self.decoder(fused)

    def compute_conv_feature_map_size(self, input_size):
        total = 0
        for module in (self.encoder, self.decoder, self.rsna_encoder):
            if hasattr(module, "compute_conv_feature_map_size"):
                total += module.compute_conv_feature_map_size(input_size)
        return total


class nnUNetTrainerORBITRSNA(nnUNetTrainer):
    """Architecture entry point required by nnUNetv2_predict."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if num_input_channels != 1:
            raise RuntimeError(
                f"ORBIT-RSNA expects one CT input, got {num_input_channels}"
            )
        base = nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )
        return ORBITRSNANetwork(base, FUSE_LAST_N_STAGES)

    def _do_i_compile(self):
        return False
