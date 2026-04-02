"""Model definitions for DINOv2-UNet polyp segmentation.

This module implements the encoder-decoder architecture described in the paper:
- **VitDinoV2Encoder**: A DINOv2 ViT-B/14 backbone with partial fine-tuning.
- **EnhancedDecoder** (Streamlined): U-shaped decoder with skip connections.
- **AttentionDecoder** (Complex): Decoder variant with Attention Gates for ablation.
- **DinoV2UNet**: Full segmentation model combining encoder + decoder + deep supervision.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as exc:
    raise RuntimeError("timm is required: install with `pip install timm`.") from exc


class VitDinoV2Encoder(nn.Module):
    """DINOv2 Vision Transformer encoder with partial fine-tuning.

    Extracts multi-scale token features from specified transformer blocks
    and projects them into a uniform channel dimension via 1x1 convolutions.

    Args:
        backbone: Name of the timm ViT model to load.
        out_indices: Tuple of block indices from which to extract features.
        pretrained: Whether to load pretrained weights.
        freeze_blocks_until: Blocks with index < this value are frozen.
        pretrained_type: Type of pretrained weights ('dinov2' or 'imagenet_supervised').
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch14_dinov2",
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        pretrained: bool = True,
        freeze_blocks_until: int = 6,
        pretrained_type: str = "dinov2",
    ):
        super().__init__()
        # Select backbone based on pretrained_type
        if pretrained_type == "imagenet_supervised":
            backbone = "vit_base_patch16_224"

        self.model = timm.create_model(
            backbone, pretrained=pretrained, dynamic_img_size=True
        )
        if hasattr(self.model, "patch_embed") and hasattr(
            self.model.patch_embed, "dynamic_img_size"
        ):
            self.model.patch_embed.dynamic_img_size = True

        self.out_indices = out_indices
        self.patch_size = (
            self.model.patch_embed.patch_size[0]
            if isinstance(self.model.patch_embed.patch_size, tuple)
            else int(self.model.patch_embed.patch_size)
        )
        self.embed_dim = self.model.embed_dim

        # Freeze early blocks, train deeper blocks
        for i, blk in enumerate(self.model.blocks):
            requires = i >= freeze_blocks_until
            for p in blk.parameters():
                p.requires_grad = requires

        # 1x1 projection to uniform channel dimension (Feature Projection Module)
        self.projs = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, 256, kernel_size=1) for _ in out_indices]
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Tuple[int, int], int]:
        """Extract multi-scale features from the ViT backbone.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            A tuple of (features, grid_hw, patch_size) where features is a list
            of projected feature maps, each of shape (B, 256, Gh, Gw).
        """
        B, C, H, W = x.shape
        x_pe = self.model.patch_embed(x)

        if hasattr(self.model, "_pos_embed"):
            pe_out = self.model._pos_embed(x_pe)
            if isinstance(pe_out, (list, tuple)):
                x_tokens, (Gh, Gw) = pe_out[0], pe_out[1]
            else:
                x_tokens, (Gh, Gw) = pe_out, (
                    H // self.patch_size,
                    W // self.patch_size,
                )
        else:
            x_tokens, (Gh, Gw) = x_pe, (H // self.patch_size, W // self.patch_size)
            if hasattr(self.model, "pos_embed") and self.model.pos_embed is not None:
                x_tokens = x_tokens + self.model.pos_embed

        x_tokens = self.model.pos_drop(x_tokens)

        feats_tokens = []
        for i, blk in enumerate(self.model.blocks):
            x_tokens = blk(x_tokens)
            if i in self.out_indices:
                feats_tokens.append(x_tokens)

        feats = []
        for tok, proj in zip(feats_tokens, self.projs):
            if tok.shape[1] == (Gh * Gw + 1):
                tok = tok[:, 1:, :]  # Discard [CLS] token
            fm = tok.transpose(1, 2).reshape(B, self.embed_dim, Gh, Gw)
            fm = proj(fm)
            feats.append(fm)

        return feats, (Gh, Gw), self.patch_size


class ConvBlock(nn.Module):
    """Standard convolutional block: Conv3x3 -> BN -> ReLU -> Dropout -> Conv3x3 -> BN -> ReLU.

    Used as the refinement unit in the U-shaped decoder (Fusion Unit).

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        dropout_p: Dropout probability between the two conv layers.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention Gate for skip connection refinement.

    Filters skip-connection features by applying a gating signal from the
    coarser decoder level, suppressing irrelevant activations.

    Args:
        F_g: Number of channels in the gating signal (from deeper level).
        F_l: Number of channels in the skip connection (from shallower level).
        F_int: Number of intermediate channels.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply attention gating to skip features.

        Args:
            g: Gating signal from deeper decoder level (coarser resolution).
            x: Skip connection features (finer resolution).

        Returns:
            Attention-weighted skip features with the same shape as x.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class EnhancedDecoder(nn.Module):
    """Streamlined U-shaped decoder with progressive feature fusion.

    Reconstructs segmentation maps by progressively combining deep semantic
    features with shallow spatial features via skip connections, bilinear
    upsampling, and convolutional refinement blocks.

    Args:
        num_in: Number of multi-scale input feature levels.
        out_ch: Number of output segmentation channels.
        channels: Channel dimensions for each feature level.
        dropout_p: Dropout probability in ConvBlock.
    """

    def __init__(
        self,
        num_in: int,
        out_ch: int = 1,
        channels: List[int] = [256, 256, 256, 256],
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.lat_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        reversed_channels = channels[::-1]

        for i in range(num_in):
            in_c = reversed_channels[i] if i == 0 else reversed_channels[i] * 2
            out_c = reversed_channels[i]
            self.lat_convs.append(nn.Conv2d(256, out_c, kernel_size=1))
            if i > 0:
                self.up_convs.append(ConvBlock(in_c, out_c, dropout_p=dropout_p))

        # Main segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(channels[0], 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, kernel_size=1),
        )

    def forward(
        self,
        feat_list: List[torch.Tensor],
        grid_hw: Tuple[int, int],
        patch_size: int,
        out_size_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Decode multi-scale features into a segmentation map.

        Args:
            feat_list: List of encoder feature maps (shallow to deep).
            grid_hw: Spatial grid dimensions (Gh, Gw) from encoder.
            patch_size: Encoder patch size for upsampling reference.
            out_size_hw: Target output spatial dimensions (H, W).

        Returns:
            A tuple of (main_logits, intermediate_features) where
            intermediate_features are the decoded features at each level
            (used for deep supervision auxiliary heads).
        """
        feats_reversed = feat_list[::-1]
        lat_feats = [conv(f) for conv, f in zip(self.lat_convs, feats_reversed)]

        intermediates = []
        p = lat_feats[0]
        intermediates.append(p)

        for i in range(1, len(lat_feats)):
            p = F.interpolate(
                p, size=lat_feats[i].shape[-2:], mode="bilinear", align_corners=False
            )
            p = torch.cat([p, lat_feats[i]], dim=1)
            p = self.up_convs[i - 1](p)
            intermediates.append(p)

        logits = self.seg_head(p)
        logits = F.interpolate(logits, size=out_size_hw, mode="bilinear", align_corners=False)
        return logits, intermediates


class AttentionDecoder(nn.Module):
    """Complex decoder variant with Attention Gates (for ablation study).

    Similar to EnhancedDecoder but applies Attention Gates to skip connections
    before concatenation, allowing the gating signal to suppress irrelevant
    spatial activations.

    Args:
        num_in: Number of multi-scale input feature levels.
        out_ch: Number of output segmentation channels.
        channels: Channel dimensions for each feature level.
        dropout_p: Dropout probability in ConvBlock.
    """

    def __init__(
        self,
        num_in: int,
        out_ch: int = 1,
        channels: List[int] = [256, 256, 256, 256],
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.lat_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.attn_gates = nn.ModuleList()

        reversed_channels = channels[::-1]
        for i in range(num_in):
            in_c = reversed_channels[i] if i == 0 else reversed_channels[i] * 2
            out_c = reversed_channels[i]
            self.lat_convs.append(nn.Conv2d(256, out_c, kernel_size=1))
            if i > 0:
                self.up_convs.append(ConvBlock(in_c, out_c, dropout_p=dropout_p))
                F_g = reversed_channels[i - 1]
                F_l = reversed_channels[i]
                F_int = F_l // 2
                self.attn_gates.append(AttentionGate(F_g, F_l, F_int))

        self.seg_head = nn.Sequential(
            nn.Conv2d(channels[0], 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, kernel_size=1),
        )

    def forward(
        self,
        feat_list: List[torch.Tensor],
        grid_hw: Tuple[int, int],
        patch_size: int,
        out_size_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Decode with attention-gated skip connections.

        Args:
            feat_list: List of encoder feature maps (shallow to deep).
            grid_hw: Spatial grid dimensions (Gh, Gw) from encoder.
            patch_size: Encoder patch size for upsampling reference.
            out_size_hw: Target output spatial dimensions (H, W).

        Returns:
            A tuple of (main_logits, intermediate_features).
        """
        feats_reversed = feat_list[::-1]
        lat_feats = [conv(f) for conv, f in zip(self.lat_convs, feats_reversed)]

        intermediates = []
        p = lat_feats[0]
        intermediates.append(p)

        for i in range(1, len(lat_feats)):
            skip = lat_feats[i]
            # Apply Attention Gate: gating signal p gates the skip features
            skip_att = self.attn_gates[i - 1](g=p, x=skip)
            p = F.interpolate(
                p, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            p = torch.cat([p, skip_att], dim=1)
            p = self.up_convs[i - 1](p)
            intermediates.append(p)

        logits = self.seg_head(p)
        logits = F.interpolate(logits, size=out_size_hw, mode="bilinear", align_corners=False)
        return logits, intermediates


class DinoV2UNet(nn.Module):
    """DINOv2-UNet: Full segmentation model with deep supervision.

    Combines a DINOv2 ViT encoder with a U-shaped decoder and optional
    deep supervision via auxiliary prediction heads attached to intermediate
    decoder features.

    Architecture (paper Section 3.2-3.5):
        Input -> DINOv2 ViT Encoder (partial fine-tuning)
              -> Feature Projection (1x1 conv per level)
              -> U-Shaped Decoder (progressive fusion)
              -> Main Head (segmentation map)
              -> Auxiliary Heads (deep supervision, training only)

    Args:
        backbone: Name of the timm ViT model.
        out_indices: Transformer block indices for multi-scale extraction.
        pretrained: Whether to load pretrained weights.
        freeze_blocks_until: Number of blocks to freeze from the start.
        num_classes: Number of output segmentation classes.
        decoder_dropout: Dropout probability in decoder ConvBlocks.
        pretrained_type: Pretrained weight type ('dinov2' or 'imagenet_supervised').
        decoder_type: Decoder variant ('simple' for Streamlined, 'complex' for Attention Gates).
        deep_supervision: Whether to enable deep supervision auxiliary heads.
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch14_dinov2",
        out_indices: Tuple[int, ...] = (2, 5, 8, 11),
        pretrained: bool = True,
        freeze_blocks_until: int = 6,
        num_classes: int = 1,
        decoder_dropout: float = 0.2,
        pretrained_type: str = "dinov2",
        decoder_type: str = "simple",
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.encoder = VitDinoV2Encoder(
            backbone, out_indices, pretrained, freeze_blocks_until,
            pretrained_type=pretrained_type,
        )

        if decoder_type == "complex":
            self.decoder = AttentionDecoder(
                num_in=len(out_indices), out_ch=num_classes, dropout_p=decoder_dropout,
            )
        else:
            self.decoder = EnhancedDecoder(
                num_in=len(out_indices), out_ch=num_classes, dropout_p=decoder_dropout,
            )

        # Deep supervision auxiliary heads (attached to intermediate decoder levels)
        # Paper Eq. 10: L_total = L_main + sum(lambda_i * L_aux_i)
        if deep_supervision:
            self.aux_heads = nn.ModuleList()
            for _ in range(len(out_indices) - 1):
                self.aux_heads.append(nn.Conv2d(256, num_classes, kernel_size=1))

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional deep supervision.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            A dict with key 'main' containing the primary logits (B, C, H, W).
            During training with deep_supervision=True, also contains 'aux_0',
            'aux_1', 'aux_2' for auxiliary head outputs.
        """
        out_h, out_w = x.shape[2], x.shape[3]
        feats, (Gh, Gw), p = self.encoder(x)
        main_logits, intermediates = self.decoder(
            feats, (Gh, Gw), p, out_size_hw=(out_h, out_w)
        )

        outputs: Dict[str, torch.Tensor] = {"main": main_logits}

        if self.deep_supervision and self.training:
            # Auxiliary heads on intermediate decoder features (P4, P3, P2)
            # intermediates[0] = deepest (P4), intermediates[-1] = shallowest (P1, main)
            for i, aux_head in enumerate(self.aux_heads):
                aux_feat = intermediates[i]  # P4, P3, P2 (deep to shallow)
                aux_logits = aux_head(aux_feat)
                aux_logits = F.interpolate(
                    aux_logits, size=(out_h, out_w),
                    mode="bilinear", align_corners=False,
                )
                outputs[f"aux_{i}"] = aux_logits

        return outputs
