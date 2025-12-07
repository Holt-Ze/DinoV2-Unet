from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as exc:
    raise RuntimeError("timm is required: install with `pip install timm`.") from exc


class VitDinoV2Encoder(nn.Module):
    def __init__(self, backbone='vit_base_patch14_dinov2', out_indices=(2, 5, 8, 11),
                 pretrained=True, freeze_blocks_until=6):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained, dynamic_img_size=True)
        if hasattr(self.model, 'patch_embed') and hasattr(self.model.patch_embed, 'dynamic_img_size'):
            self.model.patch_embed.dynamic_img_size = True
        self.out_indices = out_indices
        self.patch_size = self.model.patch_embed.patch_size[0] if isinstance(self.model.patch_embed.patch_size, tuple) else int(self.model.patch_embed.patch_size)
        self.embed_dim = self.model.embed_dim
        for i, blk in enumerate(self.model.blocks):
            requires = i >= freeze_blocks_until
            for p in blk.parameters():
                p.requires_grad = requires
        self.projs = nn.ModuleList([nn.Conv2d(self.embed_dim, 256, kernel_size=1) for _ in out_indices])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], Tuple[int, int], int]:
        B, C, H, W = x.shape
        x_pe = self.model.patch_embed(x)
        if hasattr(self.model, '_pos_embed'):
            pe_out = self.model._pos_embed(x_pe)
            if isinstance(pe_out, (list, tuple)):
                x_tokens, (Gh, Gw) = pe_out[0], pe_out[1]
            else:
                x_tokens, (Gh, Gw) = pe_out, (H // self.patch_size, W // self.patch_size)
        else:
            x_tokens, (Gh, Gw) = x_pe, (H // self.patch_size, W // self.patch_size)
            if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
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
                tok = tok[:, 1:, :]
            fm = tok.transpose(1, 2).reshape(B, self.embed_dim, Gh, Gw)
            fm = proj(fm)
            feats.append(fm)
        return feats, (Gh, Gw), self.patch_size


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EnhancedDecoder(nn.Module):
    def __init__(self, num_in: int, out_ch: int = 1, channels: List[int] = [256, 256, 256, 256],
                 dropout_p: float = 0.2):
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
        self.seg_head = nn.Sequential(
            nn.Conv2d(channels[0], 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, kernel_size=1)
        )

    def forward(self, feat_list: List[torch.Tensor], grid_hw: Tuple[int, int], patch_size: int, out_size_hw: Tuple[int, int]):
        feats_reversed = feat_list[::-1]
        lat_feats = [conv(f) for conv, f in zip(self.lat_convs, feats_reversed)]
        p = lat_feats[0]
        for i in range(1, len(lat_feats)):
            p = F.interpolate(p, size=lat_feats[i].shape[-2:], mode='bilinear', align_corners=False)
            p = torch.cat([p, lat_feats[i]], dim=1)
            p = self.up_convs[i - 1](p)
        p = self.seg_head(p)
        logits = F.interpolate(p, size=out_size_hw, mode='bilinear', align_corners=False)
        return logits


class DinoV2UNet(nn.Module):
    def __init__(self, backbone='vit_base_patch14_dinov2', out_indices=(2, 5, 8, 11),
                 pretrained=True, freeze_blocks_until=6, num_classes=1, decoder_dropout: float = 0.2):
        super().__init__()
        self.encoder = VitDinoV2Encoder(backbone, out_indices, pretrained, freeze_blocks_until)
        self.decoder = EnhancedDecoder(num_in=len(out_indices), out_ch=num_classes, dropout_p=decoder_dropout)

    def forward(self, x):
        feats, (Gh, Gw), p = self.encoder(x)
        logits = self.decoder(feats, (Gh, Gw), p, out_size_hw=(x.shape[2], x.shape[3]))
        return logits

