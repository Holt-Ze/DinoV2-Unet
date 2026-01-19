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
                 pretrained=True, freeze_blocks_until=6, pretrained_type='dinov2'):
        super().__init__()
        # Select backbone based on pretrained_type if backbone is default
        if pretrained_type == 'imagenet_supervised':
            # Use supervision-trained ViT-Base. Note: DINOv2 is Patch 14, but standard supervised ViT is usually Patch 16.
            # We use Patch 16 here as Patch 14 supervised is not standard in timm.
            backbone = 'vit_base_patch16_224'
        elif pretrained_type == 'dinov2':
            # Default
            pass

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


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # g1 and x1 might have different sizes. The signals are combined.
        # Usually Attention Gate aligns them. In our decoder loop, g is upsampled BEFORE concatenation.
        # But Attention Gates often take the non-upsampled g?
        # Standard AG: g is coarse, x is fine. W_g(g) needs upsampling to match x?
        # Or standard behavior: g and x are passed, and we deal with size.
        # But here we are integrating into the existing structure.
        # Let's assume input g has been upsampled to match x's resolution OR we upsample inside.
        # Given the user instruction: "Before upsampling and concatenation", it implies g is still coarse?
        # "在上采样和拼接（Concatenation）之前，先让 Skip Feature 通过 Attention Gate"
        # If g is coarse, we need to upsample it inside AG or before.
        # If we look at the standard implementations, usually we do W_g(g) -> upsample -> + W_x(x).
        
        # Implementation:
        # g1 = self.W_g(g)
        # x1 = self.W_x(x)
        # resampled_g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear')
        # psi = relu(resampled_g1 + x1)
        
        # Let's check size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


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


class AttentionDecoder(nn.Module):
    def __init__(self, num_in: int, out_ch: int = 1, channels: List[int] = [256, 256, 256, 256],
                 dropout_p: float = 0.2):
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
                # Attention Gate
                # g: comes from previous layer (upsampled). Channels = out_c of previous layer (which is reversed_channels[i-1] => wait.
                # In standard UNet:
                # current level filters = F_l (skip)
                # gating signal = F_g (from below)
                
                # Here:
                # i=0: Bottom level. p = lat_convs[0](feat[-1]). channels = reversed_channels[0].
                # i=1:
                # p (gating) has channels = reversed_channels[0].
                # lat_feats[1] (skip) has channels = reversed_channels[1].
                # AG needs F_g=reversed_channels[i-1], F_l=reversed_channels[i], F_int=out_c//2
                
                F_g = reversed_channels[i-1]
                F_l = reversed_channels[i]
                F_int = F_l // 2
                self.attn_gates.append(AttentionGate(F_g, F_l, F_int))

        self.seg_head = nn.Sequential(
            nn.Conv2d(channels[0], 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, kernel_size=1)
        )

    def forward(self, feat_list: List[torch.Tensor], grid_hw: Tuple[int, int], patch_size: int, out_size_hw: Tuple[int, int]):
        feats_reversed = feat_list[::-1]
        lat_feats = [conv(f) for conv, f in zip(self.lat_convs, feats_reversed)]
        
        # Bottom-most feature
        p = lat_feats[0]
        
        for i in range(1, len(lat_feats)):
            # p is the gating signal from the coarser level
            skip = lat_feats[i]
            
            # Apply Attention Gate
            # Gate signal p is used to gate skip connection
            # NOTE: p is smaller than skip here. AG will handle upsampling internally if we implemented it that way.
            # My AG implementation does interpolate g to match x.
            
            # The user said: "Before upsampling and concatenation, let Skip Feature pass through Attention Gate."
            # So:
            skip_att = self.attn_gates[i-1](g=p, x=skip)
            
            # Upsample p
            p = F.interpolate(p, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            # Concatenate
            p = torch.cat([p, skip_att], dim=1)
            
            # Conv block
            p = self.up_convs[i - 1](p)
            
        p = self.seg_head(p)
        logits = F.interpolate(p, size=out_size_hw, mode='bilinear', align_corners=False)
        return logits


class DinoV2UNet(nn.Module):
    def __init__(self, backbone='vit_base_patch14_dinov2', out_indices=(2, 5, 8, 11),
                 pretrained=True, freeze_blocks_until=6, num_classes=1, decoder_dropout: float = 0.2,
                 pretrained_type: str = 'dinov2', decoder_type: str = 'simple'):
        super().__init__()
        self.encoder = VitDinoV2Encoder(backbone, out_indices, pretrained, freeze_blocks_until, pretrained_type=pretrained_type)
        if decoder_type == 'complex':
            self.decoder = AttentionDecoder(num_in=len(out_indices), out_ch=num_classes, dropout_p=decoder_dropout)
        else:
            self.decoder = EnhancedDecoder(num_in=len(out_indices), out_ch=num_classes, dropout_p=decoder_dropout)

    def forward(self, x):
        feats, (Gh, Gw), p = self.encoder(x)
        logits = self.decoder(feats, (Gh, Gw), p, out_size_hw=(x.shape[2], x.shape[3]))
        return logits

