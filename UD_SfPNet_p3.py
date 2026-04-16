import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize


VALID_VARIANTS = (
    "baseline",
    "ablate_backbone_cnn",
    "ablate_backbone_cnn_transformer",
    "ablate_decoder_fpn",
    "ablate_decoder_unet",
)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = ConvBNAct(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MambaVisionMixer(nn.Module):
    """
    Vision-friendly Mamba mixer:
    1) Non-causal depth-wise conv in SSM branch.
    2) Symmetric non-SSM branch.
    3) Concatenate two branches then project.
    """

    def __init__(self, dim, d_state=16, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.dt_rank = max(4, math.ceil(dim / 16))

        self.in_proj = nn.Linear(dim, dim * 2)
        self.x_proj = nn.Linear(dim, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, dim)
        self.conv1d_x = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.conv1d_z = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.out_proj = nn.Linear(dim * 2, dim)

        a = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(dim, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(dim))

    def _selective_scan(self, x, dt, b_proj, c_proj):
        # x: (B, C, L), dt: (B, C, L), b_proj/c_proj: (B, d_state, L)
        batch, channels, seqlen = x.shape
        a = -torch.exp(self.A_log).unsqueeze(0)  # (1, C, d_state)
        d = self.D.view(1, channels)
        state = torch.zeros(batch, channels, self.d_state, device=x.device, dtype=x.dtype)
        y = []
        for t in range(seqlen):
            x_t = x[:, :, t]  # (B, C)
            dt_t = dt[:, :, t]  # (B, C)
            b_t = b_proj[:, :, t]  # (B, d_state)
            c_t = c_proj[:, :, t]  # (B, d_state)

            delta_a = torch.exp(dt_t.unsqueeze(-1) * a)  # (B, C, d_state)
            delta_b = dt_t.unsqueeze(-1) * b_t.unsqueeze(1)  # (B, C, d_state)
            state = delta_a * state + delta_b * x_t.unsqueeze(-1)

            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + d * x_t
            y.append(y_t)

        return torch.stack(y, dim=-1)

    def forward(self, hidden_states):
        # hidden_states: (B, L, C)
        batch, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)  # (B, 2C, L)
        x, z = xz.chunk(2, dim=1)  # (B, C, L), (B, C, L)

        x = F.silu(self.conv1d_x(x))
        z = F.silu(self.conv1d_z(z))

        x_flat = x.transpose(1, 2).reshape(batch * seqlen, self.dim)
        x_proj = self.x_proj(x_flat)
        dt, b_proj, c_proj = torch.split(x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = self.dt_proj(dt).reshape(batch, seqlen, self.dim).transpose(1, 2)
        dt = F.softplus(dt)
        b_proj = b_proj.reshape(batch, seqlen, self.d_state).transpose(1, 2)
        c_proj = c_proj.reshape(batch, seqlen, self.d_state).transpose(1, 2)

        x_ssm = self._selective_scan(x, dt, b_proj, c_proj)  # (B, C, L)

        out = torch.cat([x_ssm, z], dim=1).transpose(1, 2)  # (B, L, 2C)
        return self.out_proj(out)  # (B, L, C)


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = MambaVisionMixer(dim=dim, d_state=d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        self.attn_weights = attn_weights.detach()
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class HybridStage(nn.Module):
    def __init__(self, dim, mamba_depth, attn_depth, num_heads, d_state=16):
        super().__init__()
        self.mamba_blocks = nn.ModuleList(
            [MambaBlock(dim=dim, d_state=d_state) for _ in range(mamba_depth)]
        )
        self.attn_blocks = nn.ModuleList(
            [AttentionBlock(dim=dim, num_heads=num_heads) for _ in range(attn_depth)]
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.mamba_blocks:
            tokens = blk(tokens)
        for blk in self.attn_blocks:
            tokens = blk(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TransformerStage(nn.Module):
    def __init__(self, dim, depth, num_heads):
        super().__init__()
        self.blocks = nn.ModuleList([AttentionBlock(dim=dim, num_heads=num_heads) for _ in range(depth)])

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.blocks:
            tokens = blk(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class DecoderFusionBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNAct(in_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1),
            ConvBNAct(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class MambaVisionTinyBackbone(nn.Module):
    """
    MambaVision-T style hierarchy:
    Stage1/2: CNN residual blocks
    Stage3/4: Mamba + self-attention hybrid blocks
    """

    def __init__(
        self,
        in_ch=12,
        dims=(80, 160, 320, 640),
        conv_depths=(2, 2),
        mamba_depths=(2, 2),
        attn_depths=(2, 2),
        d_state=16,
    ):
        super().__init__()
        c1, c2, c3, c4 = dims

        self.stem = nn.Sequential(
            ConvBNAct(in_ch, c1 // 2, kernel_size=3, stride=2, padding=1),
            ConvBNAct(c1 // 2, c1, kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(*[ResidualConvBlock(c1) for _ in range(conv_depths[0])])
        self.down1 = ConvBNAct(c1, c2, kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(*[ResidualConvBlock(c2) for _ in range(conv_depths[1])])

        self.down2 = ConvBNAct(c2, c3, kernel_size=3, stride=2, padding=1)
        self.stage3 = HybridStage(
            dim=c3,
            mamba_depth=mamba_depths[0],
            attn_depth=attn_depths[0],
            num_heads=10,
            d_state=d_state,
        )

        self.down3 = ConvBNAct(c3, c4, kernel_size=3, stride=2, padding=1)
        self.stage4 = HybridStage(
            dim=c4,
            mamba_depth=mamba_depths[1],
            attn_depth=attn_depths[1],
            num_heads=10,
            d_state=d_state,
        )

    def forward(self, x):
        c1 = self.stem(x)  # 1/4
        c1 = self.stage1(c1)
        c2 = self.down1(c1)  # 1/8
        c2 = self.stage2(c2)
        c3 = self.down2(c2)  # 1/16
        c3 = self.stage3(c3)
        c4 = self.down3(c3)  # 1/32
        c4 = self.stage4(c4)
        return c1, c2, c3, c4


class CnnTinyBackbone(nn.Module):
    """
    Pure CNN ablation:
    Stage1~4 all use residual convolution blocks.
    """

    def __init__(self, in_ch=12, dims=(80, 160, 320, 640), stage_depths=(2, 2, 4, 4)):
        super().__init__()
        c1, c2, c3, c4 = dims

        self.stem = nn.Sequential(
            ConvBNAct(in_ch, c1 // 2, kernel_size=3, stride=2, padding=1),
            ConvBNAct(c1 // 2, c1, kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(*[ResidualConvBlock(c1) for _ in range(stage_depths[0])])
        self.down1 = ConvBNAct(c1, c2, kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(*[ResidualConvBlock(c2) for _ in range(stage_depths[1])])

        self.down2 = ConvBNAct(c2, c3, kernel_size=3, stride=2, padding=1)
        self.stage3 = nn.Sequential(*[ResidualConvBlock(c3) for _ in range(stage_depths[2])])

        self.down3 = ConvBNAct(c3, c4, kernel_size=3, stride=2, padding=1)
        self.stage4 = nn.Sequential(*[ResidualConvBlock(c4) for _ in range(stage_depths[3])])

    def forward(self, x):
        c1 = self.stem(x)  # 1/4
        c1 = self.stage1(c1)
        c2 = self.down1(c1)  # 1/8
        c2 = self.stage2(c2)
        c3 = self.down2(c2)  # 1/16
        c3 = self.stage3(c3)
        c4 = self.down3(c3)  # 1/32
        c4 = self.stage4(c4)
        return c1, c2, c3, c4


class CnnTransformerTinyBackbone(nn.Module):
    """
    CNN-Transformer Hybrid ablation:
    Stage1/2 are CNN, Stage3/4 are Transformer-only.
    """

    def __init__(
        self,
        in_ch=12,
        dims=(80, 160, 320, 640),
        conv_depths=(2, 2),
        attn_depths=(4, 4),
        num_heads=(10, 10),
    ):
        super().__init__()
        c1, c2, c3, c4 = dims

        self.stem = nn.Sequential(
            ConvBNAct(in_ch, c1 // 2, kernel_size=3, stride=2, padding=1),
            ConvBNAct(c1 // 2, c1, kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(*[ResidualConvBlock(c1) for _ in range(conv_depths[0])])
        self.down1 = ConvBNAct(c1, c2, kernel_size=3, stride=2, padding=1)
        self.stage2 = nn.Sequential(*[ResidualConvBlock(c2) for _ in range(conv_depths[1])])

        self.down2 = ConvBNAct(c2, c3, kernel_size=3, stride=2, padding=1)
        self.stage3 = TransformerStage(dim=c3, depth=attn_depths[0], num_heads=num_heads[0])

        self.down3 = ConvBNAct(c3, c4, kernel_size=3, stride=2, padding=1)
        self.stage4 = TransformerStage(dim=c4, depth=attn_depths[1], num_heads=num_heads[1])

    def forward(self, x):
        c1 = self.stem(x)  # 1/4
        c1 = self.stage1(c1)
        c2 = self.down1(c1)  # 1/8
        c2 = self.stage2(c2)
        c3 = self.down2(c2)  # 1/16
        c3 = self.stage3(c3)
        c4 = self.down3(c3)  # 1/32
        c4 = self.stage4(c4)
        return c1, c2, c3, c4


class FPNUNetDecoder(nn.Module):
    """
    Baseline decoder:
    FPN top-down fusion + UNet-style progressive fusion.
    """

    def __init__(self, in_dims=(80, 160, 320, 640), fpn_dim=192, out_ch=3):
        super().__init__()
        c1, c2, c3, c4 = in_dims
        self.lat1 = nn.Conv2d(c1, fpn_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(c2, fpn_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, fpn_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, fpn_dim, kernel_size=1)

        self.smooth1 = ConvBNAct(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = ConvBNAct(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = ConvBNAct(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.smooth4 = ConvBNAct(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)

        self.up43 = DecoderFusionBlock(fpn_dim, fpn_dim, fpn_dim)
        self.up32 = DecoderFusionBlock(fpn_dim, fpn_dim, fpn_dim)
        self.up21 = DecoderFusionBlock(fpn_dim, fpn_dim, fpn_dim)

        self.final_up = nn.Sequential(
            ConvBNAct(fpn_dim, fpn_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(fpn_dim // 2, fpn_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(fpn_dim // 4, fpn_dim // 8, kernel_size=3, stride=1, padding=1),
        )
        self.pred = nn.Conv2d(fpn_dim // 8, out_ch, kernel_size=1)

    def forward(self, feats, out_size):
        c1, c2, c3, c4 = feats
        p4 = self.smooth4(self.lat4(c4))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False))

        d3 = self.up43(p4, p3)
        d2 = self.up32(d3, p2)
        d1 = self.up21(d2, p1)

        out = self.final_up(d1)
        out = self.pred(out)
        out = normalize(out, p=2, dim=1)
        if out.shape[-2:] != out_size:
            out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
        return out


class PureFPNDecoder(nn.Module):
    """
    Ablation decoder:
    Pure FPN top-down fusion, without UNet progressive fusion blocks.
    """

    def __init__(self, in_dims=(80, 160, 320, 640), fpn_dim=192, out_ch=3):
        super().__init__()
        c1, c2, c3, c4 = in_dims
        self.lat1 = nn.Conv2d(c1, fpn_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(c2, fpn_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(c3, fpn_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, fpn_dim, kernel_size=1)

        self.smooth1 = ConvBNAct(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.smooth2 = ConvBNAct(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.smooth3 = ConvBNAct(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
        self.smooth4 = ConvBNAct(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)

        self.final_up = nn.Sequential(
            ConvBNAct(fpn_dim, fpn_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(fpn_dim // 2, fpn_dim // 4, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(fpn_dim // 4, fpn_dim // 8, kernel_size=3, stride=1, padding=1),
        )
        self.pred = nn.Conv2d(fpn_dim // 8, out_ch, kernel_size=1)

    def forward(self, feats, out_size):
        c1, c2, c3, c4 = feats
        p4 = self.smooth4(self.lat4(c4))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False))

        out = self.final_up(p1)
        out = self.pred(out)
        out = normalize(out, p=2, dim=1)
        if out.shape[-2:] != out_size:
            out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
        return out


class PureUNetDecoder(nn.Module):
    """
    Ablation decoder:
    Pure UNet-style progressive skip fusion, without FPN top-down lateral pathway.
    """

    def __init__(self, in_dims=(80, 160, 320, 640), out_ch=3):
        super().__init__()
        c1, c2, c3, c4 = in_dims

        self.up43 = DecoderFusionBlock(c4, c3, c3)
        self.up32 = DecoderFusionBlock(c3, c2, c2)
        self.up21 = DecoderFusionBlock(c2, c1, c1)

        mid1 = max(c1 // 2, 16)
        mid2 = max(c1 // 4, 8)
        mid3 = max(c1 // 8, 8)
        self.final_up = nn.Sequential(
            ConvBNAct(c1, mid1, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(mid1, mid2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(mid2, mid3, kernel_size=3, stride=1, padding=1),
        )
        self.pred = nn.Conv2d(mid3, out_ch, kernel_size=1)

    def forward(self, feats, out_size):
        c1, c2, c3, c4 = feats
        d3 = self.up43(c4, c3)
        d2 = self.up32(d3, c2)
        d1 = self.up21(d2, c1)

        out = self.final_up(d1)
        out = self.pred(out)
        out = normalize(out, p=2, dim=1)
        if out.shape[-2:] != out_size:
            out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=False)
        return out


class NetWork(nn.Module):
    """
    Unified model for baseline + ablation variants.

    Args:
        in_ch: input channels.
        out_ch: output channels (normal map xyz -> 3).
        variant: one of VALID_VARIANTS.
        dims: feature dimensions for 4 stages.
        fpn_dim: FPN channel width for FPN-based decoders.
        d_state: Mamba state dimension used by baseline backbone.
    """

    def __init__(
        self,
        in_ch=12,
        out_ch=3,
        variant="baseline",
        dims=(80, 160, 320, 640),
        fpn_dim=192,
        d_state=16,
    ):
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unsupported variant '{variant}'. Valid variants: {VALID_VARIANTS}")

        self.variant = variant
        self.dims = dims

        self.backbone = self._build_backbone(variant=variant, in_ch=in_ch, dims=dims, d_state=d_state)
        self.decoder = self._build_decoder(variant=variant, dims=dims, fpn_dim=fpn_dim, out_ch=out_ch)
        self._init_weights()

    @staticmethod
    def _build_backbone(variant: str, in_ch: int, dims: Tuple[int, int, int, int], d_state: int):
        if variant == "ablate_backbone_cnn":
            return CnnTinyBackbone(in_ch=in_ch, dims=dims, stage_depths=(2, 2, 4, 4))
        if variant == "ablate_backbone_cnn_transformer":
            return CnnTransformerTinyBackbone(
                in_ch=in_ch,
                dims=dims,
                conv_depths=(2, 2),
                attn_depths=(4, 4),
                num_heads=(10, 10),
            )
        return MambaVisionTinyBackbone(
            in_ch=in_ch,
            dims=dims,
            conv_depths=(2, 2),
            mamba_depths=(2, 2),
            attn_depths=(2, 2),
            d_state=d_state,
        )

    @staticmethod
    def _build_decoder(variant: str, dims: Tuple[int, int, int, int], fpn_dim: int, out_ch: int):
        if variant == "ablate_decoder_fpn":
            return PureFPNDecoder(in_dims=dims, fpn_dim=fpn_dim, out_ch=out_ch)
        if variant == "ablate_decoder_unet":
            return PureUNetDecoder(in_dims=dims, out_ch=out_ch)
        return FPNUNetDecoder(in_dims=dims, fpn_dim=fpn_dim, out_ch=out_ch)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.decoder(feats, out_size=x.shape[-2:])
        return out


def get_available_variants() -> Iterable[str]:
    return VALID_VARIANTS

