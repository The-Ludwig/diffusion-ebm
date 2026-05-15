"""
Micro-DiT model definitions for Project 2.

You will implement the ``AdaLayerNorm`` class as Task 1. Everything else in
this file is provided and does not need to be modified.
"""

import math

import torch
import torch.nn as nn

from torch.nn.utils.parametrizations import spectral_norm


IMG_SIZE = 32
PATCH_SIZE = 4
PAD_SIZE = 32
NUM_CLASSES = 10
EMBED_DIM = 256
DEPTH = 6
NUM_HEADS = 4


class PatchEmbed(nn.Module):
    """Split a (B, C, H, W) image into non-overlapping p x p patches, linearly
    project each flattened patch into an ``embed_dim``-dimensional token, and
    add a learnable positional embedding.
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.reshape(B, self.num_patches, self.patch_dim)
        x = self.proj(x)
        x = x + self.pos_embed
        return x


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm (Task 1).

    Given an input sequence ``x`` of shape (B, N, dim) and a conditioning
    vector ``cond`` of shape (B, cond_dim), compute::

        AdaLN(x, cond) = (1 + scale) * LN(x) + shift

    where ``[scale, shift] = W @ cond`` and ``LN`` is a LayerNorm with
    ``elementwise_affine=False`` (so all affine parameters come from
    ``cond``). ``scale`` and ``shift`` must broadcast over the token axis.
    """

    def __init__(self, dim, cond_dim):
        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2*dim)

    def forward(self, x, cond):
        scale_shift = self.proj(cond)

        # None inserts a size-one axis, so it matches the shape of x
        scale, shift = scale_shift[:,None,:self.dim], scale_shift[:,None,self.dim:]
        
        return (1+scale)*self.norm(x) + shift



class ETBlock(nn.Module):
    def __init__(self, dim, num_heads, cond_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = AdaLayerNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = AdaLayerNorm(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, cond):
        h = self.norm1(x, cond)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x, cond))
        return x


class MicroET(nn.Module):
    def __init__(
        self,
        img_size=PAD_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=1,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        mlp_ratio=4.0,
        device="cuda",
        max_batch_size=512
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.device = device
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.null_class_id = num_classes
        self.max_batch_size = max_batch_size
        self.register_buffer(
            "nocond_full",
            torch.full((max_batch_size,), self.null_class_id, dtype=torch.long),
        )

        self.blocks = nn.ModuleList(
            [ETBlock(embed_dim, num_heads, embed_dim, mlp_ratio) for _ in range(depth)]
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 1)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = self.img_size // p
        c = self.in_channels
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(-1, c, h * p, w * p)

    
    def accumulate(self, x):
        return x.mean(dim=1)

    def forward(self, x, c=None):
        if c is None:
            c = self.nocond_full[:x.shape[0]]

        x = self.patch_embed(x)
        cond = self.class_embed(c)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x)
        
        x = self.accumulate(x)
        
        return self.output_proj(x).squeeze()


def sn_conv(in_c, out_c, k=3, s=1, p=1):
    return spectral_norm(nn.Conv2d(in_c, out_c, k, s, p))


def sn_linear(in_f, out_f):
    return spectral_norm(nn.Linear(in_f, out_f))


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = sn_conv(in_c, out_c, 3, stride, 1)
        self.conv2 = sn_conv(out_c, out_c, 3, 1, 1)
        self.act   = nn.SiLU()
        # 1x1 conv on the skip path when shape changes
        self.shortcut = (
            sn_conv(in_c, out_c, 1, stride, 0)
            if (in_c != out_c or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(h + self.shortcut(x))


class EBM(nn.Module):
    """Small CNN energy: (B, in_channels, img_size, img_size) -> (B,) scalar energy."""
    def __init__(self, img_size=32, in_channels=1, ch=16, n_downsamples = 2, n_out = 1):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.n_out = n_out

        layers = [sn_conv(in_channels, ch, 3, 1, 1), nn.SiLU()]

        # Downsampling
        n_c = ch
        for i in range(n_downsamples):
            layers.append(ResBlock(n_c, n_c*2, stride=2))
            n_c *= 2

        # Linear output
        layers.append(nn.Flatten())

        final_size = img_size // (2 ** n_downsamples)

        layers.append(sn_linear(n_c * final_size * final_size, n_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def classify(self, x):
        assert self.n_out > 1, "Model does not output class logits"
        return self.forward(x)

    def energy(self, x):
        if self.n_out > 1:
            return torch.logsumexp(self.forward(x), dim=-1)
        else:
            return self.forward(x)
