"""
Micro-DiT model definitions for Project 2.

You will implement the ``AdaLayerNorm`` class as Task 1. Everything else in
this file is provided and does not need to be modified.
"""

import math

import torch
import torch.nn as nn

IMG_SIZE = 28
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
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )

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
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x, cond):
        proj = self.proj(cond)[:, None, :]
        scale, shift = torch.chunk(proj, 2, dim=-1)
        return (1 + scale) * self.norm(x) + shift


class DiTBlock(nn.Module):
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


class MicroDiT(nn.Module):
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
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.null_class_id = num_classes

        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads, embed_dim, mlp_ratio) for _ in range(depth)]
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

    def unpatchify(self, x):
        p = self.patch_size
        h = w = self.img_size // p
        c = self.in_channels
        x = x.reshape(-1, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(-1, c, h * p, w * p)

    def forward(self, x, t, c):
        x = self.patch_embed(x)
        cond = self.time_embed(t) + self.class_embed(c)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x)
        x = self.output_proj(x)
        return self.unpatchify(x)
