import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
        # Residual connection
        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, t):
        h = self.conv1(x)
        
        # Add time conditioning
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        
        h = self.conv2(h)
        
        # Residual connection
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        k = k.reshape(B, C, H * W)  # B, C, HW
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        
        # Attention
        scale = C ** -0.5
        attn = torch.softmax(torch.bmm(q, k) * scale, dim=-1)
        h = torch.bmm(attn, v)
        
        # Reshape back
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attention=False):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.attention = AttentionBlock(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
    
    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attention(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(in_ch * 2, out_ch, time_emb_dim)  # *2 for skip connection
        self.res2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.attention = AttentionBlock(out_ch) if use_attention else nn.Identity()
    
    def forward(self, x, skip, t):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attention(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=32, time_emb_dim=64):
        super().__init__()
        
        self.time_mlp = TimeEmbedding(time_emb_dim)
        
        # Initial projection
        self.init_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Simpler encoder - only 1 ResBlock per level
        self.down1 = nn.ModuleList([
            ResidualBlock(model_channels, model_channels, time_emb_dim),
            nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        ])
        
        self.down2 = nn.ModuleList([
            ResidualBlock(model_channels, model_channels * 2, time_emb_dim),
            nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        ])
        
        # Bottleneck with attention (smaller resolution, so attention is cheap)
        self.bottleneck = nn.ModuleList([
            ResidualBlock(model_channels * 2, model_channels * 2, time_emb_dim),
            AttentionBlock(model_channels * 2),
            ResidualBlock(model_channels * 2, model_channels * 2, time_emb_dim)
        ])
        
        # Decoder
        self.up1 = nn.ModuleList([
            nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1),
            ResidualBlock(model_channels * 4, model_channels, time_emb_dim)  # *4 for concat
        ])
        
        self.up2 = nn.ModuleList([
            nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1),
            ResidualBlock(model_channels * 2, model_channels, time_emb_dim)
        ])
        
        # Final projection
        self.final = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, in_channels, 3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x = self.init_conv(x)
        
        # Down 1
        skip1 = self.down1[0](x, t_emb)
        x = self.down1[1](skip1)
        
        # Down 2
        skip2 = self.down2[0](x, t_emb)
        x = self.down2[1](skip2)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        
        # Up 1
        x = self.up1[0](x)
        x = torch.cat([x, skip2], dim=1)
        x = self.up1[1](x, t_emb)
        
        # Up 2
        x = self.up2[0](x)
        x = torch.cat([x, skip1], dim=1)
        x = self.up2[1](x, t_emb)
        
        return self.final(x)
