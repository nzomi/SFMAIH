import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from model.base import ENCODER_REGISTRY
from einops.layers.torch import Rearrange
from einops import rearrange
from icecream import ic
import math

class RotateEncoder(nn.Module):
    def __init__(self, dim_in, dim_ff, dim_out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_out)
        )

    def forward(self, x):
        return self.layers(x)
    
class SliceEncoder(nn.Module):
    def __init__(self, dim_out=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(16, dim_out)

    def forward(self, x):
        feat = self.backbone(x).view(x.size(0), -1)
        res = self.fc(feat)
        # ic(feat.shape, res.shape)
        return res
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, N, D)
        # ic(x.shape)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SliceFuser(nn.Module):
    def __init__(self, dim_in=64, dim_hidden=256, heads=4):
        super().__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim_in, nhead=heads, dim_feedforward=dim_hidden)
        self.pos_embd = PositionalEncoding(d_model=dim_in)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=3)

    def forward(self, x):
        embed = self.pos_embd(x)
        # ic(embed.shape)
        out = self.transformer(x + embed)
        return out.mean(dim=0)

@ENCODER_REGISTRY.register('t1')
class T1Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice_encoder = SliceEncoder()
        self.slice_fuser = SliceFuser()

    def forward(self, sers):
        ser_feats = []
        for ser in sers[0]:
            slice_feats = self.slice_encoder(ser)
            ts_output = self.slice_fuser(slice_feats.unsqueeze(1))
            ser_feats.append(ts_output)
        res = torch.stack(ser_feats)
        return res.mean(dim=0)

@ENCODER_REGISTRY.register('t2')
class T2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice_encoder = SliceEncoder()
        self.slice_fuser = SliceFuser()

    def forward(self, sers):
        ser_feats = []
        for ser in sers[0]:
            slice_feats = self.slice_encoder(ser)
            ts_output = self.slice_fuser(slice_feats.unsqueeze(1))
            ser_feats.append(ts_output)
        res = torch.stack(ser_feats)
        return res.mean(dim=0)

@ENCODER_REGISTRY.register('localizer')
class LocalizerEncoder(nn.Module):
    def __init__(self, dim_in=3, dim_ff=16, dim_out=64, dropout=0.1, ser=['Ser1a', 'Ser1b', 'Ser1c']):
        super().__init__()

        self.enc = nn.ModuleList([
            LocalizerEncBlock(1, dim_out) for _ in range(3)
        ])

        self.roembed = nn.ModuleList([
            RotateEncoder(6, dim_ff, dim_out) for _ in range(3)      # (B, 5) --> broadcast (B, dim_out)
        ])

        self.enc_stack = LocalizerEncoder3D(dim_in)

        # self.enc_attn = AttentionFusion(dim_out)
        self.enc_attn = LightAttentionFusion()

        self.enc_fusion = nn.Conv2d(3+3, 1, 3, 1, 1)    # enc1/2/3 + stack

        self.dropout = nn.Dropout(dropout)

        self.ser = ser

        self.mlp = nn.Sequential(
            nn.Conv2d(1, 1, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.LazyLinear(64)
        )
    
    def forward(self, localizer_ser):
        encodings = []
        for i, ser_name in enumerate(self.ser):
            x = localizer_ser[0].get(ser_name)
            ro = localizer_ser[1].get(ser_name)
            enc = self.enc[i](x)
            ro_emb = self.roembed[i](ro)
            enc = enc + ro_emb.unsqueeze(-1).unsqueeze(-1)
            encodings.append(enc)
        
        enc_fused = self.enc_attn(*encodings)
        enc_stack = self.enc_stack(localizer_ser)    # [1, 3, 1, 64, 64]
        enc_stack = rearrange(enc_stack, 'b c d h w -> b (c d) h w')
        enc_stack = enc_stack.expand(enc_fused.size(0), -1, -1, -1)
        # ic(enc_fused.shape, enc_stack.shape)
        res = self.enc_fusion(torch.cat([enc_fused, enc_stack], dim=1))
        res = res.sum(dim=0, keepdim=True)
        # return self.dropout(res)
        res = self.mlp(self.dropout(res))
        return res


class LocalizerEncBlock(nn.Module):
    def __init__(self, dim_in=1, dim_out=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, dim_out, 3, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x 
    
class AttentionFusion(nn.Module):
    def __init__(self, dim=16*16):
        super().__init__()
        self.dim = dim
        self.q = nn.LazyLinear(dim)
        self.k = nn.LazyLinear(dim)
        self.v = nn.LazyLinear(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.upsamle = nn.Sequential(
            nn.ConvTranspose2d(3, 3, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(3, 3, 4, 2, 1)
        )

    def forward(self, x1, x2, x3):
        B, C, H, W = x1.shape
        x = torch.stack([x1, x2, x3], dim=1) # (7, 3, 64, 64, 64)
        x = x.view(B, 3, -1) # (7, 3, 256*256)
        q, k, v = self.q(x), self.k(x), self.v(x) # (7, 3, dim)
        attn = self.softmax(torch.bmm(q, k.transpose(1, 2)) / ((q.size(-1))**0.5))
        out = torch.bmm(attn, v)
        out = out.view(B, 3, 16, 16)
        return self.upsamle(out)
    
class LightAttentionFusion(nn.Module):
    def __init__(self, in_channels=64, reduced_channels=32, fusion_channels=3):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((8, 8))  

        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(reduced_channels, fusion_channels, kernel_size=1),
            nn.Softmax(dim=1) 
        )

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, fusion_channels, kernel_size=1) 
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.LeakyReLU(),
            nn.Conv2d(reduced_channels, fusion_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3): 
        x = torch.stack([x1, x2, x3], dim=1)   
        B, N, C, H, W = x.shape         # (7, 3, 64, 64, 64)
        x = x.view(B * N, C, H, W)      
        pooled = self.pool(x)           
        pooled_ori = pooled.view(B, N, C, 8, 8)   # (7, 3, 64, 8, 8)       
        pooled = pooled_ori.sum(dim=1)          # (7, 64, 8, 8) 
        attn_weights = self.attn(pooled)    # (7, 3, 8, 8)  
        fused = (attn_weights.unsqueeze(2) * pooled_ori).sum(dim=1)       # (7, 64, 64, 64)
        gate = self.gate(fused)
        out = self.upsample(fused)         
        return out * gate
    

@ENCODER_REGISTRY.register('localizer3d')
class LocalizerEncoder3D(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dropout=0.1):
        super().__init__()

        self.encoder3d = nn.Sequential(
            nn.Conv3d(dim_in, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),  # [B, 16, D/2, H/2, W/2]

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),  # [B, 32, D/4, H/4, W/4]

            nn.Conv3d(16, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm3d(dim_out),
            nn.LeakyReLU(),
        )

        self.dropout = nn.Dropout(dropout)

        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(),
        #     nn.Linear(dim_out, 4)
        # )

    def forward(self, localizer_ser):
        x = localizer_ser[0].get('Stack')   # [7, 3, 256, 256]
        # ser_tensors = [localizer_ser[0][k] for k in ['Ser1a', 'Ser1b', 'Ser1c']]
        # x = torch.stack(localizer_ser, dim=1)  # [7, 3, 1, 256, 256]
        x = rearrange(x, '(d b) c h w -> b c d h w', b=1)
        feat = self.encoder3d(x)  # [1, dim_out, D', H', W']
        feat = self.dropout(feat)
        return feat
        # out = self.head(feat)    # [1, 4]
        # ic(out.shape, feat.shape)
        # return out, feat

    
if __name__ == '__main__':
    # localizerencoder = LocalizerEncoder()
    # localizerencoder3d = LocalizerEncoder3D()
    # input_data=({'Ser1a': torch.empty(size=(7, 1, 256, 256)), 'Ser1b': torch.empty(size=(7, 1, 256, 256)), 'Ser1c': torch.empty(size=(7, 1, 256, 256)), 'Stack': torch.empty(size=(7, 3, 256, 256))},
    #             {'Ser1a': torch.empty(size=(7, 6)), 'Ser1b': torch.empty(size=(7, 6)), 'Ser1c': torch.empty(size=(7, 6))})
    # localiserencoder(input_data)
    # summary(localizerencoder, input_data=(input_data,))
    # summary(localizerencoder3d, input_data=(input_data,))

    t1encoder = T1Encoder()
    t1_data = [torch.empty(size=(20, 1, 320, 320)), torch.empty(size=(56, 1, 320, 320)), torch.empty(size=(42, 1, 320, 320))]
    summary(t1encoder, input_data=(t1_data,))
    t2encoder = T2Encoder()
    t2_data = [torch.empty(size=(20, 1, 384, 384)), torch.empty(size=(56, 1, 384, 384)), torch.empty(size=(42, 1, 384, 384))]
    summary(t2encoder, input_data=(t2_data,))