import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from model.base import ENCODER_REGISTRY
from einops.layers.torch import Rearrange
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

@ENCODER_REGISTRY.register('t1')
class T1Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

@ENCODER_REGISTRY.register('t2')
class T2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass

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

        self.enc_stack = nn.Sequential(
            nn.Conv2d(dim_in, dim_ff, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(dim_ff, dim_in, 3, 2, 1)
        )

        # self.enc_attn = AttentionFusion(dim_out)
        self.enc_attn = LightAttentionFusion()

        self.enc_fusion = nn.Conv2d(3+3, 1, 3, 1, 1)    # enc1/2/3 + stack

        self.dropout = nn.Dropout(dropout)

        self.ser = ser

        self.mlp = nn.Sequential(
            nn.Conv2d(1, 1, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(4)
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
        enc_stack = self.enc_stack(localizer_ser[0].get('Stack'))

        res = self.enc_fusion(torch.cat([enc_fused, enc_stack], dim=1))
        return self.mlp(self.dropout(res))


class LocalizerEncBlock(nn.Module):
    def __init__(self, dim_in=1, dim_out=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim_in, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, dim_out, 3, padding=1),
            nn.ReLU(),
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
            nn.ReLU(),
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
            nn.ReLU(),
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
    
if __name__ == '__main__':
    localiserencoder = LocalizerEncoder()
    input_data=({'Ser1a': torch.empty(size=(7, 1, 256, 256)), 'Ser1b': torch.empty(size=(7, 1, 256, 256)), 'Ser1c': torch.empty(size=(7, 1, 256, 256)), 'Stack': torch.empty(size=(7, 3, 256, 256))},
                {'Ser1a': torch.empty(size=(7, 6)), 'Ser1b': torch.empty(size=(7, 6)), 'Ser1c': torch.empty(size=(7, 6))})
    # localiserencoder(input_data)
    summary(localiserencoder, input_data=(input_data,))