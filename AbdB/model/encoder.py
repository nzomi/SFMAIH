import torch.nn as nn
from model.base import ENCODER_REGISTRY

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
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass