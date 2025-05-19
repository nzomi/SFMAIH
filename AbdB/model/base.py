import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

import inspect

from abc import ABC, abstractmethod

class HparamsBase:
    def save_hparams(self, ignore=('self', 'args', 'kwargs')):
        frame = inspect.currentframe().f_back
        _, _, _, local_vals = inspect.getargvalues(frame)
        all_params = {**local_vals, **local_vals.get('kwargs',{})}
        self.hparams = {k:v for k,v in all_params.items()
                        if k not in ignore and not k.startswith(_)}
        for k,v in self.hparams.items():
            setattr(self, k, v)

class BaseModel(nn.Module, ABC, HparamsBase):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self):
        pass