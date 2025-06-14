import torch
import torch.nn as nn
from model.encoder import ENCODER_REGISTRY
from torchinfo import summary
from icecream import ic

class ModalityCompletion(nn.Module):
    def __init__(self, input_modalities, hidden_dim):
        super().__init__()
        self.input_modalities = input_modalities
        self.fuse = nn.Linear(len(input_modalities) * hidden_dim, hidden_dim)

    def forward(self, modal_feats: dict):
        existing_feats = []
        for m in self.input_modalities:
            feat = modal_feats.get(m, None)
            if feat is not None:
                existing_feats.append(feat)
        if not existing_feats:
            raise ValueError("No available modalities for completion.")

        template = existing_feats[0]
        for m in self.input_modalities:
            if modal_feats.get(m, None) is None:
                existing_feats.append(torch.zeros_like(template))
        fused = torch.cat(existing_feats, dim=-1)
        return self.fuse(fused)

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, modal_feats: list):  # list of [B, H]
        feats = torch.stack(modal_feats, dim=1)  # [B, M, H]
        weights = self.attn(feats).squeeze(-1)   # [B, M]
        weights = torch.softmax(weights, dim=1)
        fused = torch.sum(feats * weights.unsqueeze(-1), dim=1)
        return fused

@ENCODER_REGISTRY.register('multimodal')
class MultiModalNet(nn.Module):
    def __init__(self, hidden_dim, modalities, num_classes=3):
        super().__init__()
        self.modalities = modalities
        self.encoders = nn.ModuleDict({
            m: ENCODER_REGISTRY.get(m) for m in modalities
        })
        self.completion = nn.ModuleDict({
            m: ModalityCompletion([mm for mm in modalities if mm != m], hidden_dim)
            for m in modalities
        })
        self.fusion = AttentionFusion(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: dict):
        modal_feats = {}

        for m in self.modalities:
            x = inputs.get(m, None)
            if x is not None:
                modal_feats[m] = self.encoders[m](x)
        # ic(modal_feats['t1'].shape, modal_feats['t2'].shape, modal_feats['localizer'].shape)
        # for m in self.modalities:
        #     if m not in modal_feats:
        #         modal_feats[m] = self.completion[m](modal_feats)
        # ic(modal_feats['t1'].shape, modal_feats['t2'].shape, modal_feats['localizer'].shape)
        fused_feat = self.fusion(list(modal_feats.values()))
        # ic(fused_feat.shape)
        res = self.classifier(fused_feat)
        # ic(res.shape)
        return res
    

if __name__ == '__main__':
    net = MultiModalNet(hidden_dim=64, modalities=['t1', 't2', 'localizer'])

    localizer_data=({'Ser1a': torch.empty(size=(7, 1, 256, 256)), 'Ser1b': torch.empty(size=(7, 1, 256, 256)), 'Ser1c': torch.empty(size=(7, 1, 256, 256)), 'Stack': torch.empty(size=(7, 3, 256, 256))},
                {'Ser1a': torch.empty(size=(7, 6)), 'Ser1b': torch.empty(size=(7, 6)), 'Ser1c': torch.empty(size=(7, 6))})
    t1_data = [torch.empty(size=(20, 1, 320, 320)), torch.empty(size=(56, 1, 320, 320)), torch.empty(size=(42, 1, 320, 320))]
    t2_data = [torch.empty(size=(20, 1, 384, 384)), torch.empty(size=(56, 1, 384, 384)), torch.empty(size=(42, 1, 384, 384))]

    data = {'localizer': localizer_data, 't1': t1_data, 't2': t2_data}
    summary(net, input_data=(data,))
    # ic(net.encoders)
