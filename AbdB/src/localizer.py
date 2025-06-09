import torch
import torch.nn as nn
import torch.nn.functional as F
from src.trainer import BaseTrainer, BaseLogger
from torchvision.utils import make_grid

def recursive_to(obj, device):
    if isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_to(v, device) for v in obj)
    elif hasattr(obj, 'to'):
        return obj.to(device)
    else:
        return obj

class ModelTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.logger = ModelLogger(config)
        localizer_data = ({'Ser1a': torch.empty(size=(7, 1, 256, 256)), 'Ser1b': torch.empty(size=(7, 1, 256, 256)), 'Ser1c': torch.empty(size=(7, 1, 256, 256)), 'Stack': torch.empty(size=(7, 3, 256, 256))},
                               {'Ser1a': torch.empty(size=(7, 6)), 'Ser1b': torch.empty(size=(7, 6)), 'Ser1c': torch.empty(size=(7, 6))})
        t1_data = [torch.empty(size=(20, 1, 320, 320)), torch.empty(size=(56, 1, 320, 320)), torch.empty(size=(42, 1, 320, 320))]
        t2_data = [torch.empty(size=(20, 1, 384, 384)), torch.empty(size=(56, 1, 384, 384)), torch.empty(size=(42, 1, 384, 384))]
        self.input_to_model = {'localizer': localizer_data, 't1': t1_data, 't2': t2_data}
    
        self.global_step = 0

    def train_process(self, model, train_ds, val_ds, epochs, valid_freq):
        return super().train_process(model, train_ds, val_ds, epochs, valid_freq)

    def process_epoch(self, model, dataloader, valid):
        is_training = model.training
        acc = []

        for batch in dataloader:
            batch = recursive_to(batch, model.device)
            if None in map(batch['sub_seq'].get, ['localizer', 't1', 't2']):
                continue
            label = torch.tensor(batch['label']).to(model.device)#.unsqueeze(0)

            if is_training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(is_training):
                logits = model(batch['sub_seq'])
                loss = self.loss_fn(logits, label.unsqueeze(0))

                if is_training:
                    loss.backward()
                    self.optimizer.step()
                    # self.logger.log_image(fused_res, self.global_step)
                    self.logger.log_loss(loss.item(), self.global_step)
                
            if valid:
                probs = F.softmax(logits, dim=1)
                predict = torch.argmax(probs, dim=1)
                acc.append((label==predict).sum().item()/len(predict))

            self.global_step += 1

        if valid:
            mean_acc = sum(acc)/len(acc)
        
        return {
            'loss': loss,
        }
    
    def get_loss_fn(self):
        return nn.CrossEntropyLoss()
    
    def get_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), self.config.lr)
    

class ModelLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)

    def log_loss(self, loss, step):
        self.writer.add_scalar(tag='train_loss', scalar_value=loss, global_step=step)

    def log_image(self, fused_res, step):
        fused_gray = fused_res.detach().cpu().reshape(-1, 1, 64, 64)

        fused_grid = make_grid(fused_gray, nrow=7, normalize=True)

        self.writer.add_image('fused_grid_channels', fused_grid, step)