import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from abc import ABC, abstractmethod
from icecream import ic
from datetime import datetime

import os

def recursive_to(obj, device):
    if isinstance(obj, dict):
        return {k: recursive_to(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(recursive_to(v, device) for v in obj)
    elif hasattr(obj, 'to'):
        return obj.to(device)
    else:
        return obj

def custom_collate(batch):

    return {
        'seq': batch[0]['seq'],
        'label': batch[0]['label'],
        'sub_seq': batch[0]['sub_seq'], 
    }

class BaseLogger(ABC):
    def __init__(self, config):
        self.config = config
        self.setup()
        pass

    def setup(self):
        log_dir = os.path.join('runs', self.config.model_name)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_graph(self, model, input_to_model):
        model.eval()
        assert input_to_model is not None, f"set input_to_model in {model.__class__.__name__}Trainer"
        with torch.no_grad():
            model(input_to_model)
        self.writer.add_graph(model, input_to_model)

    def log_image(self):
        pass

    def log_loss(self):
        pass

    def close(self):
        self.writer.close()

class BaseTrainer(ABC):
    def __init__(self, config, world_size=2, **kwargs):
        self.config = config
        self.logger = None
        self.input_to_model = None
        self.world_size = world_size
        self.save_path = self.config.save_path if self.config.save_path is not None else 'weights'
        pass

    def setup(self, model):
        self.optimizer = self.get_optimizer(model)
        self.loss_fn = self.get_loss_fn()

    def setup_distributed(self, rank, world_size):
        self.rank = rank
        self.local_rank = rank % torch.cuda.device_count()

        torch.cuda.set_device(self.local_rank)
        self.device = f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu'

        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:10086',
            world_size=world_size,
            rank=rank
        )

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def train(self, model, ds, epoch):
        model.train()
        self.process_whole(model, ds, epoch, valid=False)

    def valid(self, model, ds, epoch):
        model.eval()
        with torch.no_grad():
            self.process_whole(model, ds, epoch, valid=True)

    def process_whole(self, model, ds, epoch, valid):
        sampler = DistributedSampler(ds)
        dataloader = DataLoader(
            ds,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=custom_collate,
            num_workers=4,
            pin_memory=True
        )

        model = model.to(self.device)
        model = DDP(model, device_ids=[self.local_rank], find_unused_parameters=False)

        overall_loss = []

        sampler.set_epoch(epoch)
        res = self.process_epoch(model, dataloader, valid)
        overall_loss.append(res.get('loss'))

        dist.all_reduce(overall_loss[-1].clone().detach().to(self.device))
        mean_loss = sum(overall_loss)/len(overall_loss)

        if self.rank == 0:
            ic(epoch, mean_loss.item())

    @abstractmethod
    def process_epoch(self, model, dataloader, valid, *args, **kwargs):
        pass

    @abstractmethod
    def get_loss_fn(self):
        pass

    @abstractmethod
    def get_optimizer(self):
        pass

    def save_model(self, model):
        if self.rank == 0:
            now = datetime.now()
            time_str = now.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_path, f'{model.__class__.__name__}_{time_str}')
            torch.save(model.state_dict(), f'{save_path}.pth')       

    def load_model(self, model, load_path, device):
        state_dict = torch.load(load_path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict)
        return model

    def main(self, model, dataset, log_graph=False):
        self.move_to_device(model)
        self.print_arch(model)

        train_ds, val_ds = dataset

        if log_graph:
            self.logger.log_graph(model, self.input_to_model)

        self.setup(model)

        start_time = datetime.now()
        self.train_process(model, train_ds, val_ds, epochs=self.config.epochs, valid_freq=self.config.valid_freq)
        end_time = datetime.now()
        if self.rank == 0:
            training_time = end_time - start_time
            ic(str(training_time))

        if self.logger is not None:
            self.logger.close()       

    @abstractmethod
    def train_process(self, model, train_ds, val_ds, epochs, valid_freq):
        for epoch in range(epochs):
            self.train(model, train_ds, epoch)
            if (epoch+1) % valid_freq == 0:
                self.valid(model, val_ds, epoch)
                self.save_model(model)

    def move_to_device(self, model):
        model.to(self.device)       # in-place
        self.input_to_model = recursive_to(self.input_to_model, self.device)

    def print_arch(self, model):
        model.eval()
        if self.input_to_model is not None:
            with torch.no_grad():
                summary(model, input_data=(self.input_to_model,), device=self.device, depth=5)
        else:
            raise ValueError(f'check input_to_model: {self.input_to_model}')