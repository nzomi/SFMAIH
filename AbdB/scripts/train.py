import torch
import numpy as np
import random
import torch.multiprocessing as mp

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from config import BaseConfig
from model.encoder import *
from src.localizer import LocalizerTrainer
from dataset.dataloader import build_abd_dataloader, build_abd_dataset

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run(rank, world_size, model_name):
    excel_path = '/home/liyuan.jiang/workspace/SFMAIH/AbdB/data/data.xlsx'
    data_path = '/home/liyuan.jiang/workspace/SFMAIH/AbdB/data/seq'
    config = BaseConfig.load_config(model_name)
    model = LocalizerEncoder()
    dataset = build_abd_dataset(
        data_dir=data_path,
        excel_path=excel_path,
        batch_size=1,
        shuffle=True,
        preload=False,  
        verbose=False     
    )
    trainer = LocalizerTrainer(config)

    trainer.setup_distributed(rank, world_size)
    trainer.main(model, dataset, log_graph=False)
    trainer.cleanup_distributed()
    

def train_distributed(world_size, model_name):
    mp.spawn(
        run,
        args=(world_size, model_name,),
        nprocs=world_size,
        join=True
    )

def main():
    parser = argparse.ArgumentParser(description='Distributed Training Entry')
    parser.add_argument('--model-name', type=str, required=True, help='Model name to use')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use')
    parser.add_argument('--world-size', type=int, default=1, help='Number of processes')

    args = parser.parse_args()

    train_distributed(args.world_size, args.model_name, args.dataset)

def debug():
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    train_distributed(1, 'localizer')

if __name__=='__main__':
    set_seed()
    # main()
    debug()