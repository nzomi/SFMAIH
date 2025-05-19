import torch
import numpy as np
import random
import torch.multiprocessing as mp

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import argparse

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run(rank, world_size, model_name, data_name):
    config = None
    model = None
    dataset = None
    trainer = None

    trainer.setup_distributed(rank, world_size)
    trainer.main(model, dataset, log_graph=True)
    trainer.cleanup_distributed()
    

def train_distributed(world_size, model_name, dataset):
    mp.spawn(
        run,
        args=(world_size, model_name, dataset,),
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
    pass

if __name__=='__main__':
    set_seed()
    # main()
    debug()