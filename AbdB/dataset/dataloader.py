import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict
from util.func import load_dcm_as_tensor_batch_from_dir

import torch
import torch.nn.functional as F

import math

def excel2label_dict(excel_path, sheet_name, keys):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df[keys].dropna()
    return {int(row[keys[0]]): row[keys[1]] for _, row in df.iterrows()}

def collect_dataset(data_dir, label_dict):
    dataset = []
    for seq in sorted(os.listdir(data_dir)):
        seq_path = os.path.join(data_dir, seq)
        if not os.path.isdir(seq_path):
            continue
        label = label_dict.get(int(seq), None)
        if label is None:
            continue
        sub_seq_paths = {
            sub.lower(): os.path.join(seq_path, sub)
            for sub in os.listdir(seq_path)
            if os.path.isdir(os.path.join(seq_path, sub))
        }
        dataset.append({
            'seq': seq,
            'label': label,
            'sub_seq_paths': sub_seq_paths
        })
    return dataset

def resize_localizer_tensor(localizer_tensor, target_size=(256, 256)):
    # localizer_tensor: [B, C, H, W]
    if localizer_tensor.shape[-2:] != target_size:
        localizer_tensor = F.interpolate(
            localizer_tensor, size=target_size, mode='bilinear', align_corners=False
        )
    return localizer_tensor

class AbdSeqDataset(Dataset):
    def __init__(self, dataset_entries, preload=False, verbose=False):
        self.data = dataset_entries
        self.preload = preload
        self.verbose = verbose
        self.required_localizer_sers = ['Ser1a', 'Ser1b', 'Ser1c']

        if preload:
            for entry in self.data:
                for sub_name, sub_path in entry['sub_seq_paths'].items():
                    tensor_dict, orientation = load_dcm_as_tensor_batch_from_dir(sub_path)

                    if sub_name.lower().startswith("localizer"):
                        stacked_tensor = self._process_and_stack_localizer(tensor_dict)
                        entry['sub_seq_paths'][sub_name] = (stacked_tensor, orientation)
                    else:
                        entry['sub_seq_paths'][sub_name] = (tensor_dict, orientation)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        seq_id = entry['seq']
        label = entry['label']
        sub_data = {}

        for sub_name, sub_path in entry['sub_seq_paths'].items():
            if self.preload:
                sub_data[sub_name] = sub_path  # already processed
            else:
                tensor_dict, orientation = load_dcm_as_tensor_batch_from_dir(sub_path)
                if sub_name.lower().startswith("localizer"):
                    stacked_tensor = self._process_and_stack_localizer(tensor_dict)
                    if stacked_tensor is None:
                        continue
                    sub_data[sub_name] = (stacked_tensor, orientation)
                elif sub_name.lower().startswith("t1"):
                    stacked_tensor, stacked_orientation = self._process_and_stack_t1(tensor_dict, orientation)
                    if stacked_tensor is None:
                        continue
                    sub_data[sub_name] = (stacked_tensor, stacked_orientation)
                elif sub_name.lower().startswith("t2"):
                    stacked_tensor, stacked_orientation = self._process_and_stack_t2(tensor_dict, orientation)
                    if stacked_tensor is None:
                        continue
                    sub_data[sub_name] = (stacked_tensor, stacked_orientation)
                else:
                    sub_data[sub_name] = (tensor_dict, orientation)

            if self.verbose:
                print(f"[{seq_id}] Loaded {sub_name}")

        return {
            'seq': seq_id,
            'label': label,
            'sub_seq': sub_data
        }

    def _process_and_stack_localizer(self, tensor_dict):
        tensors = defaultdict()
        stack_tensors = []
        for ser_name in self.required_localizer_sers:
            t = tensor_dict.get(ser_name)
            if t is None:
                # raise ValueError(f"Missing localizer series: {ser_name}")
                return None
            if t.shape[-2:] != (256, 256):
                t = F.interpolate(t.float(), size=(256, 256), mode='bilinear', align_corners=False)
            tensors[ser_name] = t
            stack_tensors.append(t)
        tensors['Stack'] = torch.cat(stack_tensors, dim=1)
        return tensors
    
    def _process_and_stack_t1(self, tensor_dict, orientations):
        tensors = defaultdict()
        stack_tensors = []
        orientation = []
        for ser_name in tensor_dict.keys():
            t = tensor_dict.get(ser_name)
            if len(t) <= 1:
                continue
            if t is None:
                # raise ValueError(f"Missing localizer series: {ser_name}")
                return None
            if t.shape[-2:] != (320, 320):
                t = F.interpolate(t.float(), size=(320, 320), mode='bilinear', align_corners=False)
            tensors[ser_name] = t
            stack_tensors.append(t)
            orientation.append(orientations[ser_name])
        # tensors['Stack'] = torch.cat(stack_tensors, dim=1)
        return stack_tensors, orientation
    
    def _process_and_stack_t2(self, tensor_dict, orientations):
        tensors = defaultdict()
        stack_tensors = []
        orientation = []
        for ser_name in tensor_dict.keys():
            t = tensor_dict.get(ser_name)
            if len(t) <= 1:
                continue
            if t is None:
                # raise ValueError(f"Missing localizer series: {ser_name}")
                return None
            if t.shape[-2:] != (384, 384):
                t = F.interpolate(t.float(), size=(384, 384), mode='bilinear', align_corners=False)
            tensors[ser_name] = t
            stack_tensors.append(t)
            orientation.append(orientations[ser_name])
        # tensors['Stack'] = torch.cat(stack_tensors, dim=1)
        return stack_tensors, orientation

def custom_collate(batch):
    return {
        'seq': batch[0]['seq'],
        'label': batch[0]['label'],
        'sub_seq': batch[0]['sub_seq'], 
    }

def build_abd_dataloader(data_dir, excel_path, sheet_name='processed', keys=['num', 'label'],
                          batch_size=1, shuffle=True, num_workers=0,
                          preload=False, verbose=False):
    label_dict = excel2label_dict(excel_path, sheet_name, keys)
    dataset_entries = collect_dataset(data_dir, label_dict)
    dataset = AbdSeqDataset(dataset_entries, preload=preload, verbose=verbose)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
    #                         num_workers=num_workers, collate_fn=custom_collate)

    total_size = len(dataset)
    val_size = math.floor(0.1 * total_size)
    train_size = total_size - val_size

    torch.manual_seed(42)  
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=custom_collate)

    return train_loader, val_loader

def build_abd_dataset(data_dir, excel_path, sheet_name='processed', keys=['num', 'label'],
                          batch_size=1, shuffle=True, num_workers=0,
                          preload=False, verbose=False):
    label_dict = excel2label_dict(excel_path, sheet_name, keys)
    dataset_entries = collect_dataset(data_dir, label_dict)
    dataset = AbdSeqDataset(dataset_entries, preload=preload, verbose=verbose)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
    #                         num_workers=num_workers, collate_fn=custom_collate)

    total_size = len(dataset)
    val_size = math.floor(0.1 * total_size)
    train_size = total_size - val_size

    torch.manual_seed(42)  
    train_set, val_set = random_split(dataset, [train_size, val_size])

    return train_set, val_set

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('log_data.txt', mode='w')  
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

if __name__ == '__main__':
    excel_path = 'data/data.xlsx'
    data_path = 'data/seq'

    excel_path = '/home/liyuan.jiang/workspace/SFMAIH/AbdB/data/data.xlsx'
    data_path = '/home/liyuan.jiang/workspace/SFMAIH/AbdB/data/seq'
    
    train_ds, val_ds = build_abd_dataloader(
        data_dir=data_path,
        excel_path=excel_path,
        batch_size=1,
        shuffle=True,
        preload=False,  
        verbose=False     
    )

    for batch in train_ds:
        logger.info(f'Seq number: {batch["seq"]}')
        if 't1' not in batch['sub_seq'].keys():
            logger.info('miss t1')
        if 't2' not in batch['sub_seq'].keys():
            logger.info('miss t2')

        for sub_seq, seq_tensor in batch['sub_seq'].items():
            if sub_seq in ['t1']:# ['t1', 't2', 'localizer']:
                for ser_tensor in seq_tensor[0]:
                    logger.info(f'{sub_seq}, shape: {ser_tensor.shape} ')
            
            elif sub_seq in ['t2']:# ['t1', 't2', 'localizer']:
                for ser_tensor in seq_tensor[0]:
                    logger.info(f'{sub_seq}, shape: {ser_tensor.shape} ')
            
        logger.info('####################')


    # all_modalities = set()

    # for batch in dataloader:
    #     modalities = set(batch['sub_seq'].keys())
    #     all_modalities.update(modalities)

    # print("All observed modalities:", all_modalities)
        