import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from util.func import load_dcm_as_tensor_batch_from_dir

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

class AbdSeqDataset(Dataset):
    def __init__(self, dataset_entries, preload=False, verbose=False):
        self.data = dataset_entries
        self.preload = preload
        self.verbose = verbose
        if preload:
            for entry in self.data:
                for sub_name, sub_path in entry['sub_seq_paths'].items():
                    entry['sub_seq_paths'][sub_name] = load_dcm_as_tensor_batch_from_dir(sub_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        seq_id = entry['seq']
        label = entry['label']
        sub_data = {}
        for sub_name, sub_path in entry['sub_seq_paths'].items():
            if self.preload:
                sub_data[sub_name] = sub_path  # already loaded tensors
            else:
                sub_data[sub_name] = load_dcm_as_tensor_batch_from_dir(sub_path)
            if self.verbose:
                print(f"[{seq_id}] Loaded {sub_name} from {sub_path}")

        return {
            'seq': seq_id,
            'label': label,
            'sub_seq': sub_data
        }

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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, collate_fn=custom_collate)
    return dataloader

if __name__ == '__main__':
    excel_path = 'data/data.xlsx'
    data_path = 'data/seq'

    excel_path = '/home/liyuan.jiang/workspace/SFMAIH/AbdB/data/data.xlsx'
    data_path = '/home/liyuan.jiang/workspace/SFMAIH/AbdB/data/seq'
    
    dataloader = build_abd_dataloader(
        data_dir=data_path,
        excel_path=excel_path,
        batch_size=1,
        shuffle=True,
        preload=False,  
        verbose=False     
    )

    # for batch in dataloader:
    #     if 't1' not in batch['sub_seq'].keys():
    #         print(batch['seq'], 'miss t1')
    #     if 't2' not in batch['sub_seq'].keys():
    #         print(batch['seq'], 'miss t2')
    #     if 'localizer' not in batch['sub_seq'].keys():
    #         print(batch['seq'], 'miss localizer')
    #     print('####################')


    all_modalities = set()

    for batch in dataloader:
        modalities = set(batch['sub_seq'].keys())
        all_modalities.update(modalities)

    print("All observed modalities:", all_modalities)
        