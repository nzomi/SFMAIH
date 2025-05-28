import torch
import glob, os
import pydicom
import numpy as np
from icecream import ic

def load_dcm_as_tensor(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array
    img = img.astype(np.float32)
    img = torch.from_numpy(img)
    return img

def load_dcm_as_tensor_batch(dcm_paths):
    return torch.stack([load_dcm_as_tensor(dcm_path) for dcm_path in dcm_paths])

def load_dcm_as_tensor_batch_from_dir(dcm_dir):
    dcm_paths = glob.glob(os.path.join(dcm_dir, '*.dcm'))
    
    def sort_key(path):
        filename = os.path.basename(path)  
        parts = filename.split('.')  
        seq_num = int(parts[0][3:])  # Seq5 -> 5
        ser_num = int(parts[1][3:])  # Ser3 -> 3
        img_num = int(parts[2][3:])  # Img1 -> 1
        return (seq_num, ser_num, img_num)  

    dcm_paths.sort(key=sort_key)  
    ic(dcm_paths)
    return load_dcm_as_tensor_batch(dcm_paths)

if __name__ == '__main__':
    dcm_dir = '/home/liyuan.jiang/workspace/Seq5.Ser3'
    load_dcm_as_tensor_batch_from_dir(dcm_dir)
