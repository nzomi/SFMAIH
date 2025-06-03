import os
import glob
import re
from collections import defaultdict
import numpy as np
import torch
import pydicom
import SimpleITK as sitk
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler('log.txt', mode='w')  
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def extract_number(s):
    return int(''.join(filter(str.isdigit, s)) or 0)

def extract_ser_name(dcm_path):
    filename = os.path.basename(dcm_path)
    match = re.search(r'(Ser[^\._]+)', filename)
    return match.group(1) if match else "UnknownSer"

def load_dcm_as_sitk_image(dcm_path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(dcm_path)
    return reader.Execute()


def normalize_image(image, hu_range=None):
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    if hu_range:
        min_val, max_val = hu_range
        array = np.clip((array - min_val) / (max_val - min_val), 0, 1)
    else:
        array -= array.min()
        array /= (array.max() + 1e-5)
    return torch.from_numpy(array)

def resample_image_to_reference(moving, reference, is_label=False):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference)
    resample.SetTransform(sitk.Euler3DTransform())
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resample.Execute(moving)

def choose_reference_series(sitk_images):
    spacing_dict = {
        ser: np.mean([np.mean(img.GetSpacing()) for img in imgs])
        for ser, imgs in sitk_images.items()
    }
    ref_ser = min(spacing_dict, key=spacing_dict.get)
    logger.info(f"Auto-selected reference series: {ref_ser} (mean spacing={spacing_dict[ref_ser]:.4f})")
    return ref_ser

def load_dcm_as_tensor_batch_by_ser(dcm_paths, reference_ser_name=None, hu_range=None):
    ser_groups = defaultdict(list)
    ser_orientations = defaultdict(list)
    sitk_images = defaultdict(list)

    for dcm_path in dcm_paths:
        ser_name = extract_ser_name(dcm_path)
        sitk_img = load_dcm_as_sitk_image(dcm_path)
        sitk_images[ser_name].append(sitk_img)

        dcm = pydicom.dcmread(dcm_path)
        orientation = dcm.ImageOrientationPatient
        ser_orientations[ser_name].append(orientation)

    if reference_ser_name is None:
        reference_ser_name = choose_reference_series(sitk_images)

    reference = sitk_images[reference_ser_name][0]
    stacked_tensors = {}
    ser_orientation_info = {}

    for ser_name, images in sitk_images.items():
        try:
            if ser_name != reference_ser_name:
                images = [resample_image_to_reference(img, reference) for img in images]
                logger.info(f"Resampled {ser_name} to match {reference_ser_name}")

            tensor_list = [normalize_image(img, hu_range=hu_range) for img in images]
            batch_tensor = torch.stack(tensor_list)
            stacked_tensors[ser_name] = batch_tensor
            ser_orientation_info[ser_name] = ser_orientations[ser_name]

            unique_orient = set(tuple(o) for o in ser_orientations[ser_name])
            logger.info(f"Ser {ser_name}: shape={batch_tensor.shape}, orientations={unique_orient}")

        except Exception as e:
            logger.error(f"Failed to process Ser {ser_name}: {e}")

    return stacked_tensors, ser_orientation_info

def load_dcm_as_tensor_batch_from_dir(dcm_dir, reference_ser_name=None, hu_range=None):
    dcm_paths = glob.glob(os.path.join(dcm_dir, '*.dcm'))

    def sort_key(path):
        filename = os.path.basename(path)
        parts = filename.split('.')
        seq_num = extract_number(parts[0][3:])  # Seq5 -> 5
        ser_num = extract_number(parts[1][3:])  # Ser3a -> 3
        img_num = extract_number(parts[2][3:])  # Img1 -> 1
        return (seq_num, ser_num, img_num)

    dcm_paths.sort(key=sort_key)
    return load_dcm_as_tensor_batch_by_ser(dcm_paths, reference_ser_name=reference_ser_name, hu_range=hu_range)


if __name__ == '__main__':
    dcm_dir = '/home/liyuan.jiang/workspace/Seq5.Ser3'
    load_dcm_as_tensor_batch_from_dir(dcm_dir)
