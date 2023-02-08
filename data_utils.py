import os
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from skimage.transform import resize
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from monai.transforms import AsDiscrete

DATA_DIR = './data/brain_age'


def get_image_dataloaders(
    img_size: int,
    batch_size: int,
    num_workers: int = 0,
    low_data: str = None,
    segmentations: bool = False,
    folder: str = 'images'
):
    print('Loading data. This might take a while...')

    if segmentations:
        DATASET = BrainSegmentationDataset
    else:
        DATASET = BrainAgeImageDataset
    if low_data == '100':
      train_ds = DATASET('train_lowdata_100', img_size, folder)
    elif low_data == '50':
      train_ds = DATASET('train_lowdata_50', img_size, folder)
    elif low_data == '5':
      train_ds = DATASET('train_lowdata_5', img_size, folder)
    elif low_data == '3':
      train_ds = DATASET('train_lowdata_3', img_size, folder)
    else:
      train_ds = DATASET('train', img_size, folder)
    val_ds = DATASET('val', img_size, folder)
    test_ds = DATASET('test', img_size, folder)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_dl, 'val': val_dl, 'test': test_dl}

class BrainSegmentationDataset(Dataset):
    def __init__(self, mode: str, img_size: int, folder: str = 'images'):
        print(f'Loading {mode} data...')

        one_hot_transform = AsDiscrete(to_onehot=4)

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train.csv'))
        elif mode == 'train_lowdata_100':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train_lowdata.csv'))        
        elif mode == 'train_lowdata_50':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train_lowdata_50.csv'))       
        elif mode == 'train_lowdata_5':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train_lowdata_5.csv'))  
        elif mode == 'train_lowdata_3':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train_lowdata_3.csv'))        
        elif mode == 'val':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_segmentation_train.csv'))
        if mode == 'test':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_test.csv'))

        self.images = prefetch_samples(self.df['subject_id'].tolist(), img_size, folder=folder)
        self.segmentations = load_segmentations(self.df['subject_id'].tolist(), img_size)
        self.segmentations = [one_hot_transform(seg) for seg in self.segmentations]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get a brain MR-image and it's corresponding segmentation

        :param idx: Sample index of the dataset
        :return img: Loaded brain MRI. Shape (H, W, D)
        """
        img = self.images[idx]
        seg = self.segmentations[idx]
        return img, seg

class BrainAgeImageDataset(Dataset):
    def __init__(self, mode: str, img_size: int):
        assert mode in ['train', 'train_lowdata_50', 'train_lowdata_100', 'val', 'test']
        print(f'Loading {mode} data...')

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train.csv'))
        elif mode == 'train_lowdata_100':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train_lowdata.csv'))        
        elif mode == 'train_lowdata_50':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_train_lowdata_50.csv'))        
        elif mode == 'val':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_segmentation_train.csv'))
        if mode == 'test':
            self.df = pd.read_csv(os.path.join(
                DATA_DIR, 'meta', 'meta_data_regression_test.csv'))

        self.images = prefetch_samples(self.df['subject_id'].tolist(), img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Get a brain MR-image and it's corresponding age

        :param idx: Sample index of the dataset
        :return img: Loaded brain MRI. Shape (H, W, D)
        """
        # ----------------------- ADD YOUR CODE HERE --------------------------
        img = self.images[idx]
        age = self.df.iloc[[idx]]['age'].item()
        # ------------------------------- END ---------------------------------
        return img, age


def normalize(img: np.ndarray, mask: np.ndarray):
    """Normalize a brain MR-image to have zero mean and unit variance.
    For getting the normalization parameters, only the brain pixels should be
    used. All background pixels should be 0 in the end.

    :param img: Brain MR-image. Shape (H, W, D)
    :param mask: Mask to indicate which voxels correspond to the brain (1s) and
                 which are background (0s). Shape (H, W, D)
    :return normalized_img: Brain MR-image normalized to zero mean and unit
                            variance. Shape (H, W, D)
    """
    # ------------------------- ADD YOUR CODE HERE ----------------------------
    mean = img[mask > 0].mean()
    std = img[mask > 0].mean()

    normalized_img = (img - mean) / std
    normalized_img[mask == 0] = 0
    # --------------------------------- END -----------------------------------
    return normalized_img


def load_nii(path: str, dtype: str = 'float32') -> np.ndarray:
    """Load an MRI scan from disk and convert it to a given datatype

    :param path: Path to file
    :param dtype: Target dtype
    :return img: Loaded image. Shape (H, W, D)
    """
    # ------------------------- ADD YOUR CODE HERE ----------------------------
    img = nib.load(path).get_fdata().astype(np.dtype(dtype))
    # --------------------------------- END -----------------------------------
    return img


def preprocess(img: np.ndarray, mask: np.ndarray, img_size: int) -> np.ndarray:
    """Preprocess an MRI.

    :param img: MR-image. Shape (H, W, D)
    :param mask: Brain mask. Shape (H, W, D)
    :param img_size: Target size, a scalar.
    :return preprocessed_img: The preprocessed image.
                              Shape (img_size, img_size, img_size)
    """
    # Feel free to add more pre-processing here.
    # ------------------------- ADD YOUR CODE HERE ----------------------------
    preprocessed_img = normalize(img, mask)
    preprocessed_img = resize(preprocessed_img, [img_size] * 3)
    # --------------------------------- END -----------------------------------
    return preprocessed_img


def load_and_preprocess(ID: str, img_size: int, folder: str) -> np.ndarray:
    """Load an MRI from disk and preprocess it"""
    img = load_nii(os.path.join(DATA_DIR, f"{folder}/sub-{ID}_T1w_unbiased.nii.gz"))
    mask = load_nii(
        os.path.join(DATA_DIR, f"masks/sub-{ID}_T1w_brain_mask.nii.gz"),
        dtype='int'
    )
    return preprocess(img, mask, img_size)


def prefetch_samples(
    IDs: List[str],
    img_size: int,
    num_processes: int = 4,
    folder: str = 'images'
) -> np.ndarray:
    load_fn = partial(load_and_preprocess, img_size=img_size, folder=folder)
    # res = [load_fn(ID) for ID in IDs]
    # To speed up loading, comment the line above and uncomment the two below.
    with Pool(num_processes) as p:
        res = p.map(load_fn, IDs)
    return np.array(res)[:, None]


def load_and_preprocess_segs(ID: str, img_size: int) -> np.ndarray:
    """Load an MRI from disk and preprocess it"""
    img = load_nii(os.path.join(DATA_DIR, f"segs_refs/sub-{ID}_T1w_seg.nii.gz"))
    return resize(img, [img_size] * 3)

def load_segmentations(
    IDs: List[str],
    img_size: int,
    num_processes: int = 4):
    """Load all segmentations and associated subject_ids"""
    load_fn = partial(load_and_preprocess_segs, img_size=img_size)
    with Pool(num_processes) as p:
        res = p.map(load_fn, IDs)
    return np.array(res)[:, None]