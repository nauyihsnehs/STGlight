import os, re, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from datasets.base_dataset import BaseDataModule
from datasets.dataset_registry import DatasetRegistry

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_PIZ]


def collate_pointcloud(batch, pad_value=0.0):
    Ns = [b['pos'].shape[0] for b in batch]
    Nmax = max(Ns)

    pos_b, x_b, y_b, mask_b, names = [], [], [], [], []

    for b in batch:
        pos = b['pos']  # [Ni,3]
        x = b['x']  # [Ni,3]
        y = b['y']  # [Ni,3]
        h = b['height']  # [Ni,1] or [Ni]
        name = b.get('name', '')

        if h.dim() == 1:
            h = h.unsqueeze(-1)

        x4 = torch.cat([x, h], dim=-1)

        Ni = pos.shape[0]
        pad_n = Nmax - Ni

        pos_p = torch.cat([pos, torch.full((pad_n, 3), pad_value, device=pos.device, dtype=pos.dtype)], dim=0)
        x_p = torch.cat([x4, torch.full((pad_n, 4), pad_value, device=x4.device, dtype=x4.dtype)], dim=0)  # [Nmax,4]
        y_p = torch.cat([y, torch.full((pad_n, 3), pad_value, device=y.device, dtype=y.dtype)], dim=0)  # [Nmax,3]

        m = torch.zeros(Nmax, device=pos.device, dtype=torch.bool)
        m[:Ni] = True

        pos_b.append(pos_p)  # [Nmax,3]
        x_b.append(x_p.transpose(0, 1))  # [4,Nmax]
        y_b.append(y_p.transpose(0, 1))  # [3,Nmax]
        mask_b.append(m)  # [Nmax]
        names.append(name)

    pos = torch.stack(pos_b, dim=0)  # [B,Nmax,3]
    x = torch.stack(x_b, dim=0)  # [B,4,Nmax]
    y = torch.stack(y_b, dim=0)  # [B,3,Nmax]
    mask = torch.stack(mask_b, dim=0)  # [B,Nmax]

    return {'pos': pos, 'x': x, 'y': y, 'mask': mask, 'names': names}


def normalize_pointcloud(pos):
    centroid = pos.mean(0, keepdim=True)  # [1,3]
    pos = pos - centroid
    scale = pos.norm(dim=1).max()
    pos = pos / (scale + 1e-6)
    return pos, pos[:, -1]


class PointNetDataset(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.data_root = str(os.path.join(root_dir, split))
        self.data_list = sorted(os.listdir(self.data_root))
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0

        self.sizes = []
        for fname in self.data_list:
            path = os.path.join(self.data_root, fname)
            cdata = np.load(path, allow_pickle=True).item()
            self.sizes.append(int(cdata['pos'].shape[0]))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data_path = os.path.join(self.data_root, self.data_list[data_idx])
        data_name = self.data_list[data_idx].split('.')[0]
        cdata = np.load(data_path, allow_pickle=True).item()

        pos = torch.from_numpy(cdata['pos']).float()  # [Ni,3]
        pos, height = normalize_pointcloud(pos)
        x = torch.from_numpy(cdata['x']).float()  # [Ni,3]
        y = torch.from_numpy(cdata['y']).float()  # [Ni,3]

        data = {'pos': pos, 'x': x, 'y': y, 'height': height, 'name': data_name}
        return data


@DatasetRegistry.register("pointnet")
class PointNetDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get("data_dir", 'data/infinigen')
        self.split = config.get("split", 'train')
        self.batch_size = config.get("batch_size", 8)

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            mtds = PointNetDataset(self.data_path, self.split)  # self.model_type, self.frames, 'train'
            self.train_dataset, self.val_dataset = random_split(mtds, [0.99, 0.01], torch.Generator().manual_seed(42))

        if stage == "test":
            mtds = PointNetDataset(self.data_path, self.split)
            self.test_dataset = mtds

        if stage == "predict":
            mtds = PointNetDataset(self.data_path, self.split)
            self.predict_dataset = mtds

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set up. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_pointcloud
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not set up. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_pointcloud
        )
