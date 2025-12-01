import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl

from datasets.base_dataset import BaseDataModule
from datasets.dataset_registry import DatasetRegistry
from datasets.nfnet_dataset import NFNetDataset
from datasets.nrnet_dataset import NRNetDataset


def _split_room_frame(stem):
    room, frame_str = stem.rsplit("_", 1)
    frame_id = int(frame_str)
    return room, frame_id


class NFNetSequenceDataset(Dataset):
    def __init__(self, base: NFNetDataset, max_gap=10):
        super().__init__()
        self.base = base
        self.max_gap = max_gap

        records = []
        for idx, path in enumerate(self.base.input_list):
            stem = os.path.basename(path).split(".")[0]
            room, frame = _split_room_frame(stem)
            records.append((room, frame, idx, stem))

        records.sort(key=lambda x: (x[0], x[1]))  # sort by (room, frame)

        self.pairs = []
        self.pair_names = []
        for i in range(len(records) - 1):
            room0, frame0, idx0, stem0 = records[i]
            room1, frame1, idx1, stem1 = records[i + 1]
            if room0 == room1 and (frame1 - frame0) <= self.max_gap:
                self.pairs.append((idx0, idx1))
                self.pair_names.append((stem0, stem1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx0, idx1 = self.pairs[idx]

        gt0, env0, vls0, sg0, mask0, name0 = self.base[idx0]
        gt1, env1, vls1, sg1, mask1, name1 = self.base[idx1]

        gt_seq = torch.stack([gt0, gt1], dim=0)
        env_seq = torch.stack([env0, env1], dim=0)
        vls_seq = torch.stack([vls0, vls1], dim=0)
        sg_seq = torch.stack([sg0, sg1], dim=0)
        mask_seq = torch.stack([mask0, mask1], dim=0)
        img_name_seq = [name0, name1]

        return gt_seq, env_seq, vls_seq, sg_seq, mask_seq, img_name_seq


@DatasetRegistry.register("nfnet_tf")
class NFNetDataModuleTF(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get("data_dir", "data/")
        self.resolution = (config.get("w", 256), config.get("h", 128))
        self.batch_size = config.get("batch_size", 1)
        self.shuffle = bool(config.get("shuffle", False))
        self.pin_memory = bool(config.get("pin_memory", False))
        self.max_gap = int(config.get("max_gap", 10))

        self.save_hyperparameters()

    def setup(self, stage=None):
        base = NFNetDataset(self.data_path, self.resolution)
        full = NFNetSequenceDataset(base, max_gap=self.max_gap)

        if stage == "fit" or stage is None:
            n = len(full)
            n_train = int(n * 0.9)
            n_val = n - n_train
            self.train_dataset, self.val_dataset = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))


class NRNetSequenceDataset(Dataset):
    def __init__(self, base: NRNetDataset, max_gap=10):
        super().__init__()
        self.base = base
        self.max_gap = max_gap

        records = []
        for idx, stem in enumerate(self.base.items):
            room, frame = _split_room_frame(stem)
            records.append((room, frame, idx, stem))

        records.sort(key=lambda x: (x[0], x[1]))

        self.pairs = []
        self.pair_names = []
        for i in range(len(records) - 1):
            room0, frame0, idx0, stem0 = records[i]
            room1, frame1, idx1, stem1 = records[i + 1]
            if room0 == room1 and (frame1 - frame0) <= self.max_gap:
                self.pairs.append((idx0, idx1))
                self.pair_names.append((stem0, stem1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx0, idx1 = self.pairs[idx]

        GT0, A0, G0, D0, mask_A0, mask_G0, stem0 = self.base[idx0]
        GT1, A1, G1, D1, mask_A1, mask_G1, stem1 = self.base[idx1]

        GT_seq = torch.stack([GT0, GT1], dim=0)
        A_seq = torch.stack([A0, A1], dim=0)
        G_seq = torch.stack([G0, G1], dim=0)
        D_seq = torch.stack([D0, D1], dim=0)
        mask_A_seq = torch.stack([mask_A0, mask_A1], dim=0)
        mask_G_seq = torch.stack([mask_G0, mask_G1], dim=0)
        stem_seq = [stem0, stem1]

        return GT_seq, A_seq, G_seq, D_seq, mask_A_seq, mask_G_seq, stem_seq


@DatasetRegistry.register("nrnet_tf")
class NRNetDataModuleTF(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_path = config.get("data_dir", "data/")
        self.resolution = (config.get("w", 256), config.get("h", 128))
        self.batch_size = config.get("batch_size", 1)
        self.num_workers = config.get("num_workers", 4)
        self.pin_memory = bool(config.get("pin_memory", False))
        self.shuffle = bool(config.get("shuffle", False))
        self.max_gap = int(config.get("max_gap", 10))

        self.save_hyperparameters()

    def setup(self, stage=None):
        base = NRNetDataset(self.data_path, self.resolution)
        full = NRNetSequenceDataset(base, max_gap=self.max_gap)

        if stage == "fit" or stage is None:
            n = len(full)
            n_train = int(n * 0.95)
            n_val = n - n_train
            self.train_dataset, self.val_dataset = random_split(
                full, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
