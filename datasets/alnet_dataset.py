import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path

import cv2 as cv
import torch
from torch.utils.data import random_split, Dataset

from datasets.base_dataset import BaseDataModule
from datasets.dataset_registry import DatasetRegistry

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_PIZ]


class ALNetDataset(Dataset):
    def __init__(self, root_dir, res):
        self.root_dir = root_dir
        self.res = res

        self.hdr_fix = ['hdr', 'exr']
        self.ldr_fix = ['ldr', 'jpg']
        self.mask_fix = ['mask', 'png']

        input_list = [p.as_posix() for p in Path(f'{root_dir}/{self.hdr_fix[0]}').glob('*.exr')]
        input_list.sort()
        self.input_list = input_list

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        hdr_path = self.input_list[idx]
        img_name = hdr_path.split('/')[-1].split('.')[0]

        hdr_pano = cv.imread(hdr_path, cv.IMREAD_UNCHANGED)  # H,W,3 (float32 HDR)
        hdr_pano = cv.GaussianBlur(hdr_pano, (5, 5), 5)
        hdr_pano = torch.from_numpy(hdr_pano).float().permute(2, 0, 1)  # [3,H,W], linear

        ldr_path = f'{self.root_dir}/{self.ldr_fix[0]}/{img_name}.{self.ldr_fix[1]}'
        ldr_pano = cv.imread(ldr_path) / 255.0  # H,W,3
        ldr_pano = torch.from_numpy(ldr_pano).float().permute(2, 0, 1)  # [3,H,W]

        partial_mask_path = f'{self.root_dir}/{self.mask_fix[0]}/{img_name}_mask.{self.mask_fix[1]}'
        partial_mask = cv.imread(partial_mask_path, cv.IMREAD_UNCHANGED)[..., None] / 255.0  # H,W,1
        partial_mask = torch.from_numpy(partial_mask).float().permute(2, 0, 1)  # [1,H,W]

        return hdr_pano, ldr_pano, partial_mask, img_name


@DatasetRegistry.register("alnet")
class ALNetDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.predict_data, self.test_data, self.val_data, self.train_data = None, None, None, None
        self.data_path = config.get("data_dir", "data/")
        self.resolution = config.get("resolution", (128, 256))
        self.batch_size = config.get("batch_size", 1)
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mtds = ALNetDataset(self.data_path, self.resolution)
            self.train_dataset, self.val_dataset = random_split(mtds, [0.95, 0.05], torch.Generator().manual_seed(42))

        if stage == "test":
            mtds = ALNetDataset(self.data_path, self.resolution)
            self.test_dataset = mtds

        if stage == "predict":
            mtds = ALNetDataset(self.data_path, self.resolution)
            self.predict_dataset = mtds


class ALNetTFDataset(Dataset):
    def __init__(
            self,
            root_dir,
            res,
            seq_len=2,
            max_frame_gap=10,
    ):
        self.root_dir = root_dir
        self.res = res
        self.seq_len = seq_len
        self.max_frame_gap = max_frame_gap

        self.hdr_fix = ["hdr", "exr"]
        self.ldr_fix = ["ldr", "jpg"]
        self.mask_fix = ["mask", "png"]

        hdr_dir = Path(f"{root_dir}/{self.hdr_fix[0]}")
        hdr_paths = [p for p in hdr_dir.glob("*.exr")]
        hdr_paths.sort()

        groups = {}
        for p in hdr_paths:
            name = p.stem  # {room_id}_{frame_id}
            parts = name.split("_")
            if len(parts) < 2:
                room_id = name
                frame_id = 0
            else:
                room_id = "_".join(parts[:-1])
                try:
                    frame_id = int(parts[-1])
                except ValueError:
                    room_id = name
                    frame_id = 0
            groups.setdefault(room_id, []).append((frame_id, p.as_posix()))

        self.sequences = []
        for room_id, frames in groups.items():
            frames.sort(key=lambda x: x[0])  # sort by frame_id
            n = len(frames)
            if n < self.seq_len:
                continue

            for start in range(0, n - self.seq_len + 1):
                end = start + self.seq_len
                window = frames[start:end]

                ok = True
                for i in range(len(window) - 1):
                    if window[i + 1][0] - window[i][0] > self.max_frame_gap:
                        ok = False
                        break
                if not ok:
                    continue

                seq_paths = [f[1] for f in window]
                self.sequences.append(seq_paths)

    def __len__(self):
        return len(self.sequences)

    def _load_single(self, hdr_path):
        img_name = Path(hdr_path).stem

        hdr_pano = cv.imread(hdr_path, cv.IMREAD_UNCHANGED)  # H,W,3
        hdr_pano = cv.GaussianBlur(hdr_pano, (5, 5), 5)
        hdr_pano = torch.from_numpy(hdr_pano).float().permute(2, 0, 1)  # [3,H,W]
        hdr_pano = torch.log1p(hdr_pano.clamp_min(0.0)).clamp_max(5.0)

        ldr_path = f"{self.root_dir}/{self.ldr_fix[0]}/{img_name}.{self.ldr_fix[1]}"
        ldr_pano = cv.imread(ldr_path) / 255.0  # H,W,3
        ldr_pano = torch.from_numpy(ldr_pano).float().permute(2, 0, 1)  # [3,H,W]

        partial_mask_path = (f"{self.root_dir}/{self.mask_fix[0]}/{img_name}_mask.{self.mask_fix[1]}")
        partial_mask = cv.imread(partial_mask_path, cv.IMREAD_UNCHANGED)[..., None] / 255.0
        partial_mask = torch.from_numpy(partial_mask).float().permute(2, 0, 1)  # [1,H,W]
        ldr_pano = ldr_pano * partial_mask

        return hdr_pano, ldr_pano, img_name

    def __getitem__(self, idx):
        seq_paths = self.sequences[idx]

        hdr_seq = []
        ldr_seq = []
        name_seq = []

        for hdr_path in seq_paths:
            hdr_pano, ldr_pano, img_name = self._load_single(hdr_path)
            hdr_seq.append(hdr_pano)
            ldr_seq.append(ldr_pano)
            name_seq.append(img_name)

        # [T,C,H,W]
        hdr_seq = torch.stack(hdr_seq, dim=0)
        ldr_seq = torch.stack(ldr_seq, dim=0)

        return hdr_seq, ldr_seq, name_seq


@DatasetRegistry.register("alnet_tf")
class ALNetTFDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.predict_data, self.test_data, self.val_data, self.train_data = (None, None, None, None)
        self.data_path = config.get("data_dir", "data/")
        self.resolution = config.get("resolution", (128, 256))
        self.batch_size = config.get("batch_size", 1)

        self.frames = config.get("frames", 2)  # sequence length
        self.max_frame_gap = config.get("max_frame_gap", 10)

        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_ds = ALNetTFDataset(
                self.data_path,
                self.resolution,
                seq_len=self.frames,
                max_frame_gap=self.max_frame_gap,
            )
            train_len = int(0.95 * len(full_ds))
            val_len = len(full_ds) - train_len
            self.train_dataset, self.val_dataset = random_split(
                full_ds, [train_len, val_len], generator=torch.Generator().manual_seed(42)
            )

        if stage == "test":
            self.test_dataset = ALNetTFDataset(
                self.data_path,
                self.resolution,
                seq_len=self.frames,
                max_frame_gap=self.max_frame_gap,
            )

        if stage == "predict":
            self.predict_dataset = ALNetTFDataset(
                self.data_path,
                self.resolution,
                seq_len=self.frames,
                max_frame_gap=self.max_frame_gap,
            )
