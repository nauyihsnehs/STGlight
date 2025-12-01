import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from datasets.base_dataset import BaseDataModule
from datasets.dataset_registry import DatasetRegistry
from datasets.pointnet_dataset import collate_pointcloud, normalize_pointcloud


def collate_temporal_pointcloud(batch, pad_value=0.0):
    B = len(batch)
    T = batch[0]["pos"].shape[0]
    Ns = [b["pos"].shape[1] for b in batch]
    Nmax = max(Ns)

    device = batch[0]["pos"].device
    dtype_pos = batch[0]["pos"].dtype
    dtype_x = batch[0]["x"].dtype
    dtype_y = batch[0]["y"].dtype

    pos_b = torch.full((B, T, Nmax, 3), pad_value, device=device, dtype=dtype_pos)
    x_b = torch.full((B, T, 4, Nmax), pad_value, device=device, dtype=dtype_x)
    y_b = torch.full((B, T, 3, Nmax), pad_value, device=device, dtype=dtype_y)
    mask_b = torch.zeros((B, T, Nmax), device=device, dtype=torch.bool)

    names = []
    room_ids = []
    frame_ids = []

    for i, b in enumerate(batch):
        Ni = b["pos"].shape[1]

        pos_b[i, :, :Ni, :] = b["pos"]  # [T, N_i, 3]
        x_b[i, :, :, :Ni] = b["x"]  # [T, 4, N_i]
        y_b[i, :, :, :Ni] = b["y"]  # [T, 3, N_i]
        mask_b[i, :, :Ni] = b["mask"]  # [T, N_i]

        names.append(b.get("names", []))
        room_ids.append(b.get("room_id", None))
        frame_ids.append(b.get("frame_ids", None))

    return {
        "pos": pos_b,  # [B, T, Nmax, 3]
        "x": x_b,  # [B, T, 4, Nmax]
        "y": y_b,  # [B, T, 3, Nmax]
        "mask": mask_b,  # [B, T, Nmax]
        "names": names,  # list[list[str]]
        "room_id": room_ids,
        "frame_ids": frame_ids,
    }


class PointNetTFDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split,
            seq_len=2,
            max_frame_gap=10,
            seq_stride=1,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.data_root = str(os.path.join(root_dir, split))
        self.seq_len = seq_len
        self.max_frame_gap = max_frame_gap
        self.seq_stride = seq_stride

        self.data_list = sorted(os.listdir(self.data_root))
        assert len(self.data_list) > 0, "No data files found"

        self.frames = []
        for fname in self.data_list:
            if not fname.endswith(".npy"):
                continue
            base = os.path.splitext(fname)[0]
            parts = base.split("_")

            if len(parts) < 4 or parts[0] != "infinigen":
                try:
                    frame_id = int(parts[-1])
                    room_id = "_".join(parts[:-1])
                except Exception:
                    continue
            else:
                try:
                    frame_id = int(parts[-2])
                except Exception:
                    continue
                room_id = "_".join(parts[1:-2])

            self.frames.append(
                {
                    "room_id": room_id,
                    "frame_id": frame_id,
                    "fname": fname,
                }
            )

        assert len(self.frames) > 0, "No valid frame files parsed"

        self.sequences = []
        self._build_sequences()

    def _build_sequences(self):
        by_room = {}
        for meta in self.frames:
            by_room.setdefault(meta["room_id"], []).append(meta)

        for room_id, frame_list in by_room.items():
            frame_list.sort(key=lambda m: m["frame_id"])

            current = []
            prev_fid = None

            def flush_segment(seg):
                if len(seg) < self.seq_len:
                    return
                for start in range(0, len(seg) - self.seq_len + 1, self.seq_stride):
                    self.sequences.append(seg[start: start + self.seq_len])

            for meta in frame_list:
                if prev_fid is None or (meta["frame_id"] - prev_fid) <= self.max_frame_gap:
                    current.append(meta)
                else:
                    flush_segment(current)
                    current = [meta]
                prev_fid = meta["frame_id"]

            flush_segment(current)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_meta = self.sequences[idx]
        frame_dicts = []

        for meta in seq_meta:
            fname = meta["fname"]
            data_path = os.path.join(self.data_root, fname)
            data_name = os.path.splitext(fname)[0]

            cdata = np.load(data_path, allow_pickle=True).item()
            pos = torch.from_numpy(cdata["pos"]).float()
            pos, height = normalize_pointcloud(pos)
            x = torch.from_numpy(cdata["x"]).float()
            y = torch.from_numpy(cdata["y"]).float()

            frame_dicts.append(
                {
                    "pos": pos,
                    "x": x,
                    "y": y,
                    "height": height,
                    "name": data_name,
                }
            )

        seq_batch = collate_pointcloud(frame_dicts)
        # Attach sequence-level meta information
        seq_batch["room_id"] = seq_meta[0]["room_id"]
        seq_batch["frame_ids"] = [m["frame_id"] for m in seq_meta]

        return seq_batch


@DatasetRegistry.register("pointnet_tf")
class PointNetTemporalDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get("data_dir", "data/infinigen")
        self.split = config.get("split", "train")
        self.batch_size = config.get("batch_size", 1)
        self.seq_len = config.get("seq_len", 2)
        self.max_frame_gap = config.get("max_frame_gap", 10)
        self.seq_stride = config.get("seq_stride", 1)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_ds = PointNetTFDataset(
                self.data_path,
                self.split,
                seq_len=self.seq_len,
                max_frame_gap=self.max_frame_gap,
                seq_stride=self.seq_stride,
            )
            self.train_dataset, self.val_dataset = random_split(
                full_ds,
                [max(1, int(0.99 * len(full_ds))), max(1, len(full_ds) - int(0.99 * len(full_ds)))],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset is not set up. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_temporal_pointcloud,
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
            collate_fn=collate_temporal_pointcloud,
        )
