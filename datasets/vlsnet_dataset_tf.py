import os
import re
import sys
from pathlib import Path

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.utils.data import random_split, Dataset

from datasets.base_dataset import BaseDataModule
from datasets.dataset_registry import DatasetRegistry
from datasets.vlsnet_dataset import VLSNetDataset


class VLSNetTFDataset(Dataset):
    def __init__(
            self,
            root_dir,
            res,
            seq_len=2,
            max_frame_gap=10,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.res = res
        self.seq_len = seq_len
        self.max_frame_gap = max_frame_gap

        self.base_dataset = VLSNetDataset(root_dir, res)

        self._build_sequences()

    @staticmethod
    def _parse_room_and_frame(img_name):
        m = re.match(r"^(?P<room>.+)_(?P<frame>\d+)$", img_name)
        if m is None:
            return img_name, 0
        room_id = m.group("room")
        frame_id = int(m.group("frame"))
        return room_id, frame_id

    def _build_sequences(self) -> None:
        room_to_entries = {}

        for idx, rgb_path in enumerate(self.base_dataset.input_list):
            img_name = Path(rgb_path).stem
            room_id, frame_id = self._parse_room_and_frame(img_name)
            room_to_entries.setdefault(room_id, []).append((idx, frame_id, img_name))

        sequences = []

        for room_id, entries in room_to_entries.items():
            entries.sort(key=lambda x: x[1])  # (idx, frame_id, img_name)
            n = len(entries)
            for start in range(n):
                seq_idxs = [entries[start][0]]
                last_fid = entries[start][1]
                next_pos = start + 1
                while len(seq_idxs) < self.seq_len and next_pos < n:
                    idx_i, fid_i, _ = entries[next_pos]
                    if fid_i - last_fid > self.max_frame_gap:
                        break
                    seq_idxs.append(idx_i)
                    last_fid = fid_i
                    next_pos += 1
                if len(seq_idxs) >= 2:
                    sequences.append(seq_idxs)

        if len(sequences) == 0:
            raise RuntimeError(
                "VLSNetTFDataset: no sequences could be formed. "
                "Check that filenames follow {room_id}_{frame_id} and max_frame_gap is reasonable."
            )

        self.sequences = sequences
        print(
            f"[VLSNetTFDataset] Built {len(self.sequences)} sequences "
            f"from {len(self.base_dataset.input_list)} frames."
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq_indices = self.sequences[idx]

        hdr_imgs = []
        init_pos_list = []
        init_color_list = []
        input_imgs = []
        depths = []
        img_names = []

        for base_idx in seq_indices:
            hdr_img, init_pos, init_color, input_img, depth, img_name = self.base_dataset[base_idx]

            hdr_imgs.append(hdr_img)
            init_pos_list.append(init_pos)  # [N_t, 2]
            init_color_list.append(init_color)  # [N_t, 3]
            input_imgs.append(input_img)
            depths.append(depth)
            img_names.append(img_name)

        hdr_seq = torch.stack(hdr_imgs, dim=0)  # [T, 3, H, W]
        input_seq = torch.stack(input_imgs, dim=0)  # [T, 3, H, W]
        depth_seq = torch.stack(depths, dim=0)  # [T, H, W]

        return hdr_seq, init_pos_list, init_color_list, input_seq, depth_seq, img_names


@DatasetRegistry.register("vlsnet_tf")
class VLSNetTFDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get("data_dir", "datasets/vlsnet")
        self.resolution = config.get("resolution", (320, 240))
        self.batch_size = config.get("batch_size", 1)
        self.seq_len = config.get("seq_len", 2)
        self.max_frame_gap = config.get("max_frame_gap", 10)

        self.shuffle = False
        self.pin_memory = True
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_ds = VLSNetTFDataset(
                self.data_path,
                self.resolution,
                seq_len=self.seq_len,
                max_frame_gap=self.max_frame_gap,
            )
            self.train_dataset, self.val_dataset = random_split(
                full_ds,
                [0.99, 0.01],
                torch.Generator().manual_seed(42),
            )
