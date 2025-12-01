import datetime
import os
import random
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import random_split, Dataset

from datasets.base_dataset import BaseDataModule
from datasets.dataset_registry import DatasetRegistry

exr_save_params = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF, cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_PIZ]


def max_pooling(img, size):
    H, W = img.shape[:2]
    w, h = size
    dh = H // h
    dw = W // w
    pad_h = abs(dh * h - H)
    pad_w = abs(dw * w - W)
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')  # 填充
    weights = np.array([0.0722, 0.7152, 0.2126])
    brightness = np.dot(img_padded[..., :3], weights)  # 计算亮度（加权和）
    pooled = np.zeros((h, w, img.shape[2]), dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            y_start, y_end = i * dh, (i + 1) * dh
            x_start, x_end = j * dw, (j + 1) * dw
            region_brightness = brightness[y_start:y_end, x_start:x_end]
            region_img = img_padded[y_start:y_end, x_start:x_end]
            mask = region_brightness > 1
            region_brightness_filtered = region_brightness[mask]
            region_img_filtered = region_img[mask]
            if region_brightness_filtered.size > 0:
                pooled[i, j] = np.mean(region_img_filtered, axis=0)
            else:
                max_idx = np.unravel_index(np.argmax(region_brightness), (dh, dw))
                pooled[i, j] = region_img[max_idx[0], max_idx[1]]
    return pooled


def ls_detector(image, threshold=250):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    pooled_image = max_pooling(image, (80, 60))
    weights = np.array([0.0722, 0.7152, 0.2126]).astype(np.float32)
    gray = (np.dot(pooled_image[..., :3], weights) * 255).astype(np.uint8)

    _, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
    _, _, _, centroids = cv.connectedComponentsWithStats(binary)

    if len(centroids) < 2:
        return None, None
    cen_x = centroids[1:, 0] / 40 - 1
    cen_y = 1 - centroids[1:, 1] / 30
    init_colors = []
    for centroid in centroids[1:]:
        centroid = np.floor(centroid).astype(np.uint8)
        color = pooled_image[centroid[1], centroid[0]]
        init_colors.append(color)
    init_poses = np.concatenate((cen_x[..., None], cen_y[..., None]), axis=-1).astype(np.float32)
    init_colors = np.array(init_colors)
    # visualize_image_coords(init_poses, init_colors, dh=480, dw=640, radius=10)
    return init_poses, init_colors


def visualize_image_coords(coords, colors=None, dh=None, dw=None, radius=8):
    coords_norm = (coords + 1) / 2.0
    pixel_x = coords_norm[:, 0] * (dw - 1)
    pixel_y = (1 - coords_norm[:, 1]) * (dh - 1)
    pixel_coords = np.stack([pixel_x, pixel_y], axis=1).astype(np.int32)
    colors = (colors * 255).astype(np.uint8)
    img = np.zeros((dh, dw, 3), dtype=np.uint8)
    for coord, color in zip(pixel_coords, colors):
        cv.circle(img, coord, radius, tuple((int(color[0]), int(color[1]), int(color[2]))), thickness=-1)
    img_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cv.imwrite(f'/mnt/data2/ssy/STGLight/test_img/outputs/vlsnet_tf/{img_name}_{img.max()}.png', img)
    img = (img / 255.0).astype(np.float32)
    return img


class VLSNetDataset(Dataset):
    def __init__(self, root_dir, res):
        self.root_dir = root_dir
        self.res = res
        self.rgb_fix = ['hdr', 'exr']
        self.input_fix = ['ldr', 'jpg']
        self.depth_fix = ['depth', 'exr']
        input_list = [p.as_posix() for p in Path(f'{root_dir}/{self.rgb_fix[0]}').glob('*.exr')]
        input_list.sort()
        self.input_list = input_list
        self.vls_list = []

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        rgb_path = self.input_list[idx]
        img_name = rgb_path.split('/')[-1].split('.')[0]
        hdr_img = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)
        hdr_img = cv.cvtColor(hdr_img, cv.COLOR_BGR2RGB)
        hdr_img = torch.from_numpy(hdr_img).float()
        hdr_img = torch.clamp_max(hdr_img, 400)
        lum = torch.mean(hdr_img, dim=-1, keepdim=True).repeat(1, 1, 3)
        hdr_img[lum < 1] = 0.
        if hdr_img.max() < 0.1:
            print(f'Warning: {img_name} max value is {hdr_img.max()}')
        ldr_img = hdr_img ** (1 / 2.2)  # Gamma correction
        ldr_img = torch.clamp(ldr_img, 0, 1)
        try:
            init_pos, init_color = ls_detector(ldr_img)
        except TypeError:
            print(f'Error in detecting light sources for image {img_name}, img_max: {hdr_img.max()}')
            init_pos, init_color = None, None
        if init_pos is not None:
            self.vls_list.append(rgb_path)
        else:
            assert len(self.vls_list) > 0, f"Image {img_name} has no light source, please check the data."
            rgb_path = random.choice(self.vls_list)
            img_name = rgb_path.split('/')[-1].split('.')[0]
            hdr_img = cv.imread(rgb_path, cv.IMREAD_UNCHANGED)
            hdr_img = cv.cvtColor(hdr_img, cv.COLOR_BGR2RGB)
            hdr_img = torch.from_numpy(hdr_img).float()
            hdr_img = torch.clamp_max(hdr_img, 400)
            lum = torch.mean(hdr_img, dim=-1, keepdim=True).repeat(1, 1, 3)
            hdr_img[lum < 1] = 0.
            ldr_img = hdr_img ** (1 / 2.2)  # Gamma correction
            ldr_img = torch.clamp(ldr_img, 0, 1)
            init_pos, init_color = ls_detector(ldr_img)
        init_pos = torch.from_numpy(init_pos).float()
        init_color = torch.from_numpy(init_color).float()

        hdr_img = torch.log1p(hdr_img).permute(2, 0, 1)

        input_path = f'{self.root_dir}/{self.input_fix[0]}/{img_name}.{self.input_fix[1]}'
        input_img = cv.imread(input_path)
        input_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB) / 255.
        input_img = torch.from_numpy(input_img).float().permute(2, 0, 1) * 2 - 1


        depth_path = f'{self.root_dir}/{self.depth_fix[0]}/{img_name}.{self.depth_fix[1]}'
        depth = cv.imread(depth_path, cv.IMREAD_UNCHANGED)
        depth = torch.from_numpy(depth).float()

        return hdr_img, init_pos, init_color, input_img, depth, img_name


@DatasetRegistry.register("vlsnet")
class VLSNetDataModule(BaseDataModule):
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get("data_dir", "datasets/vlsnet")
        self.resolution = config.get("resolution", (320, 240))
        self.batch_size = config.get("batch_size", 1)
        self.shuffle = False
        self.pin_memory = True
        self.save_hyperparameters()

    def setup(self, stage):
        if stage == "fit" or stage is None:
            mtds = VLSNetDataset(self.data_path, self.resolution)  # self.model_type, self.frames, 'train'
            self.train_dataset, self.val_dataset = random_split(mtds, [0.99, 0.01], torch.Generator().manual_seed(42))
