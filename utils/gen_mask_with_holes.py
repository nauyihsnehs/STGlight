import os
import cv2
import numpy as np


def _draw_random_blob(h, w, rng):
    m = np.zeros((h, w), np.uint8)
    n = rng.integers(1, 5)
    for _ in range(n):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        ax = rng.integers(max(8, w // 6), max(12, w // 2))
        ay = rng.integers(max(8, h // 6), max(12, h // 2))
        ang = rng.uniform(0, 180)
        cv2.ellipse(m, (cx, cy), (ax, ay), ang, 0, 360, 255, -1)

    k_size = rng.integers(15, 42)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.GaussianBlur(m, (0, 0), rng.uniform(1.2, 3.2))
    return ((m > rng.integers(70, 141)).astype(np.uint8) * 255)


def _add_big_missing(m, rng):
    h, w = m.shape
    n = rng.integers(0, 4)
    for _ in range(n):
        pts = []
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        radx = rng.integers(max(8, w // 10), max(12, w // 3))
        rady = rng.integers(max(6, h // 10), max(10, h // 3))
        k = rng.integers(4, 10)
        for i in range(k):
            a = 2 * np.pi * i / k + rng.uniform(-0.3, 0.3)
            x = int(cx + radx * np.cos(a) * rng.uniform(0.6, 1.0))
            y = int(cy + rady * np.sin(a) * rng.uniform(0.6, 1.0))
            pts.append([np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)])
        cv2.fillPoly(m, [np.array(pts, np.int32)], 0)

    if rng.random() < 0.6:
        sz = rng.integers(3, 12)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
        m = cv2.erode(m, k) if rng.random() < 0.5 else cv2.dilate(m, k)
    return m


def _add_honeycomb_holes(m, rng):
    ys, xs = np.where(m > 0)
    if len(xs) == 0: return m

    n_holes = int(len(xs) * rng.uniform(0.00025, 0.0012))
    idx = rng.choice(len(xs), size=min(n_holes, len(xs)), replace=False)
    for i in idx:
        cv2.circle(m, (int(xs[i]), int(ys[i])), rng.integers(1, 4), 0, -1)

    if rng.random() < 0.5:
        sz = rng.integers(3, 8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
        m = cv2.erode(m, k) if rng.random() < 0.7 else cv2.dilate(m, k)
    return m


def soften_mask_distance(mask_u8, rng):
    m = (mask_u8 > 0).astype(np.uint8)
    d_in = cv2.distanceTransform(m, cv2.DIST_L2, 3)

    feather = rng.uniform(3.0, 14.0)
    alpha = np.clip(d_in / feather, 0.0, 1.0)

    band = (d_in < feather * 1.2) & (cv2.distanceTransform(1 - m, cv2.DIST_L2, 3) < feather * 1.2)
    alpha[band] = np.clip(alpha[band] + rng.normal(0, 0.02, size=alpha[band].shape), 0, 1)

    alpha = alpha ** rng.uniform(1.0, 1.8)
    alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(0.5, 1.6))
    return np.clip(alpha, 0, 1).astype(np.float32)


def soften_mask_downsample(mask_u8, rng):
    h, w = mask_u8.shape
    ds = rng.choice([2, 3, 4])
    m_small = cv2.resize(mask_u8, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA).astype(
        np.float32) / 255.0
    m_small = cv2.GaussianBlur(m_small, (0, 0), rng.uniform(0.2, 1.2))
    alpha = cv2.resize(m_small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0, 1).astype(np.float32)


def generate_pseudo_masks(n, h=128, w=256, seed=0, save_dir=None):
    rng = np.random.default_rng(seed)
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    hard_list, soft_list = [], []
    for i in range(n):
        m = _draw_random_blob(h, w, rng)
        m = _add_big_missing(m, rng)
        m = _add_honeycomb_holes(m, rng)
        if rng.random() < 0.75: m = _add_honeycomb_holes(m, rng)

        alpha = soften_mask_downsample(m, rng) if rng.random() < 0.25 else soften_mask_distance(m, rng)

        hard_list.append(m)
        soft_list.append(alpha)

        if save_dir:
            cv2.imwrite(f"{save_dir}/mask_soft_{i:06d}.png", (alpha * 255).astype(np.uint8))

    return np.array(hard_list), np.array(soft_list)


if __name__ == "__main__":
    hard, soft = generate_pseudo_masks(200, 128, 256, 42, "masks_out_256x128")
    print(f"Done: {hard.shape}, {soft.shape}")
