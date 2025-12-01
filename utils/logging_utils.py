import os

os.environ.setdefault("O3D_RENDERING_BACKEND", "egl")
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec

import open3d as o3d

_O3D_RENDERER = None


def setup_logging(config, name=None):
    if name is None:
        name = config.get('experiment', {}).get('name', 'default')

    log_level = config.get('logging', {}).get('level', 'INFO')
    log_dir = config.get('logging', {}).get('dir', 'logs')

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class MetricLogger:
    def __init__(self, logger):
        self.logger = logger
        self.metrics = {}

    def update(self, metrics):
        for k, v in metrics.items():
            if k in self.metrics:
                self.metrics[k].append(v)
            else:
                self.metrics[k] = [v]

    def log_metrics(self, step=None, prefix=''):
        log_str = f"Step {step}: " if step is not None else ""
        for k, v in self.metrics.items():
            if isinstance(v[-1], torch.Tensor):
                value = v[-1].item()
            else:
                value = v[-1]
            metric_name = f"{prefix}{k}" if prefix else k
            log_str += f"{metric_name}: {value:.4f}, "

        self.logger.info(log_str[:-2])

    def reset(self):
        self.metrics = {}

    def get_latest(self, metric_name):
        if metric_name not in self.metrics:
            return None

        value = self.metrics[metric_name][-1]
        if isinstance(value, torch.Tensor):
            return value.item()
        return value


def ply_logger(save_path, pos, color, binary=False):
    num_points = pos.shape[0]

    if binary:
        with open(save_path, 'wb') as f:
            header = [
                b'ply\n',
                b'format binary_little_endian 1.0\n',
                f'element vertex {num_points}\n'.encode(),
                b'property float x\n',
                b'property float y\n',
                b'property float z\n',
                b'property uchar red\n',
                b'property uchar green\n',
                b'property uchar blue\n',
                b'end_header\n'
            ]
            for line in header:
                f.write(line)

            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                     ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            data = np.empty(num_points, dtype=dtype)

            data['x'] = pos[:, 0]
            data['y'] = pos[:, 1]
            data['z'] = pos[:, 2]
            data['red'] = color[:, 0]
            data['green'] = color[:, 1]
            data['blue'] = color[:, 2]

            data.tofile(f)
            f.flush()
    else:
        with open(save_path, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {num_points}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            for i in range(num_points):
                f.write(f'{pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f} '
                        f'{int(color[i, 0])} {int(color[i, 1])} {int(color[i, 2])}\n')
            f.flush()


def visualize_point_cloud(pos, color, ax, elev=60, azim=0, roll=90,
                          point_size=8, bg_color='white'):
    ax.set_facecolor(bg_color)
    color = color / 255.0
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
               c=color, s=point_size, marker='.')
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_axis_off()
    ax.autoscale()
    max_range = np.array([
        pos[:, 0].max() - pos[:, 0].min(),
        pos[:, 1].max() - pos[:, 1].min(),
        pos[:, 2].max() - pos[:, 2].min()
    ]).max() / 2.0
    mid_x = (pos[:, 0].max() + pos[:, 0].min()) * 0.5
    mid_y = (pos[:, 1].max() + pos[:, 1].min()) * 0.5
    mid_z = (pos[:, 2].max() + pos[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_point_cloud_comparison(save_path, pos, input_color, pred_color, gt_color,
                                     dpi=300, elev=60, azim=0, roll=90,
                                     figsize=(24, 8), point_size=10, bg_color='white'):
    fig = plt.figure(figsize=figsize, facecolor=bg_color)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    titles = ['Input', 'Prediction', 'Ground Truth']
    colors = [input_color, pred_color, gt_color]

    for i in range(3):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        visualize_point_cloud(pos, colors[i], ax, elev=elev, azim=azim,
                              roll=roll, point_size=point_size, bg_color=bg_color)
        ax.set_title(titles[i], pad=5, fontsize=17)

    fig.patch.set_alpha(1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _get_renderer(width=1024, height=1024, bg_color=(1, 1, 1, 1)):
    global _O3D_RENDERER
    if _O3D_RENDERER is None:
        _O3D_RENDERER = o3d.visualization.rendering.OffscreenRenderer(width, height)
        _O3D_RENDERER.scene.set_background(bg_color)
        _O3D_RENDERER.scene.scene.set_sun_light(
            np.array([0.577, 0.577, 0.577], dtype=np.float32),  # direction
            np.array([1.0, 1.0, 1.0], dtype=np.float32),  # color
            75000.0  # intensity
        )
        _O3D_RENDERER.scene.scene.enable_sun_light(False)
    return _O3D_RENDERER


def _pcd_from_np(pos, color, max_points=None, voxel_downsample=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos.astype(np.float32))
    if color.dtype != np.float32 and color.dtype != np.float64:
        col = (color.astype(np.float32) / 255.0)
    else:
        col = np.clip(color, 0, 1).astype(np.float32)
    pcd.colors = o3d.utility.Vector3dVector(col)

    if max_points is not None and len(pcd.points) > max_points:
        if voxel_downsample:
            b = pcd.get_axis_aligned_bounding_box()
            diag = np.linalg.norm(np.array(b.get_max_bound()) - np.array(b.get_min_bound()))
            voxel = max(diag / (max_points ** (1 / 3)), 1e-6)
            pcd = pcd.voxel_down_sample(voxel)
        else:
            sel = np.random.choice(np.asarray(pcd.points).shape[0], size=max_points, replace=False)
            pcd = pcd.select_by_index(sel.tolist())
    return pcd


def _set_camera_orbit(scene, center, radius,
                      elev_deg=60, azim_deg=0, roll_deg=0,
                      ortho=True, fov_deg=60.0, zoom=1.0):
    ea = np.deg2rad(elev_deg)
    aa = np.deg2rad(azim_deg)
    front = np.array([
        np.cos(ea) * np.cos(aa),
        np.cos(ea) * np.sin(aa),
        np.sin(ea)
    ], dtype=np.float32)
    front = front / (np.linalg.norm(front) + 1e-8)

    cam_dist = radius / max(zoom, 1e-3)
    cam_pos = center + (-front) * cam_dist

    up = np.array([0, 0, 1], dtype=np.float32)
    if abs(roll_deg) > 1e-6:
        r = np.deg2rad(roll_deg)
        axis = front / (np.linalg.norm(front) + 1e-8)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + np.sin(r) * K + (1 - np.cos(r)) * (K @ K)
        up = (R @ up).astype(np.float32)

    cam = scene.camera
    aspect = 1.0
    cam.set_projection(fov_deg, aspect, 0.01, 1000.0, o3d.visualization.rendering.Camera.FovType.Horizontal)
    cam.look_at(center, cam_pos, up)


def render_pc_o3d(pos, color,
                  width=1024, height=1024,
                  bg_color=(1, 1, 1, 1),
                  elev=60, azim=0, roll=0,
                  point_size=2.0,
                  ortho=True, fov_deg=60.0, zoom=1.0,
                  max_points=None, voxel_downsample=True):
    renderer = _get_renderer(width, height, bg_color)
    scene = renderer.scene
    scene.clear_geometry()

    pcd = _pcd_from_np(pos, color, max_points=max_points, voxel_downsample=voxel_downsample)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = float(point_size)

    scene.add_geometry("pcd", pcd, mat)

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center().astype(np.float32)
    radius = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound()).astype(np.float32)
    radius = float(max(radius, 1e-4))

    _set_camera_orbit(scene, center, radius, elev_deg=elev, azim_deg=azim,
                      roll_deg=roll, ortho=ortho, fov_deg=fov_deg, zoom=zoom)

    img = renderer.render_to_image()
    return img


def render_pc_compare_o3d(save_path, pos, input_color, pred_color, gt_color,
                          width=768, height=768,
                          bg_color=(0.12, 0.12, 0.12, 1.0),
                          elev=60, azim=0, roll=0,
                          point_size=5.0,
                          ortho=True, fov_deg=60.0, zoom=1.0,
                          max_points=100000, voxel_downsample=True,
                          is_save=True):
    tmp_imgs = []
    for name, col in zip(("input", "pred", "gt"), (input_color, pred_color, gt_color)):
        img = render_pc_o3d(pos, col,
                            width=width, height=height,
                            bg_color=bg_color,
                            elev=elev, azim=azim, roll=roll,
                            point_size=point_size,
                            ortho=ortho, fov_deg=fov_deg, zoom=zoom,
                            max_points=max_points, voxel_downsample=voxel_downsample)
        tmp_imgs.append(img)

    imgs = [np.asarray(p) for p in tmp_imgs]
    H = min(im.shape[0] for im in imgs)
    imgs = [im[:H, :, :] for im in imgs]
    strip = np.concatenate(imgs, axis=1)

    if is_save:
        o3d.io.write_image(save_path, o3d.geometry.Image(strip))
        return save_path
    else:
        return strip
