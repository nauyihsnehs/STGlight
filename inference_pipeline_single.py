import os

from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import math
from pathlib import Path
import argparse

import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
import open3d as o3d

from diff_gaussian_rasterization_pano import GaussianRasterizationSettings, GaussianRasterizer
from datasets.vlsnet_dataset import ls_detector
from models.base_model import ModelRegistry
from tdgs.gaussian_spherical_splatting import Camera
from utils.config_utils import ConfigManager

EXR_SAVE_PARAMS = [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF,
                   cv.IMWRITE_EXR_COMPRESSION, cv.IMWRITE_EXR_COMPRESSION_PIZ]


class Intrinsics:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class PipelineModels:
    def __init__(self, vlsnet=None, alnet=None, nfnet=None):
        self.vlsnet = vlsnet
        self.alnet = alnet
        self.nfnet = nfnet


def _to_device(model, device):
    model = model.to(device)
    model.eval()
    return model


def load_model(config_path, ckpt_path, device):
    cfg = ConfigManager.load_config(config_path)
    mcfg = cfg.get("model", {})
    model = ModelRegistry.get_model(mcfg.get("name"), mcfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return _to_device(model, device)


def read_rgb_depth(rgb_path, depth_path):
    rgb = cv.imread(str(rgb_path))
    if rgb is None:
        raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
    rgb = rgb.astype(np.float32) / 255.0
    depth = cv.imread(str(depth_path), cv.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image: {depth_path}")
    depth = depth.astype(np.float32)
    if depth.max() > 100.0:
        depth = depth / 1000.0  # mm -> m
    return rgb, depth


def depth_to_pointcloud(depth, intr):
    h, w = depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    x = (xx - intr.cx) / intr.fx
    y = (yy - intr.cy) / intr.fy
    z = np.ones_like(x)
    points = np.stack([x, y, z], axis=-1) * depth[..., None]
    return points.reshape(-1, 3).astype(np.float32)


def downsample_point_cloud(pos, color, voxel_size=0.04):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.colors = o3d.utility.Vector3dVector(color)
    down = pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(down.points)
    cols = np.asarray(down.colors)
    return pts, cols


def pack_isotropic_gaussians(points, colors, s=0.03, opacity=1.0, device="cuda"):
    pts = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    cols = torch.from_numpy(colors).to(device=device, dtype=torch.float32)

    count = pts.shape[0]
    scales = torch.full((count, 3), float(s), dtype=torch.float32, device=device)

    quats = torch.zeros((count, 4), dtype=torch.float32, device=device)
    quats[:, 3] = 1.0

    opacities = torch.full((count, 1), float(opacity), dtype=torch.float32, device=device)
    return [pts, scales, quats, cols, opacities]


def pixel_to_camera(pixels, depth, intr, offset=0.2):
    h, w = depth.shape
    points = []
    for u, v, d in pixels:
        if not (0 <= u < w and 0 <= v < h):
            continue
        z = float(depth[int(v), int(u)]) * d - offset
        if z <= 0:
            continue
        x = (u - intr.cx) / intr.fx * z
        y = (v - intr.cy) / intr.fy * z
        points.append(np.array([x, y, z], dtype=np.float32))
    return points


def _resolve_insertion_pixels(depth_shape):
    h, w = depth_shape
    pixels = []
    for u_norm, v_norm, d_norm in INSERTIONS:
        u = int(round(u_norm * (w - 1)))
        v = int(round(v_norm * (h - 1)))
        pixels.append((u, v, d_norm))
    return pixels


def _build_camera(width, height, intr, cam_center, device):
    fov_x = 2 * math.atan(width / (2 * intr.fx))
    fov_y = 2 * math.atan(height / (2 * intr.fy))
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = cam_center.detach().cpu().numpy()
    return Camera(pose, fov_x, fov_y, "single", width, height, device=device)


def _render_gaussians(camera, means3d, scales, quats, colors, opacities):
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=camera.render_height, image_width=camera.render_width, tanfovx=tanfovx, tanfovy=tanfovy,
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device=means3d.device),
        scale_modifier=1.0, viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=3, campos=camera.camera_center, prefiltered=False, debug=False
    )

    screenspace_points = torch.zeros_like(means3d, dtype=means3d.dtype, requires_grad=True, device=means3d.device)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, _ = rasterizer(means3D=means3d, means2D=screenspace_points, shs=None, colors_precomp=colors,
                                   opacities=opacities, scales=scales, rotations=quats, cov3D_precomp=None)
    return rendered_image


def splat_points_to_envmap(gaussians, query, intr, height=512, width=1024, background=0.0):
    means3d, scales, quats, colors, opacities = gaussians
    if means3d.numel() == 0:
        env = torch.full((3, height, width), background, device=query.device)
        dep = torch.full((1, height, width), torch.inf, device=query.device)
        return env, dep

    camera = _build_camera(width, height, intr, query, means3d.device)
    env = _render_gaussians(camera, means3d, scales, quats, colors, opacities)
    dists = torch.linalg.norm(means3d - camera.camera_center, dim=1)
    depth_colors = dists[:, None].repeat(1, 3)
    depth = _render_gaussians(camera, means3d, scales, quats, depth_colors, opacities)
    depth = depth[:1]
    depth[depth <= 0.0] = torch.inf
    return env, depth


def resize_long(image, target_size):
    h, w = image.shape[:2]
    if h >= w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized


def run_vlsnet(model, rgb, depth, device):
    rgb_small = resize_long(rgb, 320)
    init_pos, init_color = ls_detector(rgb_small, threshold=253)
    if init_pos is None:
        print(f'[Warning] No line segments detected by VLS detector.')
        return None, None, None

    depth_small = resize_long(depth, 320)
    depth_small = np.clip(depth_small, 0, 10)

    rgb_t = torch.from_numpy(rgb_small).float().permute(2, 0, 1).unsqueeze(0).to(device)
    depth_t = torch.from_numpy(depth_small).float().unsqueeze(0).to(device)
    init_pos_t = torch.from_numpy(init_pos).float().unsqueeze(0).to(device)
    init_color_t = torch.from_numpy(init_color).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pred_gs = model(rgb_t, init_pos_t, init_color_t)
        _, means3d, colors, scales, _, _ = model.gs_activate(pred_gs, depth_t, init_pos_t)

    return means3d[0].detach(), colors[0].detach(), scales[0].detach()


def run_alnet(model, envmap):
    with torch.no_grad():
        _, pred_pano = model(envmap.unsqueeze(0))
    pred_pano = pred_pano[0].clamp_min(0.0)
    return torch.expm1(pred_pano)


def _build_nfnet_inputs(env_lin, light_lin):
    mask = (env_lin.sum(dim=0, keepdim=True) > 0).float()
    ldr_env = torch.log1p(env_lin).clamp(0.0, 1.0)
    env_input = ldr_env * mask * 2.0 - 1.0
    light_input = torch.log1p(light_lin).clamp_max(3.0) * 2.0 - 1.0
    return env_input, light_input, mask


def predict_nfnet_weight(nfnet_model, env_lin, light_lin):
    env_log = torch.log1p(env_lin).clamp_max(3.0) * 2.0 - 1.0
    light_log = torch.log1p(light_lin).clamp_max(3.0) * 2.0 - 1.0
    if hasattr(nfnet_model, "backbone"):
        fused, w = nfnet_model.backbone(env_log.unsqueeze(0), light_log.unsqueeze(0), None)
    elif hasattr(nfnet_model, "model"):
        fused, w = nfnet_model.model(env_log.unsqueeze(0), light_log.unsqueeze(0), None)
    else:
        raise RuntimeError("NFNet model does not expose a backbone for weight prediction.")
    return fused


def inference_pipeline_single(rgb, depth, insertion_pixels, intrinsics, models, env_size=(128, 256), device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_env, w_env = env_size
    points = depth_to_pointcloud(depth, intrinsics)
    colors = rgb.reshape(-1, 3)
    points_ds, colors_ds = downsample_point_cloud(points, colors, voxel_size=0.01)
    gaussians = pack_isotropic_gaussians(points_ds, colors_ds, s=0.02, opacity=1.0, device=device)

    insertion_points = pixel_to_camera(insertion_pixels, depth, intrinsics)
    query_points = [torch.from_numpy(p).float().to(device) for p in insertion_points]

    vls_points, vls_colors, vls_scales = run_vlsnet(models.vlsnet, rgb, depth, device)

    outputs = {
        "env": [],
        "ambient": [],
        "vls": [],
        "final": [],
    }

    for query in query_points:
        envmap_ori, depth_env = splat_points_to_envmap(gaussians, query, intrinsics)
        envmap = F.interpolate(envmap_ori.unsqueeze(0), size=[h_env, w_env], mode='area').squeeze(0)

        if vls_points is not None:
            vls_points_np = vls_points.detach().cpu().numpy()
            vls_points_np[:, 1] *= -1
            vls_points_np[:, 2] *= -1
            vls_colors_np = vls_colors.detach().cpu().numpy()
            vls_gaussians = pack_isotropic_gaussians(vls_points_np, vls_colors_np, device=device)
            vls_gaussians[1] = vls_scales * 3  # todo
            vls_map, depth_vls = splat_points_to_envmap(vls_gaussians, query, intrinsics)
            vls_mask = depth_vls <= depth_env
            vls_map = vls_map * vls_mask.float()
            vls_map = F.interpolate(vls_map.unsqueeze(0), size=[h_env, w_env], mode='area').squeeze(0)
        else:
            vls_map = torch.zeros_like(envmap)

        amb_map = run_alnet(models.alnet, envmap)

        final_map = predict_nfnet_weight(models.nfnet, envmap, amb_map)
        final_map = torch.expm1(final_map[0] * 0.5 + 0.5) + vls_map

        outputs["env"].append(envmap_ori)
        outputs["ambient"].append(amb_map)
        outputs["vls"].append(vls_map)
        outputs["final"].append(final_map)

    return outputs


def _save_exr(path, envmap):
    arr = envmap.permute(1, 2, 0).float().detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), arr, EXR_SAVE_PARAMS)


def _save_jpg(path, envmap):
    arr = envmap.permute(1, 2, 0).float().detach().cpu().numpy()
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), arr)


def _list_sorted_files(path):
    entries = sorted([p for p in Path(path).iterdir() if p.is_file()])
    return entries


def _load_intrinsics_from_folder(folder, stem, image=None):
    json_path = Path(folder) / f"{stem}.json"
    if not json_path.exists():
        return None
    data = json_path.read_text()
    try:
        parsed = eval(data)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse intrinsics json: {json_path}") from exc
    cx = parsed.get("cx", None)
    cy = parsed.get("cy", None)
    if cx is None or cy is None:
        if image is not None:
            h, w = image.shape[:2]
            cx = w / 2.0
            cy = h / 2.0
        else:
            raise RuntimeError(f"Invalid intrinsics json: {json_path}")
    fx = parsed.get("fx", None)
    fy = parsed.get("fy", None)
    if fx is None or fy is None:
        fov_x = parsed.get("fov_x", None)
        fov_y = parsed.get("fov_y", None)
        if fov_x is None and fov_y is None:
            raise RuntimeError(f"Invalid intrinsics json: {json_path}")
        fx = cx / np.tan(np.deg2rad(fov_x) / 2.0)
        fy = cy / np.tan(np.deg2rad(fov_y) / 2.0)
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def _resolve_intrinsics(args, stem, image=None):
    if args.fxfycxcy:
        intr = _load_intrinsics_from_folder(args.fxfycxcy, stem, image)
        return intr
        # raise RuntimeError(f"Missing intrinsics json for frame: {stem}")
    fx, fy, cx, cy = FXFYCXCY
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


INSERTIONS = [(0.5, 0.5, 0.75), ]

FXFYCXCY = [853.33 / 4, 853.33 / 4, 160.0, 120.0]


def main():
    parser = argparse.ArgumentParser(description="Single-image inference pipeline.")
    parser.add_argument("--rgb", type=str, default='test_img/pipeline_single/inputs/rgb')
    parser.add_argument("--depth", type=str, default='test_img/pipeline_single/inputs/depth')
    parser.add_argument("--fxfycxcy", type=str, default='test_img/pipeline_single/inputs/intrinsic')
    parser.add_argument("--output-dir", type=str, default='test_img/pipeline_single/outputs')
    parser.add_argument("--alnet-config", type=str, default='configs/alnet.yaml')
    parser.add_argument("--alnet-ckpt", type=str, default='checkpoints/alnet_e149.ckpt')
    parser.add_argument("--vlsnet-config", type=str, default='configs/vlsnet.yaml')
    parser.add_argument("--vlsnet-ckpt", type=str, default='checkpoints/vlsnet_e004.ckpt')
    parser.add_argument("--nfnet-config", type=str, default='configs/nfnet.yaml')
    parser.add_argument("--nfnet-ckpt", type=str, default='checkpoints/nfnet_e4.ckpt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = PipelineModels()
    if args.vlsnet_config and args.vlsnet_ckpt:
        models.vlsnet = load_model(args.vlsnet_config, args.vlsnet_ckpt, device)
    if args.alnet_config and args.alnet_ckpt:
        models.alnet = load_model(args.alnet_config, args.alnet_ckpt, device)
    if args.nfnet_config and args.nfnet_ckpt:
        models.nfnet = load_model(args.nfnet_config, args.nfnet_ckpt, device)

    rgb_path = Path(args.rgb)
    depth_path = Path(args.depth)

    if rgb_path.is_dir() and depth_path.is_dir():
        rgb_files = _list_sorted_files(rgb_path)
        depth_files = _list_sorted_files(depth_path)
        if len(rgb_files) != len(depth_files):
            raise RuntimeError("RGB and depth folder must have the same number of files.")
        pairs = list(zip(rgb_files, depth_files))
    else:
        pairs = [(rgb_path, depth_path)]

    out_dir = Path(args.output_dir)
    for rgb_file, depth_file in tqdm(pairs):
        rgb, depth = read_rgb_depth(rgb_file, depth_file)
        stem = rgb_file.stem
        intr = _resolve_intrinsics(args, stem, rgb)
        insert_pixels = _resolve_insertion_pixels(depth.shape)
        outputs = inference_pipeline_single(rgb, depth, insert_pixels, intr, models, device=device)
        base_dir = out_dir / stem
        for idx, final in enumerate(outputs["final"]):
            _save_exr(base_dir / f"insert_{idx:02d}_final.exr", final)
            _save_jpg(base_dir / f"insert_{idx:02d}_env.jpg", outputs["env"][idx])
            _save_exr(base_dir / f"insert_{idx:02d}_ambient.exr", outputs["ambient"][idx])
            _save_exr(base_dir / f"insert_{idx:02d}_vls.exr", outputs["vls"][idx])


if __name__ == "__main__":
    main()
