import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from inference_pipeline_single import (
    PipelineModels,
    _list_sorted_files,
    _resolve_insertion_pixels,
    Intrinsics,
    _save_exr,
    _save_jpg,
    depth_to_pointcloud,
    downsample_point_cloud,
    load_model,
    pack_isotropic_gaussians,
    pixel_to_camera,
    read_rgb_depth,
    run_alnet,
    run_vlsnet,
    splat_points_to_envmap,
    predict_nfnet_weight,
)


def _load_pose(path):
    pose = np.loadtxt(path)
    pose = np.asarray(pose, dtype=np.float32)
    return pose


def _apply_pose(points, pose):
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([points, ones], axis=1)
    transformed = hom @ pose.T
    return transformed[:, :3]


def _apply_pose_to_point(point, pose):
    hom = np.concatenate([point, np.array([1.0], dtype=np.float32)], axis=0)
    transformed = pose @ hom
    return transformed[:3]


def _fuse_point_cloud(global_points, global_colors, new_points, new_colors, voxel_size=0.02):
    if global_points is None:
        combined_points = new_points
        combined_colors = new_colors
    else:
        combined_points = np.concatenate([global_points, new_points], axis=0)
        combined_colors = np.concatenate([global_colors, new_colors], axis=0)
    return downsample_point_cloud(combined_points, combined_colors, voxel_size=voxel_size)


def _combine_vls(global_vls, new_vls):
    if new_vls is None:
        return global_vls
    if global_vls is None:
        return new_vls
    points = torch.cat([global_vls[0], new_vls[0]], dim=0)
    colors = torch.cat([global_vls[1], new_vls[1]], dim=0)
    scales = torch.cat([global_vls[2], new_vls[2]], dim=0)
    return points, colors, scales


def _transform_vls_to_world(vls_points, vls_colors, vls_scales, pose, device):
    if vls_points is None:
        return None
    points = vls_points.detach().cpu().numpy()
    points[:, 1] *= -1
    points[:, 2] *= -1
    points = _apply_pose(points, pose)
    points_t = torch.from_numpy(points).float().to(device)
    colors_t = vls_colors.detach().to(device)
    scales_t = vls_scales.detach().to(device)
    return points_t, colors_t, scales_t


def inference_pipeline_static(rgb_files, depth_files, pose_files, intrinsics, models, env_size=(128, 256), device=None,
                              voxel_size=0.02):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_env, w_env = env_size

    outputs = []
    global_points = None
    global_colors = None
    global_vls = None

    for model in (models.vlsnet, models.alnet, models.nfnet):
        if hasattr(model, "reset_temporal_state"):
            model.reset_temporal_state(device=device)

    for idx, (rgb_file, depth_file, pose_file) in enumerate(tqdm(list(zip(rgb_files, depth_files, pose_files)))):
        rgb, depth = read_rgb_depth(rgb_file, depth_file)
        pose = _load_pose(pose_file)

        intr = intrinsics[idx]

        points_cam = depth_to_pointcloud(depth, intr)
        colors = rgb.reshape(-1, 3)
        points_world = _apply_pose(points_cam, pose)
        global_points, global_colors = _fuse_point_cloud(global_points, global_colors, points_world, colors,
                                                         voxel_size=voxel_size)

        insert_pixels = _resolve_insertion_pixels(depth.shape)
        insertion_points = pixel_to_camera(insert_pixels, depth, intr)
        insertion_world = [_apply_pose_to_point(p, pose) for p in insertion_points]
        query_points = [torch.from_numpy(p).float().to(device) for p in insertion_world]

        vls_points, vls_colors, vls_scales = run_vlsnet(models.vlsnet, rgb, depth, device)
        vls_world = _transform_vls_to_world(vls_points, vls_colors, vls_scales, pose, device)
        global_vls = _combine_vls(global_vls, vls_world)

        gaussians = pack_isotropic_gaussians(global_points, global_colors, s=0.02, opacity=1.0, device=device)
        if global_vls is not None:
            vls_gaussians = pack_isotropic_gaussians(
                global_vls[0].detach().cpu().numpy(),
                global_vls[1].detach().cpu().numpy(),
                device=device,
            )
            vls_gaussians[1] = global_vls[2] * 3
        else:
            vls_gaussians = None

        frame_outputs = {"env": [], "ambient": [], "vls": [], "final": []}
        for query in query_points:
            envmap_ori, depth_env = splat_points_to_envmap(gaussians, query, intr)
            envmap = F.interpolate(envmap_ori.unsqueeze(0), size=[h_env, w_env], mode="area").squeeze(0)

            if vls_gaussians is not None:
                vls_map, depth_vls = splat_points_to_envmap(vls_gaussians, query, intr)
                vls_mask = depth_vls <= depth_env
                vls_map = vls_map * vls_mask.float()
                vls_map = F.interpolate(vls_map.unsqueeze(0), size=[h_env, w_env], mode="area").squeeze(0)
            else:
                vls_map = torch.zeros_like(envmap)

            amb_map = run_alnet(models.alnet, envmap)
            light_map = vls_map + amb_map
            final_map = predict_nfnet_weight(models.nfnet, envmap, light_map)
            final_map = torch.expm1(final_map[0] * 0.5 + 0.5)

            frame_outputs["env"].append(envmap_ori)
            frame_outputs["ambient"].append(amb_map)
            frame_outputs["vls"].append(vls_map)
            frame_outputs["final"].append(final_map)

        outputs.append(frame_outputs)

    return outputs


def _match_pose_files(pose_path, rgb_files):
    pose_path = Path(pose_path)
    if pose_path.is_dir():
        pose_files = _list_sorted_files(pose_path)
        if len(pose_files) != len(rgb_files):
            raise RuntimeError("Pose folder must have the same number of files as RGB.")
        return pose_files
    if pose_path.is_file():
        return [pose_path for _ in rgb_files]
    raise RuntimeError(f"Pose path not found: {pose_path}")


def _resolve_intrinsics(args, stem):
    if args.fxfycxcy:
        txt_path = Path(args.fxfycxcy) / f"{stem}.txt"
        mat = np.loadtxt(txt_path)
        mat = np.asarray(mat, dtype=np.float32)
        fx = float(mat[0, 0])
        fy = float(mat[1, 1])
        cx = float(mat[0, 2])
        cy = float(mat[1, 2])
    else:
        fx, fy, cx, cy = FXFYCXCY
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


FXFYCXCY = [585.0, 585.0, 320.0, 240.0]


def main():
    parser = argparse.ArgumentParser(description="Video sequence inference pipeline (static scene).")
    parser.add_argument("--rgb", type=str, default="test_img/pipeline_static/inputs/rgb")
    parser.add_argument("--depth", type=str, default="test_img/pipeline_static/inputs/depth")
    parser.add_argument("--pose", type=str, default="test_img/pipeline_static/inputs/pose")
    # parser.add_argument("--fxfycxcy", type=str, default='test_img/pipeline_static/inputs/intrinsic')
    parser.add_argument("--fxfycxcy", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="test_img/pipeline_static/outputs")
    parser.add_argument("--alnet-config", type=str, default="configs/alnet_tf.yaml")
    parser.add_argument("--alnet-ckpt", type=str,
                        default="logs/alnet_tf/version_29/checkpoints/alnet_tf_epoch=9-val_loss=0.0649.ckpt")
    parser.add_argument("--vlsnet-config", type=str, default="configs/vlsnet_tf.yaml")
    parser.add_argument("--vlsnet-ckpt", type=str,
                        default="logs/vlsnet_tf/version_33/checkpoints/vlsnet_tf_epoch=17-val_loss=14.8538.ckpt")
    parser.add_argument("--nfnet-config", type=str, default="configs/nfnet_tf.yaml")
    parser.add_argument("--nfnet-ckpt", type=str,
                        default="logs/nfnet_tf/version_6/checkpoints/nfnet_tf_epoch=2-val_loss=0.0803.ckpt")
    parser.add_argument("--voxel-size", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = PipelineModels()
    if models.vlsnet is None and args.vlsnet_config and args.vlsnet_ckpt:
        models.vlsnet = load_model(args.vlsnet_config, args.vlsnet_ckpt, device)
    if models.alnet is None and args.alnet_config and args.alnet_ckpt:
        models.alnet = load_model(args.alnet_config, args.alnet_ckpt, device)
    if models.nfnet is None and args.nfnet_config and args.nfnet_ckpt:
        models.nfnet = load_model(args.nfnet_config, args.nfnet_ckpt, device)

    rgb_path = Path(args.rgb)
    depth_path = Path(args.depth)

    if rgb_path.is_dir() and depth_path.is_dir():
        rgb_files = _list_sorted_files(rgb_path)
        depth_files = _list_sorted_files(depth_path)
        if len(rgb_files) != len(depth_files):
            raise RuntimeError("RGB and depth folder must have the same number of files.")
    else:
        rgb_files = [rgb_path]
        depth_files = [depth_path]

    pose_files = _match_pose_files(args.pose, rgb_files)

    intrinsics = []
    for rgb_file, depth_file in zip(rgb_files, depth_files):
        rgb, _ = read_rgb_depth(rgb_file, depth_file)
        intrinsics.append(_resolve_intrinsics(args, rgb_file.stem))

    outputs = inference_pipeline_static(
        rgb_files,
        depth_files,
        pose_files,
        intrinsics,
        models,
        device=device,
        voxel_size=args.voxel_size,
    )

    out_dir = Path(args.output_dir)
    for rgb_file, frame_outputs in zip(rgb_files, outputs):
        base_dir = out_dir / rgb_file.stem
        for insert_idx, final in enumerate(frame_outputs["final"]):
            _save_exr(base_dir / f"insert_{insert_idx:02d}_final.exr", final)
            _save_jpg(base_dir / f"insert_{insert_idx:02d}_env.jpg", frame_outputs["env"][insert_idx])
            _save_exr(base_dir / f"insert_{insert_idx:02d}_ambient.exr", frame_outputs["ambient"][insert_idx])
            _save_exr(base_dir / f"insert_{insert_idx:02d}_vls.exr", frame_outputs["vls"][insert_idx])


if __name__ == "__main__":
    main()
