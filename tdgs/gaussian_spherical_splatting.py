import math

import numpy as np
import torch
from diff_gaussian_rasterization_pano import GaussianRasterizationSettings, GaussianRasterizer
from plyfile import PlyData
from torch import nn


class PanoRender:
    def __init__(self, intrinsic, extrinsic, device="cuda", znear=0.01, zfar=50.0):
        self.width, self.height = int(intrinsic[0][-1] * 2), int(intrinsic[1][-1] * 2)
        k_mat = np.array(intrinsic, dtype=np.float32)
        self.fov_x, self.fov_y = 2 * math.atan(self.width / (2 * k_mat[0, 0])), 2 * math.atan(
            self.height / (2 * k_mat[1, 1]))
        pose = self.openglC2W_to_colmapC2W(extrinsic)
        self.camera = Camera(pose, self.fov_x, self.fov_y, "single", self.width, self.height, znear, zfar, device)

    def openglC2W_to_colmapC2W(self, C2W):
        O = np.eye(4, dtype=np.float32)
        O[:3, :3], O[:3, 3] = C2W[:3, :3] @ np.diag([1, -1, -1]), C2W[:3, 3]
        return O

    def rendering(self, means3D, scales, quats, colors, opacities, device='cuda'):
        tanfovx = math.tan(self.fov_x * 0.5)
        tanfovy = math.tan(self.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=self.height,
            image_width=self.width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            scale_modifier=1.0,
            viewmatrix=self.camera.world_view_transform,
            projmatrix=self.camera.full_proj_transform,
            sh_degree=3,  # not use
            campos=self.camera.camera_center,
            prefiltered=False,
            debug=False
        )

        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=device)

        rendered_image, radii = GaussianRasterizer(raster_settings=raster_settings)(
            means3D=means3D,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=quats,
            cov3D_precomp=None)

        return rendered_image

def render(viewpoint_camera, pc, scaling_modifier=1.0):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=viewpoint_camera.render_height,
        image_width=viewpoint_camera.render_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=3,  # not use
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device="cuda")

    rendered_image, radii = GaussianRasterizer(raster_settings=raster_settings)(
        means3D=pc.xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pc.rgb,
        opacities=pc.opacity,
        scales=pc.scaling,
        rotations=pc.rotation,
        cov3D_precomp=None)

    return rendered_image


class GaussianModel(nn.Module):
    def __init__(self, path, device="cuda"):
        super().__init__()
        e = PlyData.read(path).elements[0]
        n = len(e.data)
        xyz = np.column_stack(
            [e["x"], e["y"], e["z"]]
        )
        rgb = np.column_stack(
            [e["red"], e["green"], e["blue"]]
        ) / 255.0 if "red" in e._property_lookup else np.zeros((n, 3), np.float32)
        sc = np.column_stack(
            [e["scale_x"], e["scale_y"], e["scale_z"]]
        ) if "scale_x" in e._property_lookup else np.full((n, 3), 0.05, np.float32)
        rot = np.column_stack(
            [e["rot_x"], e["rot_y"], e["rot_z"], e["rot_w"]]
        ) if "rot_w" in e._property_lookup else np.zeros((n, 4), np.float32)
        op = e["opacity"][:, None] if "opacity" in e._property_lookup else np.ones((n, 1), np.float32)
        toP = lambda a: nn.Parameter(
            torch.tensor(a, dtype=torch.float32, device=torch.device(device), requires_grad=False))
        self.xyz, self.rgb, self.scaling, self.rotation, self.opacity = map(toP, [xyz, rgb, sc, rot, op])


def get_projection_matrix(znear, zfar, fovX, fovY):
    tx, ty = math.tan(fovX / 2), math.tan(fovY / 2)
    l, r, b, t = -tx * znear, tx * znear, -ty * znear, ty * znear
    P = torch.zeros(4, 4)
    P[0, 0], P[1, 1] = 2 * znear / (r - l), 2 * znear / (t - b)
    P[0, 2], P[1, 2] = (r + l) / (r - l), (t + b) / (t - b)
    P[2, 2], P[2, 3], P[3, 2] = zfar / (zfar - znear), -(zfar * znear) / (zfar - znear), 1.0
    return P


class Camera(nn.Module):
    def __init__(self, C2W, FoVx, FoVy, name, w, h, znear=0.01, zfar=100.0, device="cuda"):
        super().__init__()
        self.render_width, self.render_height = w, h
        self.FoVx, self.FoVy = float(FoVx), float(FoVy)
        self.cam_name = str(name)
        self.data_device = torch.device(device)
        self.znear, self.zfar = float(znear), float(zfar)

        C2W_t = torch.as_tensor(C2W, dtype=torch.float32, device=self.data_device)
        W2C = torch.linalg.inv(C2W_t)
        self.world_view_transform = W2C.T
        self.projection_matrix = get_projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy).to(
            self.data_device).T
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = C2W_t[:3, 3]
