#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import Union

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, focal2fov, getProjectionMatrix2

_tnp = Union[torch.Tensor, np.ndarray]


def _npy(v: np.ndarray | torch.Tensor):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().float().numpy()
    return v


def _torch(v: np.ndarray | torch.Tensor, device=None):
    if isinstance(v, np.ndarray):
        v = torch.tensor(v, dtype=torch.float)
    if device is not None:
        v = v.to(device)
    return v


@torch.jit.script
def rotate_halfplanes_and_check_is_inside(
        points_cam: torch.Tensor, frustum_halfplanes: torch.Tensor, R: torch.Tensor, camera_center: torch.Tensor):
    halfplanes_world = torch.matmul(frustum_halfplanes, R.T)
    dots: torch.Tensor = torch.matmul(halfplanes_world, (points_cam[:, :3] - camera_center).transpose(0, 1))
    return dots > 0

@torch.jit.script
def camZ_of_pointsW(points_world: torch.Tensor, R: torch.Tensor, camera_center) -> torch.Tensor:
    camera_view_direction_world = R[:, 2] # GLM
    vectors_to_points_world = points_world - camera_center

    z_in_camera_space = torch.mv(vectors_to_points_world, camera_view_direction_world)
    # z_in_camera_space = torch.sum(torch.mul(vectors_to_points_world, camera_view_direction_world), dim=1)
    return z_in_camera_space


class MiniCamKRT(nn.Module):
    def __init__(self, K: _tnp, R: _tnp, T: _tnp, image_width: int, image_height: int, zfar=1000.0, znear=0.01,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, uid: int = -1, image_name: str = "", data_device="cuda"):
        super(MiniCamKRT, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.uid = uid
        self.image_name = image_name

        self.K = _torch(K, self.data_device)
        self.R = _torch(R, self.data_device)
        self.T = _torch(T, self.data_device)
        self.trans = trans
        self.scale = scale
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.zfar = zfar
        self.znear = znear

        self.FoVx: float = None
        self.FoVy: float = None

        self.focal_x: float = None
        self.focal_y: float = None

        self.world_view_transform: torch.Tensor = None
        self.projection_matrix: torch.Tensor = None
        self.full_proj_transform: torch.Tensor = None
        self.camera_center: torch.Tensor = None

        self.init_derived()


    def init_derived(self):
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        self.focal_x = float(fx)
        self.focal_y = float(fy)

        self.FoVx = focal2fov(fx, self.image_width)
        self.FoVy = focal2fov(fy, self.image_height)

        self.world_view_transform = getWorld2View2(self.R, self.T, self.trans, self.scale).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, cx=cx, cy=cy, fx=fx, fy=fy, W=self.image_width, H=self.image_height).transpose(0, 1).to(
            self.data_device)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0).to(self.data_device)
        self.camera_center = self.world_view_transform.inverse()[3, :3].to(self.data_device)


    def points_W2C(self, points3d):
        num_points = points3d.shape[0]

        ones = torch.ones(num_points, 1, dtype=points3d.dtype, device=points3d.device)
        points_homogeneous = torch.cat([points3d, ones], dim=1)

        RT = torch.eye(4, device=self.R.device)
        RT[:3, :3] = self.R.T  # GLM
        RT[:3, 3] = self.T.squeeze()

        points_camera_frame_homogeneous = torch.matmul(points_homogeneous, RT.transpose(0, 1))
        points_camera_frame = points_camera_frame_homogeneous[:, :3] / points_camera_frame_homogeneous[:, 3:4]

        return points_camera_frame

    def camZ_of_pointsW(self, points_world: torch.Tensor) -> torch.Tensor:
        # camera_view_direction_world = self.R[:, 2] # GLM
        # vectors_to_points_world = points_world - self.camera_center
        #
        # z_in_camera_space = torch.mv(vectors_to_points_world, camera_view_direction_world)
        # # z_in_camera_space = torch.sum(torch.mul(vectors_to_points_world, camera_view_direction_world), dim=1)
        # return z_in_camera_space
        return camZ_of_pointsW(points_world, self.R, self.camera_center)


class DynamicMiniCamKRT(nn.Module):
    def __init__(self, K, R, T, image_width, image_height, zfar=1000.0, znear=0.01,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):

        super(DynamicMiniCamKRT, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.K = _torch(K, self.data_device)
        self.R = _torch(R, self.data_device)
        self.T = _torch(T, self.data_device)
        self.image_width = int(image_width)
        self.image_height = int(image_height)

        self.zfar = zfar
        self.znear = znear

        self.trans = trans
        self.scale = scale

    @property
    def FoVx(self):
        fx = self.K[0, 0]
        return focal2fov(fx, self.image_width)

    @property
    def FoVy(self):
        fy = self.K[1, 1]
        return focal2fov(fy, self.image_height)

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T, self.trans, self.scale).transpose(0, 1).to(self.data_device)

    @property
    def projection_matrix(self):
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        return getProjectionMatrix2(znear=self.znear, zfar=self.zfar,
                                    cx=cx, cy=cy, fx=fx, fy=fy,
                                    W=self.image_width, H=self.image_height).transpose(0, 1).to(self.data_device)

    @property
    def full_proj_transform(self):
        return self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)
        ).squeeze(0).to(self.data_device)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3].to(self.data_device)

class FRU1:
    def __init__(self, cam: MiniCamKRT):
        self.cam = cam
        self.frustum_halfplanes = None

        self.frustum_halfplanes = self.calc_frustum_halfplanes()


    def calc_frustum_halfplanes(self):
        """
        Computes half-planes for the frustum in world coordinates based on given visible corners and the camera center.
        """

        fx = self.cam.K[0, 0]
        fy = self.cam.K[1, 1]
        cx = self.cam.K[0, 2]
        cy = self.cam.K[1, 2]

        h = self.cam.image_height
        w = self.cam.image_width

        w_margin = 0.15*w
        h_margin = 0.15*h

        left = -w_margin
        top = -h_margin
        bottom = h + h_margin
        right = w + w_margin

        # Visible corners in Normalized Device Coordinates (NDC)
        corners_ndc = torch.tensor([
            [left,  top,    1],  # Top-left near plane
            [left,  bottom, 1],  # Bottom-left near plane
            [right, bottom, 1],  # Bottom-right near plane
            [right, top,    1]   # Top-right near plane
        ], dtype=torch.float32, device=self.cam.data_device)

        # Convert corners from NDC to camera space
        corners_cam = torch.zeros((4, 3), device=self.cam.data_device)  # Homogeneous coordinates
        corners_cam[:, 0] = (corners_ndc[:, 0] - cx) * self.cam.znear / fx
        corners_cam[:, 1] = (corners_ndc[:, 1] - cy) * self.cam.znear / fy
        corners_cam[:, 2] = self.cam.znear

        # Compute normals for the half-planes in world coordinates
        normals = []
        for i in range(len(corners_cam)):
            next_i = (i + 1) % len(corners_cam)
            # Compute two vectors in the plane
            v1 = corners_cam[i, :3]# - camera_center_world
            v2 = corners_cam[next_i, :3]# - camera_center_world
            # Cross product to get the normal, pointing inward
            normal = torch.cross(v2, v1, dim=0)  # Order of cross product reversed to ensure inward pointing normals
            normal /= torch.norm(normal, dim=0)  # Normalize
            normals.append(normal)

        # Convert normals to list of tensors
        normals = torch.stack(normals)

        # The half-planes are defined by their normals originating from the camera center in world coordinates
        return normals


    def frustum_local(self, points_cam):
        # Calculate dot products between normals and points. Since normals are inward, a negative result indicates a point outside the frustrum.
        dots = torch.matmul(self.frustum_halfplanes, points_cam[:, :3].transpose(0, 1))
        inside = torch.all(dots > 0, axis=0)
        return inside


    def frustum_world(self, points_cam):
        # Calculate dot products between normals and points. Since normals are inward, a negative result indicates a point outside the frustrum.

        dots_ = rotate_halfplanes_and_check_is_inside(points_cam, self.frustum_halfplanes, self.R, self.camera_center)
        return torch.all(dots_, axis=0)

    #
    # def points_inside_frustum(self, points: torch.Tensor) -> torch.Tensor:
    #     """
    #     Checks if the given points are inside the camera frustum.
    #
    #     Args:
    #         points: A tensor of shape (N, 3) representing N 3D points in world coordinates.
    #
    #     Returns:
    #         A tensor of shape (N,) of type bool indicating whether each point is within the frustum.
    #     """
    #
    #     points_cam = self.cam.points_W2C(points)
    #     inside = self.frustum_local(points_cam)
    #
    #     return inside


class Camera(MiniCamKRT):
    def __init__(self, colmap_id, K: _tnp, R: _tnp, T: _tnp, image: torch.Tensor, image_name: str, uid: int,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 gt_alpha_mask=None,
                 sky_mask=None,
                 normal=None,
                 depth=None):

        image_width = image.shape[2]
        image_height = image.shape[1]

        super().__init__(K, R, T, image_width, image_height, trans=trans, scale=scale, data_device=data_device,
                         uid=uid, image_name=image_name)

        self.colmap_id = colmap_id

        self.original_image = image.clamp(0.0, 1.0).to(torch.float32).to(self.data_device)

        self.sky_mask: torch.Tensor | None = sky_mask
        self.normal: torch.Tensor | None = normal
        self.depth: torch.Tensor | None = depth
        self.mask: torch.Tensor | None = gt_alpha_mask


    def updateImage(self, image,
                    gt_alpha_mask=None,
                    sky_mask=None,
                    normal=None,
                    depth=None):

        new_image_width = image.shape[2]
        new_image_height = image.shape[1]

        scale_x = new_image_width / self.image_width
        scale_y = new_image_height / self.image_height
        assert np.allclose(scale_x, scale_y, rtol=1e-6)
        scale = scale_x

        self.K[0, 0] = self.K[0, 0] * scale
        self.K[1, 1] = self.K[1, 1] * scale
        self.K[0, 2] = self.K[0, 2] * scale
        self.K[1, 2] = self.K[1, 2] * scale

        self.image_width = new_image_width
        self.image_height = new_image_height
        self.original_image = image.clamp(0.0, 1.0).to(torch.float32).to(self.data_device)
        self.mask = gt_alpha_mask
        self.sky_mask = sky_mask
        self.normal = normal
        self.depth = depth

        # Recalculate derived attributes based on updated K matrix and image dimensions
        self.init_derived()


class MiniCam:
    def __init__(self, width: int, height: int, fovy: float, fovx: float, znear: float, zfar: float, world_view_transform, full_proj_transform):
        self.data_device = torch.device("cuda")

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform.to(self.data_device)
        self.full_proj_transform = full_proj_transform.to(self.data_device)

        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def project_points_to_image(points3d: torch.Tensor, cam: 'MiniCamKRT'):
    pixel_coords, points3d_cam_frame = project2d3d(cam, points3d)

    valid_indices = (
            (pixel_coords[:, 0] >= 0)
            & (pixel_coords[:, 0] < cam.image_width)
            & (pixel_coords[:, 1] >= 0)
            & (pixel_coords[:, 1] < cam.image_height)
            & (points3d_cam_frame[:, 2] > cam.znear)
            & (points3d_cam_frame[:, 2] < cam.zfar)
    )

    return pixel_coords, points3d_cam_frame[:, :3], valid_indices


def project2d3d(cam, points3d):
    points3d_cam_frame = cam.points_W2C(points3d)
    pixel_coords_homogeneous = torch.matmul(points3d_cam_frame, cam.K.transpose(0, 1))
    pixel_coords = torch.round(pixel_coords_homogeneous[:, :2] / pixel_coords_homogeneous[:, 2:3]).to(torch.int32)
    return pixel_coords, points3d_cam_frame
