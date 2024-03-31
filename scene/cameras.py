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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal, getProjectionMatrix2

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

        self.FoVx = focal2fov(fx, self.image_width)
        self.FoVy = focal2fov(fy, self.image_height)

        self.world_view_transform = getWorld2View2(self.R, self.T, self.trans, self.scale).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, cx=cx, cy=cy, fx=fx, fy=fy, W=self.image_width, H=self.image_height).transpose(0, 1).to(
            self.data_device)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0).to(self.data_device)
        self.camera_center = self.world_view_transform.inverse()[3, :3].to(self.data_device)


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


class Camera(MiniCamKRT):
    def __init__(self, colmap_id, K, R, T, image, image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):

        image_width = image.shape[2]
        image_height = image.shape[1]

        super().__init__(K, R, T, image_width, image_height, trans=trans, scale=scale, data_device=data_device,
                         uid=uid, image_name=image_name)

        self.colmap_id = colmap_id

        self.original_image = image.clamp(0.0, 1.0).to(torch.float32).to(self.data_device)

    def updateImage(self, image):

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
