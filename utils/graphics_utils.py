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

import torch
import math
import numpy as np
from typing import NamedTuple, Tuple, Any


class BasicPointCloud(NamedTuple):
    points : np.ndarray
    colors : np.ndarray
    normals : np.ndarray

def geom_transform_points(points, transf_matrix) -> torch.Tensor:
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getView2World2(Rt: np.ndarray, translate=np.array([.0, .0, .0]), scale=1.0) -> tuple[np.ndarray, np.ndarray]:
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center - translate) / scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)

    R = Rt[:3, :3].transpose()
    t = Rt[:3, 3]

    return R, t


def getWorld2View_npy(R, t) -> np.ndarray:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R: torch.Tensor, t: torch.Tensor,
                   translate=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32), scale=1.0) -> torch.Tensor:

    Rt = torch.eye(4, dtype=torch.float32)
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t

    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY) -> torch.Tensor:
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H) -> torch.Tensor:
    # from https://github.com/muskie82/MonoGS
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def fov2focal(fov, pixels) -> float:
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels) -> float:
    return 2*math.atan(pixels/(2*focal))