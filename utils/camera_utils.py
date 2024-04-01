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

from math import floor
import typing

import numpy as np


if typing.TYPE_CHECKING:
    from scene.dataset_readers import CameraInfo

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

WARNED = False


def rescale_K(K: np.ndarray, scale):
    K = np.copy(K) * scale
    K[2, 2] = 1
    return K


def resize_single_channel(input_tensor, resolution):
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    resized_tensor = F.interpolate(input_tensor, size=resolution, mode='nearest-exact')
    return resized_tensor.squeeze()


def loadCam(args, id, cam_info: 'CameraInfo', resolution_scale):
    orig_w, orig_h = cam_info.image.size

    # if args.resolution in [1, 2, 4, 8]:
    scale_down = 1/resolution_scale
    # else:  # should be a type that converts to float
    #     if args.resolution == -1:
    #         if orig_w > 1600:
    #             global WARNED
    #             if not WARNED:
    #                 print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
    #                     "If this is not desired, please explicitly specify '--resolution/-r' as 1")
    #                 WARNED = True
    #             global_down = orig_w / 1600
    #         else:
    #             global_down = 1
    #     else:
    #         global_down = orig_w / args.resolution
    #
    #     scale_down = max(float(global_down), 1/resolution_scale)

    if scale_down != 1:
        resolution = (floor(orig_w / scale_down), floor(orig_h / scale_down))
        K = rescale_K(cam_info.K, 1/scale_down)
    else:
        resolution = (orig_w, orig_h)
        K = cam_info.K

    r_inv = (resolution[1], resolution[0])

    orig_f_cuda = (VF.pil_to_tensor(cam_info.image) / 255).float().unsqueeze(0).cuda()
    resized_image_rgb = F.interpolate(orig_f_cuda, r_inv, mode='bilinear').squeeze(0)

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if cam_info.sky_mask is not None:
        sky_mask_tensor = torch.tensor(cam_info.sky_mask.astype(np.float32), device=resized_image_rgb.device)
        resized_sky_mask = resize_single_channel(sky_mask_tensor, r_inv).to(torch.bool)
    else:
        resized_sky_mask = None

    if cam_info.gt_mask is not None:
        gt_mask_tensor = torch.tensor(cam_info.gt_mask.astype(np.float32), device=resized_image_rgb.device)
        resized_gt_mask = resize_single_channel(gt_mask_tensor, r_inv)
    else:
        resized_gt_mask = None

    if cam_info.normal is not None:
        normal_tensor = torch.tensor(cam_info.normal, device=resized_image_rgb.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        resized_normal = F.interpolate(normal_tensor, r_inv, mode='nearest').squeeze(0)
    else:
        resized_normal = None

    if cam_info.depth is not None:
        depth_tensor = torch.tensor(cam_info.depth.astype(np.float32), device=resized_image_rgb.device)
        resized_depth = resize_single_channel(depth_tensor, r_inv)
    else:
        resized_depth = None

    gt_image = resized_image_rgb[:3, ...]

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    else:
        loaded_mask = resized_gt_mask

    from scene.cameras import Camera
    return Camera(colmap_id=cam_info.uid, K=K, R=cam_info.R, T=cam_info.T,
                  image=gt_image, image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  gt_alpha_mask=loaded_mask, sky_mask=resized_sky_mask, normal=resized_normal, depth=resized_depth)

def cameraList_from_camInfos(cam_infos: list['CameraInfo'], resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : 'CameraInfo'):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fx': float(camera.K[0, 0]),
        'fy': float(camera.K[1, 1]),
        'cx': float(camera.K[0, 2]),
        'cy': float(camera.K[1, 2]),
    }
    return camera_entry


def depth_to_points3d(depth: torch.Tensor, viewpoint_cam: 'Camera'):
    K = viewpoint_cam.K
    cam2world = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
    h, w = depth.shape

    # Create a grid of 2D pixel coordinates
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))

    # Stack the 2D and depth coordinates to create 3D homogeneous coordinates
    coordinates = torch.stack([x.to(depth.device), y.to(depth.device), torch.ones_like(depth)], dim=-1)
    coordinates = coordinates.view(-1, 3).to(K.device).to(torch.float32)
    coordinates_3D = (K.inverse() @ coordinates.T).T
    coordinates_3D *= depth.view(-1, 1)
    world_coordinates_3D = (cam2world[:3, :3] @ coordinates_3D.T).T + cam2world[:3, 3]

    world_coordinates_3D = world_coordinates_3D.view(h, w, 3)
    return world_coordinates_3D

