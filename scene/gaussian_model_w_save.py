import os

import numpy as np
from plyfile import PlyElement, PlyData
import torch
from torch import nn

from scene.gaussian_model_mip import GaussianModel
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p


def init_gaussians_from_pcd(gaussians: GaussianModel, pcd: BasicPointCloud, spatial_lr_scale: float = 1):
    gaussians.spatial_lr_scale = spatial_lr_scale
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()

    print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    (gaussians._xyz,
     gaussians._features_dc,
     gaussians._features_rest,
     gaussians._opacity,
     gaussians._scaling,
     gaussians._rotation,
     gaussians._skysphere) = gaussians.params_from_points3d(fused_point_cloud, fused_colors)

    gaussians.statblock.create_stats_vars(gaussians.get_xyz.shape[0])


def save_gaussians_ply(gaussian: GaussianModel, path, save_sky=None, save_fused=False):
    mkdir_p(os.path.dirname(path))

    if save_sky is True:
        mask = (gaussian.get_skysphere > 0.5).squeeze()
    elif save_sky is False:
        mask = (gaussian.get_skysphere <= 0.5).squeeze()
    else:
        mask = torch.ones_like(gaussian._skysphere, dtype=torch.bool).squeeze()

    xyz = gaussian._xyz.detach()[mask].cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussian._features_dc.detach()[mask].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussian._features_rest.detach()[mask].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    if save_fused:
        scal, opa = gaussian.get_scal_opa_w_3D
        opacities = opa.detach()[mask].cpu().numpy()
        scale = scal.detach()[mask].cpu().numpy()
    else:
        opacities = gaussian._opacity.detach()[mask].cpu().numpy()
        scale = gaussian._scaling.detach()[mask].cpu().numpy()
    rotation = gaussian._rotation.detach()[mask].cpu().numpy()
    skysphere = gaussian._skysphere.detach()[mask].cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in gaussian.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, skysphere), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def load_gaussians_ply(gaussian: GaussianModel, path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    skysphere = np.asarray(plydata.elements[0]["skysphere"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

    num_extra_f = len(extra_f_names)
    assert num_extra_f % 3 == 0
    sh_degree = int((((num_extra_f // 3) + 1) ** 0.5) - 1)
    assert num_extra_f == 3 * (sh_degree + 1) ** 2 - 3

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    gaussian._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
    gaussian._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    gaussian._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    gaussian._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
    gaussian._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
    gaussian._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
    gaussian._skysphere = nn.Parameter(torch.tensor(skysphere, dtype=torch.float, device="cuda").contiguous().requires_grad_(True))

    gaussian.active_sh_degree = sh_degree

    gaussian.statblock.create_stats_vars(gaussian.get_xyz.shape[0])
