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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model_w_save import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation
import torch.nn.functional as F

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           return_normal=False, return_skyness=False, kernel_size: float=0.3, depth_threshold = None):
    """
    Render the scene.
    """

    bg_color = bg_color.to(pc.get_xyz.device)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        depth_threshold=depth_threshold,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, rendered_depth, rendered_alpha, n_touched, splat_depths = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_dict = {"render": rendered_image,
                   "viewspace_points": screenspace_points,
                   "visibility_filter": n_touched > 0,
                   "radii": radii,
                   "rendered_depth": rendered_depth,  # depth
                   "rendered_alpha": rendered_alpha,  # acc
                   "n_touched": n_touched,
                   "splat_depths": splat_depths,
                   "n_dominated": n_dominated,
                   "dominating_splat": dominating_splat
                   }

    if return_normal:
        rotations_mat = build_rotation(rotations)
        scales = pc.get_scaling
        min_scales = torch.argmin(scales, dim=1)
        indices = torch.arange(min_scales.shape[0])
        normal = rotations_mat[indices, :, min_scales]

        # convert normal direction to the camera; calculate the normal in the camera coordinate
        view_dir = means3D - viewpoint_camera.camera_center
        normal   = normal * ((((view_dir * normal).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]

        R_w2c = viewpoint_camera.R.T
        normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)
  
        render_normal, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = normal,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        render_normal = F.normalize(render_normal, dim = 0)
        return_dict.update({'rendered_normal': render_normal})

    if False:
        density = torch.ones_like(means3D)

        render_opacity, _, _, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = density,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        return_dict.update({'rendered_opacity': render_opacity.mean(dim=0)})


    if return_skyness:
        # default is uncertain = probability 0.5
        bg_color = torch.tensor([0.5,0.5,0.5], dtype=torch.float32).cuda()
        skyness = pc.get_skysphere.repeat(1, 3)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            bg=bg_color, # <<<<<<<<<<<<<<<<
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            projmatrix_raw=viewpoint_camera.projection_matrix,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            depth_threshold=depth_threshold,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        render_skyness, _, _, _, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=skyness,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        return_dict.update({'rendered_skyness': render_skyness.mean(dim=0)})

    return return_dict
