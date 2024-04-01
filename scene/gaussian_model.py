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
import numpy as np

from utils.camera_utils import depth_to_points3d
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, kl_divergence
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

from pytorch3d.ops import knn_points


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.skysphere_activation = torch.sigmoid
        self.inverse_skysphere_activation = inverse_sigmoid


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._skysphere = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._skysphere,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._skysphere,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_skysphere(self):
        return self.skysphere_activation(self._skysphere)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def params_from_points3d(self, points3d, colors, dist2=None, skyness=0.5):
        assert points3d.isfinite().all()

        if dist2 is None:
            cur_pts3d = self.get_xyz
            if cur_pts3d.shape[0] > 0:
                pts3d1 = torch.cat((points3d, cur_pts3d), dim=0)
            else:
                pts3d1 = points3d

            if pts3d1.shape[0] < 3:
                return

            dist2 = distCUDA2(pts3d1)[0][:points3d.shape[0]]

        # dist2[~dist2.isfinite()] = 1

        dist2_sqrt = torch.sqrt(dist2)
        dist2_sqrt[~dist2_sqrt.isfinite()] = 1

        sigma = float(dist2_sqrt.std())
        mean = float(dist2_sqrt.mean())
        dist2_sqrt = dist2_sqrt.clamp(1e-6, max(mean + 3 * sigma, mean * 3))

        too_big = ~(dist2_sqrt ** 6).isfinite()
        dist2_sqrt[too_big] = 1


        assert points3d.shape[0] == colors.shape[0] == dist2.shape[0]
        N_points = points3d.shape[0]

        harmonics = RGB2SH(colors.contiguous())
        features = torch.zeros((harmonics.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float, device="cuda")
        features[:, :3, 0 ] = harmonics
        features[:, 3:, 1:] = 0.0

        scales = self.scaling_inverse_activation(dist2_sqrt)[..., None].repeat(1, 3)
        assert scales.isfinite().all()
        assert self.scaling_activation(scales).isfinite().all()

        rots = torch.zeros((N_points, 4), dtype=torch.float, device="cuda")
        rots[:, 0] = 1
        opacity = inverse_sigmoid(
            torch.full((N_points, 1), fill_value=0.1, dtype=torch.float, device="cuda")
        )

        # 0.5 probability = zero confidence
        skyness = self.inverse_skysphere_activation(torch.tensor(skyness).clamp(1e-6, 1-1e-6))
        skysphere = torch.full((N_points, 1), fill_value=skyness, dtype=torch.float, device="cuda")

        p_xyz = nn.Parameter(points3d.contiguous().requires_grad_(True))
        p_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        p_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        p_scaling = nn.Parameter(scales.contiguous().requires_grad_(True))
        p_rotation = nn.Parameter(rots.contiguous().requires_grad_(True))
        p_opacities = nn.Parameter(opacity.contiguous().requires_grad_(True))
        p_skysphere = nn.Parameter(skysphere.contiguous().requires_grad_(True))

        return p_xyz, p_features_dc, p_features_rest, p_opacities, p_scaling, p_rotation, p_skysphere

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        (self._xyz,
         self._features_dc,
         self._features_rest,
         self._opacity,
         self._scaling,
         self._rotation,
         self._skysphere) = self.params_from_points3d(fused_point_cloud, fused_colors)

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), dtype=torch.float, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._skysphere], 'lr': training_args.skysphere_lr, "name": "skysphere"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('skysphere')
        return l

    def save_ply(self, path, save_sky=None):
        mkdir_p(os.path.dirname(path))

        if save_sky is True:
            mask = (self.get_skysphere > 0.5).squeeze()
        elif save_sky is False:
            mask = (self.get_skysphere <= 0.5).squeeze()
        else:
            mask = torch.ones_like(self._skysphere, dtype=torch.bool).squeeze()

        xyz = self._xyz.detach()[mask].cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach()[mask].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach()[mask].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach()[mask].cpu().numpy()
        scale = self._scaling.detach()[mask].cpu().numpy()
        rotation = self._rotation.detach()[mask].cpu().numpy()
        skysphere = self._skysphere.detach()[mask].cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, skysphere), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, drop_to=0.01):

        print("reset_opacity")
        # небо не трогаем, оно очень далеко и однотонное, не успеет обновиться до следующего prune
        sky_mask = (self.get_skysphere.detach() > 0.75).squeeze()

        opacities_old = self.get_opacity.detach()
        opacities_new = opacities_old.clone()

        opacities_new[opacities_new > drop_to] = drop_to
        opacities_new[sky_mask] = opacities_old[sky_mask]

        opacities_new_p = self.inverse_opacity_activation(opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new_p,"opacity")

        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._skysphere = nn.Parameter(torch.tensor(skysphere, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = sh_degree


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._skysphere = optimizable_tensors["skysphere"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_skysphere):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "skysphere": new_skysphere}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._skysphere = optimizable_tensors["skysphere"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, kl_threshold=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if kl_threshold:
            selected_pts_mask = self.update_mask_with_KL(kl_threshold, selected_pts_mask, "split")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_skysphere = self._skysphere[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_skysphere)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, kl_threshold=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if kl_threshold:
            selected_pts_mask = self.update_mask_with_KL(kl_threshold, selected_pts_mask, "clone")

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_skysphere = self._skysphere[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_skysphere)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, kl_threshold=None, skysphere_radius=300):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, kl_threshold=kl_threshold)
        self.densify_and_split(grads, max_grad, extent, kl_threshold=kl_threshold)
        if kl_threshold:
            self.kl_merge(grads, max_grad, extent, kl_threshold=kl_threshold/4)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        print(f"prune by opacity: {torch.count_nonzero(prune_mask)} / {prune_mask.shape[0]}")

        # а ещё грохнуть небо, слишком отклоняющееся от сферы и слишком далёкую землю
        sertain_world_mask = (self.get_skysphere.detach() < 0.25).squeeze()
        sertain_sky_mask = (self.get_skysphere.detach() > 0.75).squeeze()
        sky_dist = torch.norm(self.get_xyz[sertain_sky_mask].detach(), dim=1).squeeze()
        bad_sky = (sky_dist < skysphere_radius * 0.9) | (sky_dist > skysphere_radius * 1.1)
        xyz_world = self.get_xyz[sertain_world_mask].detach()
        bad_world_dist = torch.norm(xyz_world, dim=1).squeeze() >= skysphere_radius / 3
        print(f"bad skysphere distance: {torch.count_nonzero(bad_sky)} sky, {torch.count_nonzero(bad_world_dist)} world")
        prune_mask[sertain_sky_mask] |= bad_sky
        prune_mask[sertain_world_mask] |= bad_world_dist

        # не, без этих фоновых точек - слишком геометрию поводит
        if 0:
            dists_to_nearest: torch.Tensor = knn_points(xyz_world[None, ...], xyz_world[None, ...], K=2).dists[0, :, 1]
            mean_dist_between = dists_to_nearest.mean()
            sigma = dists_to_nearest.std()
            lone_pts = dists_to_nearest > (mean_dist_between + 7*sigma)
            print(f"lone world points: {torch.count_nonzero(lone_pts)}")
            prune_by_lonelesnes = torch.zeros_like(prune_mask)
            prune_by_lonelesnes[sertain_world_mask] = lone_pts

            prune_mask |= prune_by_lonelesnes

        if max_screen_size:
            max_extent = torch.full_like(self._scaling[:, 0], fill_value=0.1 * extent)
            # небо далеко, и пиксели там большие.
            # и не сразу становится понятно - небо там, или как
            max_extent[~sertain_world_mask] *= 100

            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > max_extent
            prune_mask |= big_points_vs | big_points_ws
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()


    def update_mask_with_KL(self, kl_threshold, selected_pts_mask, op_name="unk"):
        if torch.count_nonzero(selected_pts_mask) == 0:
            return selected_pts_mask

        kl_div, _ = self.calc_kl_div(selected_pts_mask)

        if kl_div is not None:
            kl_thres_mask = kl_div > kl_threshold

            N_pre_kl = int(kl_thres_mask.shape[0])
            N_post_kl = int(kl_thres_mask.count_nonzero())

            print(f"[kl {op_name}]: {kl_thres_mask.shape[0]}(-{N_pre_kl - N_post_kl}) of {selected_pts_mask.shape[0]}", )
            kl_full_mask = torch.zeros_like(selected_pts_mask)
            kl_full_mask[selected_pts_mask] = kl_thres_mask

            selected_pts_mask &= kl_full_mask

        return selected_pts_mask


    def kl_merge(self, grads, grad_threshold, scene_extent, kl_threshold=0.1):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if torch.count_nonzero(selected_pts_mask) == 0:
            return

        kl_div, point_ids = self.calc_kl_div(selected_pts_mask)
        if kl_div is None:
            return

        kl_thres_mask = kl_div < kl_threshold
        if torch.count_nonzero(kl_thres_mask) == 0:
            return

        print(f"[kl merge]: (-{(int(torch.count_nonzero(kl_thres_mask)))})")

        kl_full_mask = torch.zeros_like(selected_pts_mask)
        kl_full_mask[selected_pts_mask] = kl_thres_mask

        selected_pts_mask &= kl_full_mask

        # создаём усреднённые точки
        selected_point_ids = point_ids[0]
        new_xyz = self.get_xyz[selected_point_ids].mean(1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_point_ids][:,0] / 0.8)
        new_rotation = self._rotation[selected_point_ids][:,0]
        new_features_dc = self._features_dc[selected_point_ids].mean(1)
        new_features_rest = self._features_rest[selected_point_ids].mean(1)
        new_opacity = self._opacity[selected_point_ids].mean(1)
        new_skysphere = self._skysphere[selected_point_ids][:,0]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_skysphere)

        # удаляем исходные
        selected_pts_mask[selected_point_ids[:,1]] = True
        # prune_filter = torch.cat((selected_pts_mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(new_xyz.shape[0], device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def calc_kl_div(self, selected_pts_mask=None):

        xyz = self._xyz.detach()
        rot = self._rotation.detach()
        scaling = self.get_scaling.detach()

        if selected_pts_mask is not None:
            xyz_selected = xyz[selected_pts_mask].clone()
        else:
            xyz_selected = xyz

        if xyz_selected.shape[0] == 0:
            return None, None

        indices = knn_points(xyz_selected[None, ...], xyz[None, ...], K=2).idx[0]

        # первая точка из индекса knn. она-же - запрошенная, сама себе ближайшая
        idx_pt_itself = indices[:, 0]
        xyz_0 = xyz[idx_pt_itself].detach()
        rotation_0_q = rot[idx_pt_itself].detach()
        scaling_diag_0 = scaling[idx_pt_itself].detach()

        # вторая точка из индекса knn. ближайшая к запрошенной.
        idx_nn = indices[:, 1]
        xyz_1 = xyz[idx_nn].detach()
        rotation_1_q = rot[idx_nn].detach()
        scaling_diag_1 = scaling[idx_nn].detach()

        kl_div = kl_divergence(xyz_0, rotation_0_q, scaling_diag_0,
                                 xyz_1, rotation_1_q, scaling_diag_1)

        t_idx = indices[None, ...].to(xyz_selected.device)

        return kl_div, t_idx


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def densify_from_depthmap(self, viewpoint_cam, depth, mask, gt_image, skyness=0.5):
        assert depth.shape == mask.shape == gt_image.shape[1:]
        points3d_all = depth_to_points3d(depth, viewpoint_cam)
        points3d = points3d_all[mask]
        colors = gt_image.permute(1, 2, 0)[mask]

        params = self.params_from_points3d(points3d, colors, None, skyness)
        if params is not None:
            self.densification_postfix(*params)
