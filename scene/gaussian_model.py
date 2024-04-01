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

from arguments import OptimizationParams
from scene.cameras import Camera
from scene.stat_block import StatBlock
from utils.camera_utils import depth_to_points3d
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, kl_divergence
from torch import nn, Tensor
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import strip_symmetric, build_scaling_rotation

from pytorch3d.ops import knn_points



class BaseGaussianModel:

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


    def __init__(self, sh_degree: int, divide_ratio: float = 0.8):
        self.active_sh_degree = 0
        self.divide_ratio = divide_ratio
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._skysphere = torch.empty(0)

        self.statblock = StatBlock()

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
            *self.statblock.capture(),
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
        *stablock_data,
        opt_dict, 
        self.spatial_lr_scale) = model_args

        self.statblock.restore(stablock_data)

        self.training_setup(training_args)
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

    def training_setup(self, training_args: OptimizationParams):
        self.statblock.create_stats_vars(self._xyz.shape[0])

        self.percent_dense = training_args.percent_dense

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
        # self.optimizer = torch.optim.Adam(l, amsgrad=True)

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


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                if "max_exp_avg_sq" in stored_state: # для AMSGrad
                    stored_state["max_exp_avg_sq"] = torch.zeros_like(tensor)

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
                if "max_exp_avg_sq" in stored_state: # для AMSGrad
                    stored_state["max_exp_avg_sq"] = stored_state["max_exp_avg_sq"][mask]

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

        self.statblock.shrink_stats_by_mask(valid_points_mask)


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                if "max_exp_avg_sq" in stored_state: # для AMSGrad
                    stored_state["max_exp_avg_sq"] = torch.cat((stored_state["max_exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0).contiguous()

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

        self.statblock.expand_stats_by_N(new_xyz.shape[0])

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

        self.split_by_mask(selected_pts_mask, N)


    def split_by_mask(self, selected_pts_mask, N_new=2):
        stds = self.get_scaling[selected_pts_mask].repeat(N_new, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N_new, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N_new, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N_new, 1) / (self.divide_ratio * N_new))
        new_rotation = self._rotation[selected_pts_mask].repeat(N_new, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N_new, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N_new, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N_new, 1)

        old_sky = self.get_skysphere[selected_pts_mask].repeat(N_new, 1)
        # суть в том, чтобы переместить 0 в центр 50/50, а потом уменьшить на четверть уверенность. и вернуть 0 обратно в в 50/50
        v = 0.75
        sky_w_lower_confidence = ((old_sky - 0.5) * v) + 0.5
        new_skysphere = self.inverse_skysphere_activation(sky_w_lower_confidence)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_skysphere)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N_new * selected_pts_mask.sum(), device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)


    def densify_and_clone_by_proximity(self, scene_extent, N = 3, kl_threshold=None):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = dist > 2.5

        if kl_threshold:
            selected_pts_mask = self.update_mask_with_KL(kl_threshold, selected_pts_mask, "clone_prox")

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        new_skysphere = self._skysphere[new_indices]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_skysphere)


    def densify_and_clone(self, grads, grad_threshold, scene_extent, kl_threshold=None, min_dist=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        biggest_scale = torch.max(self.get_scaling, dim=1).values
        selected_pts_mask &= biggest_scale <= self.percent_dense * scene_extent
        # selected_pts_mask &= (self.get_opacity > 0.5).squeeze()

        # dist, nearest_indices = distCUDA2(self.get_xyz)
        # selected_pts_mask &= dist > 0.0025

        if kl_threshold:
            selected_pts_mask = self.update_mask_with_KL(kl_threshold, selected_pts_mask, "clone", min_dist=min_dist)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_skysphere = self._skysphere[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_skysphere)


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size,
                          kl_threshold=None, skysphere_radius=300, kill_outliers: float | None = False,
                          largest_point_divider=5, smallest_point_divider=2500):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        print(f"prune by opacity: {torch.count_nonzero(prune_mask)} / {prune_mask.shape[0]}")

        # а ещё грохнуть небо, слишком отклоняющееся от сферы и слишком далёкую землю


        if opt.skysphere_radius > 0:
            sertain_world_mask = (self.get_skysphere.detach() < 0.25).squeeze()
            sertain_sky_mask = (self.get_skysphere.detach() > 0.75).squeeze()

            sky_dist = torch.norm(self.get_xyz[sertain_sky_mask].detach(), dim=1).squeeze()
            bad_sky = (sky_dist < opt.skysphere_radius * 0.9) | (sky_dist > opt.skysphere_radius * 1.1)
            xyz_world = self.get_xyz[sertain_world_mask].detach()
            bad_world_dist = torch.norm(xyz_world, dim=1).squeeze() >= opt.skysphere_radius / 3
            print(f"bad skysphere distance: {torch.count_nonzero(bad_sky)} sky, {torch.count_nonzero(bad_world_dist)} world")
            prune_mask[sertain_sky_mask] |= bad_sky
            prune_mask[sertain_world_mask] |= bad_world_dist
        else:
            # всё считается миром.
            xyz_world = self.get_xyz.detach()
            sertain_world_mask = torch.ones_like(prune_mask)

        # не, без этих фоновых точек - слишком геометрию поводит
        if kill_outliers and extent > 0:
            dists_to_nearest: torch.Tensor = knn_points(xyz_world[None, ...], xyz_world[None, ...], K=2).dists[0, :, 1].to(prune_mask.device)
            mean_dist_between = dists_to_nearest.mean()
            sigma = dists_to_nearest.std()
            spacing_max = max(extent/100, (mean_dist_between + kill_outliers * sigma))
            lone_pts = dists_to_nearest > spacing_max
            mm = torch.rand(lone_pts.shape[0], device=prune_mask.device)
            lone_pts[mm > 0.25] = False
            print(f"lone world points: {torch.count_nonzero(lone_pts)}")
            prune_by_lonelesnes = torch.zeros_like(prune_mask)
            prune_by_lonelesnes[sertain_world_mask] = lone_pts

            prune_mask |= prune_by_lonelesnes

        # невидимыми могут и оказаться ставшие слишком мелкими и выключенные из рендеринга из-за этого
        if 0:
            invisible_points = self.n_touched_accum == 0
            print(f"invisible_points: {torch.count_nonzero(invisible_points)}")
            prune_mask |= invisible_points.squeeze(1)

        if max_screen_size:
            # max_extent = torch.full_like(self._scaling[:, 0], fill_value=0.2*extent)
            # # небо далеко, и пиксели там большие.
            # # и не сразу становится понятно - небо там, или как
            # max_extent[~sertain_world_mask] *= 100

            # big_points_vs = self.max_radii2D > max_screen_size
            # ws = big_points_vs | big_points_ws
            pass


        if opt.largest_point_divider > 0:
            big_points_ws = self.get_scaling.max(dim=1).values > extent / opt.largest_point_divider
            ws = (big_points_ws & sertain_world_mask)
            print(f"big_points: {torch.count_nonzero(big_points_ws)}")
        else:
            ws = None

        if opt.smallest_point_divider > 0:
            small_points_ws = self.get_scaling.max(dim=1).values < extent / smallest_point_divider * cam_res_down
            print(f"small_points: {torch.count_nonzero(small_points_ws)}")
            if ws is not None:
                ws |= small_points_ws
            else:
                ws = small_points_ws

        if ws is not None:
            prune_mask |= stoch1(ws, 0.5)

        self.prune_points(prune_mask)

        # s = float(self.n_touched.sum())
        denom = self.statblock.n_touched_accum
        grads = self.statblock.xyz_gradient_accum / denom
        grads[grads.isnan()] = 0.0

        # ну попробую самые яркие сплилить
        max_grad = float(grads[grads.isfinite() & (grads > 0)].quantile(opt.split_quantile))

        # weights = self.n_touched / self.n_touched.sum()
        # grads_w = self.xyz_gradient_accum * weights

        if smallest_point_divider > 0:
            min_dist = extent / smallest_point_divider * 2
        else:
            min_dist = None

        self.densify_and_clone(grads, max_grad, extent, kl_threshold=kl_threshold, min_dist=min_dist)
        # self.densify_and_clone_by_proximity(scene_extent=extent, kl_threshold=kl_threshold)
        self.densify_and_split(grads, max_grad, extent, kl_threshold=kl_threshold)
        # if kl_threshold:
        self.kl_merge(grads, max_grad, extent, kl_threshold=0.1)#kl_threshold/4)


    def update_mask_with_KL(self, kl_threshold, selected_pts_mask, op_name="unk", min_dist=None):
        if torch.count_nonzero(selected_pts_mask) == 0:
            return selected_pts_mask

        kl_div, _ = self.calc_kl_div(selected_pts_mask, min_dist)

        if kl_div is not None:
            kl_thres_mask = kl_div > kl_threshold

            N_pre_kl = int(kl_thres_mask.shape[0])
            N_post_kl = int(kl_thres_mask.count_nonzero())

            print(f"[kl+d {op_name}]: {kl_thres_mask.shape[0]}(-{N_pre_kl - N_post_kl}) of {selected_pts_mask.shape[0]}", )
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

        kl_full_mask = torch.zeros_like(selected_pts_mask)
        kl_full_mask[selected_pts_mask] = kl_thres_mask

        selected_pts_mask &= kl_full_mask

        # создаём усреднённые точки
        selected_point_ids = point_ids[:, kl_thres_mask, :][0]

        revs = torch.all(selected_point_ids == torch.dstack((torch.max(selected_point_ids, dim=1)[0], torch.min(selected_point_ids, dim=1)[0]))[0], dim=1)
        selected_point_ids = selected_point_ids[~revs]

        print(f"[kl merge]: (-{(int(selected_point_ids.shape[0]))}")

        new_xyz = self.get_xyz[selected_point_ids].mean(1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_point_ids][:,0] / 0.8)
        new_rotation = self._rotation[selected_point_ids][:,0]
        new_features_dc = self._features_dc[selected_point_ids].mean(1)
        new_features_rest = self._features_rest[selected_point_ids].mean(1)
        new_opacity = self._opacity[selected_point_ids].mean(1)
        new_skysphere = self._skysphere[selected_point_ids][:,0]

        # удаляем исходные
        selected_pts_mask[selected_point_ids[:,1]] = True
        # # prune_filter = torch.cat((selected_pts_mask, torch.zeros(selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # prune_filter = torch.cat((selected_pts_mask, torch.zeros(new_xyz.shape[0], device="cuda", dtype=bool)))
        self.prune_points(selected_pts_mask)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_skysphere)


    def calc_kl_div(self, selected_pts_mask=None, min_dist=None):

        xyz = self._xyz.detach()
        rot = self._rotation.detach()
        scaling = self.get_scaling.detach()

        if selected_pts_mask is not None:
            xyz_selected = xyz[selected_pts_mask].clone()
        else:
            xyz_selected = xyz

        if xyz_selected.shape[0] == 0:
            return None, None

        KNN = knn_points(xyz_selected[None, ...], xyz[None, ...], K=2)
        indices = KNN.idx[0]
        distances = KNN.dists[0, :, 1]

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

        if min_dist is not None:
            kl_div[distances < min_dist] = torch.nan

        return kl_div, t_idx

    def densify_from_depthmap(self, viewpoint_cam, depth, mask, gt_image, skyness=0.5):
        assert depth.shape == mask.shape == gt_image.shape[1:]
        points3d_all = depth_to_points3d(depth, viewpoint_cam)
        points3d = points3d_all[mask]
        colors = gt_image.permute(1, 2, 0)[mask]

        params = self.params_from_points3d(points3d, colors, None, skyness)
        #update gaussians
        self.densification_postfix(*params)

