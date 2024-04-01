import math

import torch


class StatBlock:

    def __init__(self):
        self.xyz_gradient_accum = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_screenPct = torch.empty(0)
        self.min_depthF = torch.empty(0)
        self.filter_3D_sq = torch.empty(0)
        self.denom = torch.empty(0)
        self.n_touched_accum = torch.empty(0)
        self.n_dominated_accum = torch.empty(0)
        self.total_px = 0
        self.associated_color = torch.empty(0)

        self.need_recalc_distF = False


    def create_stats_vars(self, N):
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.max_radii2D = torch.zeros(N, dtype=torch.float, device="cuda")
        self.max_screenPct = torch.zeros(N, dtype=torch.float, device="cuda")
        self.min_depthF = torch.full((N,), fill_value=torch.inf, device="cuda")
        self.filter_3D_sq = torch.full((N, 1), fill_value=torch.inf, device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")
        self.n_touched_accum = torch.zeros((N, 1), device="cuda")
        self.n_dominated_accum = torch.zeros((N, 1), device="cuda")
        self.total_px = 0
        self.associated_color = torch.rand((N, 3), device="cuda")

    def reset_stats_by_mask(self, mask):
        assert len(mask.shape) == 1
        self.xyz_gradient_accum[mask] = 0
        self.max_radii2D[mask] = 0
        self.max_screenPct[mask] = 0
        # self.min_depthF[mask] = torch.inf
        self.denom[mask, :] = 0
        self.n_touched_accum[mask, :] = 0
        self.n_dominated_accum[mask, :] = 0
        self.total_px = 0


    def shrink_stats_by_mask(self, valid_points_mask):
        assert len(valid_points_mask.shape) == 1
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_screenPct = self.max_screenPct[valid_points_mask]
        self.min_depthF = self.min_depthF[valid_points_mask]
        self.filter_3D_sq = self.filter_3D_sq[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.n_touched_accum = self.n_touched_accum[valid_points_mask]
        self.n_dominated_accum = self.n_dominated_accum[valid_points_mask]
        self.associated_color = self.associated_color[valid_points_mask]

    def expand_stats_by_N(self, N_to_add):
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros((N_to_add, 1), device="cuda")), dim=0)
        self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros(N_to_add, device="cuda")), dim=0)
        self.max_screenPct = torch.cat((self.max_screenPct, torch.zeros(N_to_add, device="cuda")), dim=0)
        self.min_depthF = torch.cat((self.min_depthF, torch.full((N_to_add,), fill_value=torch.inf, device="cuda")), dim=0)
        # old_max = self.find_filter3D_old_max()
        self.filter_3D_sq = torch.cat((self.filter_3D_sq, torch.full((N_to_add, 1), fill_value=torch.inf, device="cuda")), dim=0)
        self.denom = torch.cat((self.denom, torch.zeros((N_to_add, 1), device="cuda")), dim=0)
        self.n_touched_accum = torch.cat((self.n_touched_accum, torch.zeros((N_to_add, 1), device="cuda")), dim=0)
        self.n_dominated_accum = torch.cat((self.n_dominated_accum, torch.zeros((N_to_add, 1), device="cuda")), dim=0)
        self.associated_color = torch.cat((self.associated_color, torch.rand((N_to_add, 3), device="cuda")), dim=0)

        self.need_recalc_distF = N_to_add > 0


    def find_filter3D_old_max(self):
        valid_all = self.filter_3D_sq[self.filter_3D_sq.isfinite()]
        if valid_all.shape[0]:
            old_max = float(valid_all.max(dim=0).values)
        else:
            old_max = torch.inf
        return old_max


    def capture(self):
        return (
            # self.xyz_gradient_accum,
            # self.max_radii2D,
            self.filter_3D_sq,
            self.denom,
            # self.n_touched_accum,
        )

    def restore(self, args):
        (
            # self.xyz_gradient_accum,
            # self.max_radii2D,
            self.filter_3D_sq,
            self.denom,
            # self.n_touched_accum,
        ) = args

    def add_densification_stats(self, viewspace_point_tensor, update_filter, radii, n_touched, n_dominated, splat_depths, fx, total_px):
        # Keep track of max radii in image-space for pruning
        ntoched_upd = n_touched[update_filter]
        ndom_upd = n_dominated[update_filter]
        # weights = nto_upd / nto_upd.sum()

        self.max_radii2D[update_filter] = torch.max(self.max_radii2D[update_filter], radii[update_filter])
        self.max_screenPct[update_filter] = torch.max(self.max_screenPct[update_filter], ntoched_upd / total_px)


        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True) * ntoched_upd[:, None]
        self.denom[update_filter] += 1
        # там у НЕ попадающий в updated_filter - нули, что логично
        self.n_touched_accum[update_filter] += ntoched_upd[:, None]
        self.n_dominated_accum[update_filter] += ndom_upd[:, None]

        self.min_depthF[update_filter] = torch.min(self.min_depthF[update_filter], splat_depths[update_filter] / fx)

        self.total_px += total_px

    def on_new_stack(self):
        self.propagate_stack_to_filter_3D_sq()

        self.min_depthF[:] = torch.inf


    def propagate_stack_to_filter_3D_sq(self):
        filter_3D = ((self.min_depthF * (0.2 ** 0.5)) ** 2)
        new_valid = filter_3D.isfinite()
        if not self.filter_3D_sq.shape[0]:
            self.filter_3D_sq = filter_3D.unsqueeze(1)
        elif new_valid.any():
            self.filter_3D_sq[new_valid] = filter_3D[new_valid].unsqueeze(1)
        valid = self.filter_3D_sq.isfinite()
        if valid.any():
            self.filter_3D_sq[~valid] = self.filter_3D_sq[valid].max()
        else:
            self.filter_3D_sq[:] = torch.inf


class StatBlockZZZ:
    def __init__(self):
        self.max_screenPct = torch.empty(0)
        # self.max_radii2Dpct = torch.empty(0)
        self.max_Ntouched = torch.empty(0)
        self.min_invDepth = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradientWpx_accum = torch.empty(0)
        self.Nseen_accum = torch.empty(0)
        self.Npx_accum = torch.empty(0)
        self.NpxDom_accum = torch.empty(0)

    def create_stats_vars(self, N):
        self.max_screenPct = torch.zeros(N, device="cuda")
        # self.max_radii2Dpct = torch.zeros(N, device="cuda")
        self.max_Ntouched = torch.zeros(N, device="cuda")
        self.min_invDepth = torch.full((N,), fill_value=torch.inf, device="cuda")
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.xyz_gradientWpx_accum = torch.zeros((N, 1), device="cuda")
        self.Nseen_accum = torch.zeros((N, 1), device="cuda")
        self.Npx_accum = torch.zeros((N, 1), device="cuda")
        self.NpxDom_accum = torch.zeros((N, 1), device="cuda")


    def reset_stats_by_mask(self, mask):
        self.max_screenPct[mask] = 0
        # self.max_radii2Dpct[mask] = 0
        self.max_Ntouched[mask] = 0
        # self.min_invDepth[mask] = torch.inf
        self.xyz_gradient_accum[mask, :] = 0
        self.xyz_gradientWpx_accum[mask, :] = 0
        self.Nseen_accum[mask, :] = 0
        self.Npx_accum[mask, :] = 0
        self.NpxDom_accum[mask, :] = 0

    def reset_invDepth(self):
        # self.min_invDepth[:] = torch.inf
        pass


    def shrink_stats_by_mask(self, valid_points_mask):
        self.max_screenPct = self.max_screenPct[valid_points_mask]
        # self.max_radii2Dpct = self.max_radii2Dpct[valid_points_mask]
        self.max_Ntouched = self.max_Ntouched[valid_points_mask]
        self.min_invDepth = self.min_invDepth[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradientWpx_accum = self.xyz_gradientWpx_accum[valid_points_mask]
        self.Nseen_accum = self.Nseen_accum[valid_points_mask]
        self.Npx_accum = self.Npx_accum[valid_points_mask]
        self.NpxDom_accum = self.NpxDom_accum[valid_points_mask]


    def expand_stats_by_N(self, N_to_add):
        self.max_screenPct = torch.cat((self.max_screenPct, torch.zeros(N_to_add, device="cuda")), dim=0)
        # self.max_radii2Dpct = torch.cat((self.max_radii2Dpct, torch.zeros(N_to_add, device="cuda")), dim=0)
        self.max_Ntouched = torch.cat((self.max_Ntouched, torch.zeros(N_to_add, device="cuda")), dim=0)
        self.min_invDepth = torch.cat((self.min_invDepth, torch.full((N_to_add,), fill_value=torch.inf, device="cuda")), dim=0)
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros((N_to_add, 1), device="cuda")), dim=0)
        self.xyz_gradientWpx_accum = torch.cat((self.xyz_gradientWpx_accum, torch.zeros((N_to_add, 1), device="cuda")), dim=0)
        self.Nseen_accum = torch.cat((self.Nseen_accum, torch.zeros((N_to_add, 1), device="cuda")), dim=0)
        self.Npx_accum = torch.cat((self.Npx_accum, torch.zeros((N_to_add, 1), device="cuda")), dim=0)
        self.NpxDom_accum = torch.cat((self.NpxDom_accum, torch.zeros((N_to_add, 1), device="cuda")), dim=0)


    def add_densification_stats(self,
                                viewspace_point_tensor, update_filter, radii,
                                n_touched, n_dominated, splat_depths, total_px: int, fx: float):
        ntou_upd = n_touched[update_filter].detach()
        ndom_upd = n_dominated[update_filter].detach()
        grad = viewspace_point_tensor.grad[update_filter, :2].detach()
        grad_norm = torch.norm(grad, dim=-1, keepdim=True)

        sq = (2 * math.pi * (radii[update_filter] ** 2)).detach()

        self.max_screenPct[update_filter] = torch.max(self.max_screenPct[update_filter], ndom_upd / total_px)
        # self.max_radii2Dpct[update_filter] = torch.max(self.max_radii2Dpct[update_filter], sq / total_px)
        self.max_Ntouched[update_filter] = torch.max(self.max_Ntouched[update_filter], ntou_upd)
        self.min_invDepth[update_filter] = torch.min(self.min_invDepth[update_filter], splat_depths[update_filter] / fx)
        self.xyz_gradient_accum[update_filter] += grad_norm
        self.xyz_gradientWpx_accum[update_filter] += grad_norm * ndom_upd[:, None]
        self.Nseen_accum[update_filter] += 1
        self.Npx_accum[update_filter] += ntou_upd[:, None]
        self.NpxDom_accum[update_filter] += ndom_upd[:, None]


    def capture(self):
        return (
            self.max_screenPct,
            # self.max_radii2Dpct,
            self.max_Ntouched,
            self.min_invDepth,
            self.xyz_gradient_accum,
            self.xyz_gradientWpx_accum,
            self.Nseen_accum,
            self.Npx_accum,
            self.NpxDom_accum
        )

    def restore(self, args):
        (
            self.max_screenPct,
            # self.max_radii2Dpct,
            self.max_Ntouched,
            self.min_invDepth,
            self.xyz_gradient_accum,
            self.xyz_gradientWpx_accum,
            self.Nseen_accum,
            self.Npx_accum,
            self.NpxDom_accum
        ) = args


    def seen_enough(self, Nmin):
        return self.Nseen_accum[:, 0] >= Nmin

    def at_least_Npixels(self, Nmin):
        return self.max_Ntouched[:] >= Nmin

    def gradWpx(self):
        grads = self.xyz_gradientWpx_accum / self.NpxDom_accum
        grads[grads.isnan()] = 0.0
        return grads

    def gradWseen(self):
        grads = self.xyz_gradient_accum / self.Nseen_accum
        grads[grads.isnan()] = 0.0
        return grads

    def bigger_than_pct(self, pct):
        return (self.max_screenPct > pct) # | (self.max_radii2Dpct > pct)
