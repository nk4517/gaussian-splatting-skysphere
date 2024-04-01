import torch
from torch import Tensor

from scene.cameras import Camera
from scene.gaussian_model import BaseGaussianModel


@torch.jit.script
def compute_scales(scale: torch.Tensor, filter_3D_sq: torch.Tensor) -> tuple[Tensor, Tensor]:
    scales_square = torch.square(scale)
    scales_after_square = scales_square + filter_3D_sq.unsqueeze(1)

    det1 = torch.prod(scales_square, dim=1)
    det2 = torch.prod(scales_after_square, dim=1)
    coef = (det1 / det2).sqrt()

    # # через логарифмы, чтобы не копить ошибки умножений больших и малых флоатов
    # det1_log_sum = scales_square.log().sum(dim=1)
    # det2_log_sum = scales_after_square.log().sum(dim=1)
    # coef_log = (det1_log_sum - det2_log_sum) / 2
    # coef = torch.exp(coef_log)

    return torch.sqrt(scales_after_square), coef.unsqueeze(-1)


class GaussianModel(BaseGaussianModel):

    def propagate_depth_sq_to_filter3D(self, cameras: list[Camera], blur=1):
        invalid = ~self.statblock.min_depthF_sq.isfinite()
        new_depthF = self.find_min_depthF_for_all_cams(cameras, invalid)
        if new_depthF is not None:
            self.statblock.min_depthF_sq[invalid] = new_depthF ** 2

        mult = (blur * (0.2 ** 0.5)) ** 2
        self.statblock.filter3d_sq = self.statblock.min_depthF_sq * mult

        invalid_filter3d_sq = ~self.statblock.filter3d_sq.isfinite()
        self.statblock.filter3d_sq[invalid_filter3d_sq] = self.statblock.filter3d_sq.max()


    def get_scal_opa_w_3D_w_blur(self, blur_mod):
        if not self.statblock.min_depthF_sq.shape[0]:
            return self.get_scaling, self.get_opacity

        scal2, coef = self.calc_new_scaling_and_opacity_coef(self.get_scaling, blur_mod)
        opa2 = self.get_opacity * coef
        return scal2, opa2


    def calc_new_scaling_and_opacity_coef(self, scale, blur_mod=None):
        sq = self.statblock.min_depthF_sq
        if blur_mod:
            sq = sq * blur_mod**2
        return compute_scales(scale, sq)


    def find_min_depthF_for_all_cams(self, cameras, incremental_mask=None):
        N = -1 if incremental_mask is None else incremental_mask.count_nonzero()
        if N == 0:
            return None
        # print(f"find_min_depth_for_all_cams, N={N}")

        # TODO consider focal length and image width
        xyz = self.get_xyz.detach()
        if incremental_mask is not None:
            xyz = xyz[incremental_mask]

        N = xyz.shape[0]

        depths = torch.full((N,), fill_value=torch.inf, dtype=xyz.dtype, device=xyz.device)

        # we should use the focal length of the highest resolution camera
        for camera in cameras:
            # t1 = monotonic()

            valid_w = camera.fru1.frustum_world(xyz)
            if valid_w.count_nonzero() < 2:
                # print(f"cam: {camera.uid}, valid=!!!ZERO!!! or 1")
                continue

            xyz_valid_w = xyz[valid_w, :]

            # xyz_valid_cam = camera.points_W2C(xyz_valid_w)
            # z_valid_w0 = xyz_valid_cam[:, 2].clamp(0.2)

            # так быстрее
            z_valid_w = camera.camZ_of_pointsW(xyz_valid_w).clamp(camera.znear, camera.zfar)

            # assert torch.allclose(z_valid_w0, z_valid_w)

            # from open3d.utility import Vector3dVector
            # from open3d.geometry import PointCloud
            # from open3d.io import write_point_cloud
            #
            # pp = PointCloud(Vector3dVector(xyz[valid_w].cpu().numpy()))
            # write_point_cloud(f"world_{camera.uid}.ply", pp)
            # pp2 = PointCloud(Vector3dVector(xyz_valid_cam.cpu().numpy()))
            # write_point_cloud(f"cam_{camera.uid}.ply", pp2)
            # print(f"cam: {camera.uid}, valid={valid_w.count_nonzero()}") #, valid={valid1.count_nonzero()}, valid_fru={valid_fru.count_nonzero()}")

            # depths[valid] = torch.min(depths[valid], xyz_to_cam[valid])
            depths[valid_w] = torch.min(depths[valid_w], z_valid_w / camera.focal_x)

        return depths


    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.opacity_activation(self._opacity)

        if not self.statblock.min_depthF_sq.shape[0]:
            return opacity

        _, coef = self.calc_new_scaling_and_opacity_coef(self.get_scaling)

        opa2 = opacity * coef

        return opa2


    @property
    def get_scal_opa_w_3D(self):
        # raise 1
        # scale = self.scaling_activation(self._scaling)
        # opacity = self.opacity_activation(self._opacity)

        if not self.statblock.min_depthF_sq.shape[0]:
            return self.get_scaling, self.get_opacity

        scal2, coef = self.calc_new_scaling_and_opacity_coef(self.get_scaling)

        opa2 = self.get_opacity * coef

        return scal2, opa2


    @property
    def get_scaling_with_3D_filter(self):
        scale = self.scaling_activation(self._scaling)

        if not self.statblock.min_depthF_sq.shape[0]:
            return scale

        return (scale.square() + self.statblock.min_depthF_sq).sqrt()


    def reset_opacity(self, DROP_VALUE = 0.01):
        # reset opacity to by considering 3D filter

        _, coef = self.calc_new_scaling_and_opacity_coef(self.get_scaling)

        opacities_new = (self.get_opacity * coef).clone()
        opacities_new[opacities_new > DROP_VALUE] = DROP_VALUE

        opacities_new_p = self.inverse_opacity_activation(opacities_new / coef)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new_p, "opacity")
        self._opacity = optimizable_tensors["opacity"]
