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

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING

@dataclass
class ModelParams:
    sh_degree: int = 0

    resolution_scales: tuple = (1.0, 1/4)
    resolution_scales_bw: tuple = (1/4,)

    agisoft_calibs: bool = False
    waymo_calibs: bool = False
    dreamer_calibs: bool = False

    N_random_init_pts: int | None = None
    spatial_scaling_lr_mult: float = 1

    source_path: str = MISSING
    model_path: str = MISSING
    images: str = "images"
    white_background: bool = False
    data_device: str = "cuda"
    load_mask: bool = False
    load_skymask: bool = False
    load_normal: bool = False
    load_depth: bool = False
    eval: bool = False


@dataclass
class PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False


@dataclass
class ProgressParams:
    save_gaussians_iterations: list[int] = field(default_factory=list)
    save_checkoint_iterations: list[int] = field(default_factory=list)
    testing_iterations: list[int] = field(default_factory=list)

    load_gaussians_path: str | Path | None = None

    load_checkoint: int | None = None
    load_checkoint_path: str | Path | None = None


# @dataclass
# class SplatShape:
#     flatten_loss: bool = False
#     flatten_aniso_loss: bool = False
#     isotropy_loss: bool = True
#     lambda_flatten: float = 0.5
#     lambda_aniso: float = 5.0
#     lambda_iso: float = 0.25
#     aniso_ratio_threshold: float = 2.5
#
#
# @dataclass
# class SkySphere:
#     skysphere_loss: bool = False
#     sky_depth_loss: bool = False
#     lambda_skysphere_mask: float = 0.2
#     lambda_skysphere_dist: float = 0.2
#     lambda_skysphere_entropy: float = 0.2
#     lambda_sky_depth: float = 0.001
#     skysphere_radius: int = 300
#

@dataclass
class OptimizationParams:

    iterations: int = 30_000

    depth_threshold: float | None = 0.37

    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000

    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    skysphere_lr: float = 0.01

    percent_dense: float = 0.01
    normal_loss: bool = False
    depth_loss: bool = False
    depth2normal_loss: bool = False

    lambda_l1_normal: float = 0.01
    lambda_cos_normal: float = 0.01

    lambda_dssim: float = 0.2
    lambda_lpips: float = 0.2

    semitransparent_loss: bool = False
    semitransparent_from_iter: int = 5_000
    semitransparent_until_iter: int = 7_000
    lambda_semitransparent: float = 0.0005

    # splat_shape: Optional[SplatShape] = None
    # skysphere: Optional[SkySphere] = None
    flatten_loss: bool = False
    flatten_aniso_loss: bool = False
    isotropy_loss: bool = True
    lambda_flatten: float = 0.5
    lambda_aniso: float = 5.0
    lambda_iso: float = 0.25
    aniso_ratio_threshold: float = 2.5

    skysphere_loss: bool = False
    sky_depth_loss: bool = False
    lambda_skysphere_mask: float = 0.2
    lambda_skysphere_dist: float = 0.2
    lambda_skysphere_entropy: float = 0.2
    lambda_sky_depth: float = 0.001

    skysphere_radius: float = -1
    skysphere_radius_in_cam_extents: float = 100

    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002

    random_background: bool = False

    masked_image: bool = False

    fft_loss: bool = False
    fft_lowpass_sigma_iter: int = 7000
    fft_lowpass_sigma_initial: float = 0.4
    fft_lowpass_sigma_max: float = 2.5
    lambda_fft_ampl: float = .005
    lambda_fft_phi: float = .001

    silhouette_loss: bool = False
    silhouette_loss_type: str = "bce" # : Literal["bce", "mce"]
    lambda_silhouette: float = 2.5

    kl_threshold: float | None = None

    largest_point_divider: int = 5
    smallest_point_divider: int = 2_500

    divide_ratio: float = 0.8
    default_sigma: float = 0.3
    c2f: bool = False
    c2f_max_sigma: float = 300
    c2f_every_step: int = 100


@dataclass
class GaussianSplattingConf:
    model_params: ModelParams = field(default_factory=ModelParams)
    pipeline_params: PipelineParams = field(default_factory=PipelineParams)
    optimization_params: OptimizationParams = field(default_factory=OptimizationParams)
    progress_params: ProgressParams = field(default_factory=ProgressParams)

