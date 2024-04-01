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

import sys
from datetime import datetime
import random

import numpy as np
from PIL.Image import Resampling
import torch
import torch.nn.functional as F


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution, resample=Resampling.LANCZOS)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling(s):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    return L

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def kl_divergence(mu_0, rotation_0_q, scaling_0_diag, mu_1, rotation_1_q, scaling_1_diag):
    # from https://arxiv.org/abs/2312.02973 Eq. (7)

    # claculate cov_0
    rotation_0 = build_rotation(rotation_0_q)
    scaling_0 = build_scaling(scaling_0_diag)
    L_0 = rotation_0 @ scaling_0
    cov_0 = L_0 @ L_0.transpose(1, 2)

    # claculate inverse of cov_1
    rotation_1 = build_rotation(rotation_1_q)
    scaling_1_inv = build_scaling(1/scaling_1_diag)
    L_1_inv = rotation_1 @ scaling_1_inv
    cov_1_inv = L_1_inv @ L_1_inv.transpose(1, 2)

    # difference of mu_1 and mu_0
    mu_diff = mu_1 - mu_0

    # calculate kl divergence
    kl_div_0 = torch.vmap(torch.trace)(cov_1_inv @ cov_0)
    kl_div_1 = mu_diff[:,None].matmul(cov_1_inv).matmul(mu_diff[..., None]).squeeze()
    kl_div_2 = torch.log(torch.prod((scaling_1_diag/scaling_0_diag)**2, dim=1))
    kl_div = 0.5 * (kl_div_0 + kl_div_1 + kl_div_2 - 3)
    return kl_div


@torch.jit.script
def conv_gradient_center(input_tensor):
    device = input_tensor.device

    k1 = [[[
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [-0.15, -0.35, 0., 0.35, 0.15],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]
    ]]]

    k2 = [[[
        [0., 0., -0.15, 0., 0.],
        [0., 0., -0.35, 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0.35, 0., 0.],
        [0., 0., 0.15, 0., 0.]
    ]]]

    grad_x_kernel = torch.tensor(k1, device="cuda", dtype=torch.float32)

    grad_y_kernel = torch.tensor(k2, device="cuda", dtype=torch.float32)

    # Применяем kernels
    grad_x = F.conv2d(input=input_tensor, weight=grad_x_kernel, padding=2)
    grad_y = F.conv2d(input=input_tensor, weight=grad_y_kernel, padding=2)

    return grad_x, grad_y

    # mag = torch.sqrt(grad_x**2 + grad_y**2)
    # angle = torch.atan2(grad_y, grad_x) / (2 * torch.pi) + 0.5
    #
    # return mag, angle


@torch.jit.script
def pseudo_normals_from_depthmap_gradient(depth_map):
    # GW = torch.gradient(depth_map, spacing=0.1, dim=1)[0]
    # GH = torch.gradient(depth_map, spacing=0.1, dim=0)[0]

    GW, GH = conv_gradient_center(depth_map.unsqueeze(0).unsqueeze(0))
    GW = GW.squeeze(0, 1)
    GH = GH.squeeze(0, 1)

    denominator = (GW.square() + GH.square() + 1).sqrt()

    n_i = torch.stack([
        GW / denominator,
        GH / denominator,
        1 / denominator
    ], dim=0)

    return n_i


@torch.jit.script
def stoch1(prune_mask, pct=0.75):
    prune_mask &= torch.rand(prune_mask.shape[0], device=prune_mask.device) < pct
    return prune_mask
