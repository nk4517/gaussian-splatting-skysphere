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

import os
import sys
import threading
from pathlib import Path
from typing import Optional
from random import randint
import uuid
from argparse import ArgumentParser

import omegaconf
from omegaconf import OmegaConf
import torch
from torch.fft import fft2, fftshift
import torch.nn.functional as F

from torchmetrics.functional.regression import pearson_corrcoef

from tqdm import tqdm

from scene.resolution_controller import ResolutionController, unload_cam_data
from utils.fft_utils import gen_gaussian_ellipse_torch, calc_phase_dist
from utils.loss_utils import l1_loss, ssim, binary_cross_entropy
from gaussian_renderer import render
from gaussian_renderer.network_gui import NetworkGUI
from scene import Scene, GaussianModel
from scene.cameras import project2d3d
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import OptimizationParams, GaussianSplattingConf
from utils.skysphere_utils import add_skysphere_points3d

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

with torch.no_grad():
    kernelsize = 3
    conv = torch.nn.Conv2d(1, 1, kernel_size=kernelsize, padding=(kernelsize // 2))
    kernel = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]).reshape(1, 1, kernelsize, kernelsize)
    conv.weight.data = kernel  # torch.ones((1,1,kernelsize,kernelsize))
    conv.bias.data = torch.tensor([0.])
    conv.requires_grad_(False)
    conv = conv.cuda()


def nearMean_map(array, mask, kernelsize=3):
    """ array: (H,W) / mask: (H,W) """
    cnt_map = torch.ones_like(array)

    nearMean = conv((array * mask)[None, None])
    cnt_map = conv((cnt_map * mask)[None, None])
    nearMean = (nearMean / (cnt_map + 1e-8)).squeeze()

    return nearMean

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def monodisp(gt_depth: torch.Tensor, dyn_depth: torch.Tensor, loss_type: str = "l1"):
    t_d = torch.median(dyn_depth, dim=-1, keepdim=True).values
    s_d = torch.mean(torch.abs(dyn_depth - t_d), dim=-1, keepdim=True)
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth, dim=-1, keepdim=True).values
    s_gt = torch.mean(torch.abs(gt_depth - t_gt), dim=-1, keepdim=True)
    gt_depth_norm = (gt_depth - t_gt) / s_gt

    if loss_type == "l1":
        disp_loss = torch.abs((dyn_depth_norm - gt_depth_norm)).mean()
    else:
        disp_loss = ((dyn_depth_norm - gt_depth_norm) ** 2).mean()

    return dyn_depth_norm, gt_depth_norm, disp_loss


def loss_depth_smoothness(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1).unsqueeze(1))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1).unsqueeze(1))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss

def loss_depth_grad(depth, img):
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    weight_x = img_grad_x / (torch.abs(img_grad_x) + 1e-6)
    weight_y = img_grad_y / (torch.abs(img_grad_y) + 1e-6)

    depth_grad_x = depth[:, :, :, :-1] - depth[:, :, :, 1:]
    depth_grad_y = depth[:, :, :-1, :] - depth[:, :, 1:, :]
    grad_x = depth_grad_x / (torch.abs(depth_grad_x) + 1e-6)
    grad_y = depth_grad_y / (torch.abs(depth_grad_y) + 1e-6)

    loss = l1_loss(grad_x, weight_x) + l1_loss(grad_y, weight_y)
    return loss


def training(conf: GaussianSplattingConf, debug_from,
             network_gui: Optional[NetworkGUI], gui):

    opt = conf.optimization_params
    dataset = conf.model_params
    pipe = conf.pipeline_params
    progress = conf.progress_params


    kernel_size = 0.1 # opt.default_sigma

    assert not (opt.skysphere_loss and opt.silhouette_loss)

    if (opt.skysphere_loss or opt.sky_depth_loss) and not dataset.load_skymask:
        raise "SKYMASK!!!"

    print(f"Set divide_ratio to {opt.divide_ratio}")

    first_iter = 0
    tb_writer = prepare_output_and_logger(conf)
    gaussians = GaussianModel(dataset.sh_degree, divide_ratio=opt.divide_ratio)
    scene = Scene(dataset, gaussians, load_iteration=progress.load_checkoint or progress.load_gaussians_path)
    gaussians.training_setup(opt)

    if not progress.load_gaussians_path and progress.load_checkoint_path:
        (model_params, first_iter) = torch.load(progress.load_checkoint_path)
        gaussians.restore(model_params, opt)

    if not progress.load_gaussians_path and not progress.load_checkoint_path:
        if dataset.load_skymask:
            if not opt.skysphere_radius > 0 and opt.skysphere_radius_in_cam_extents > 0:
                opt.skysphere_radius = float(scene.cameras_extent * opt.skysphere_radius_in_cam_extents)
        else:
            opt.skysphere_radius = -1

        with torch.no_grad():
            if dataset.load_skymask:
                add_skysphere_points3d(scene, gaussians, opt.skysphere_radius)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    res_controller = ResolutionController(scene, conf)
    res_controller.start()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1

    N_full_stacks_done = -1
    N_full_stacks_processed = 0

    fft_lowpass_mask = None
    fft_loss_mask = None
    fft_loss_coeffs = None
    fft_mask_sigma = None

    opacity_was_reset_at_iter = -1

    fft_lowpass_sigma = opt.fft_lowpass_sigma_initial if opt.fft_loss else None

    gui.setScene(scene)

    for iteration in range(first_iter, opt.iterations + 1):

        # if network_gui is not None:
        #     with torch.no_grad():
        #         network_gui.tick(opt, pipe, dataset, gaussians, iteration, background)

        if gui and gui.e_want_to_render.is_set():
            gui.e_finished_rendering.wait()

        scene.lock.acquire(blocking=True)

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if not res_controller.c2f_phase and iteration > 15_000 and iteration % 3000 == 0:
            gaussians.oneupSHdegree()

        with torch.no_grad():
            # Pick a random Camera
            if not viewpoint_stack:
                with torch.no_grad():
                    res_controller.update_blur(iteration)
                    if res_controller.update_resolution_if_need() or bootstrap1:
                        gaussians.statblock.on_new_cam_resolution()
                        gaussians.propagate_depth_sq_to_filter3D(res_controller.trainCameras_filter3d)
                        bootstrap1 = False
                    else:
                        gaussians.statblock.on_new_stack()


                N_full_stacks_done += 1
                viewpoint_stack = scene.getTrainCameras(res_controller.cur_cam_res).copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        depth_threshold = None
        if opt.depth_threshold:
            depth_threshold = opt.depth_threshold * scene.cameras_extent

        #render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=args.normal_loss)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            return_normal=opt.normal_loss and not res_controller.c2f_phase,
                            return_skyness=dataset.load_skymask and not res_controller.c2f_phase, kernel_size=kernel_size, depth_threshold=depth_threshold)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        skyness = render_pkg.get("rendered_skyness")
        alpha_img = render_pkg.get("rendered_alpha")

        n_dominated = render_pkg["n_dominated"]
        n_touched = render_pkg["n_touched"]
        splat_depths = render_pkg["splat_depths"]
        total_px = viewpoint_cam.image_width * viewpoint_cam.image_height

        if iteration > opt.densify_from_iter + 1 and splat_depths.shape[0]:
            # если сплаты занимает больше 20% ширины\высоты экрана (а это дохрена) - это флоатеры
            floaters = radii > min(viewpoint_cam.image_height, viewpoint_cam.image_width) * 0.3
            if floaters.any():
                gaussians.prune_points(floaters)
                scene.lock.release()
                scene.was_updated.set()
                continue


        # # opacity mask
        # if False: # iteration < opt.propagated_iteration_begin and opt.depth_loss:
        #     opacity_mask = render_pkg['rendered_alpha'] > 0.999
        #     opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)
        # else:
        #     opacity_mask = render_pkg['rendered_alpha'] > 0.0
        #     opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)

        gt_image = viewpoint_cam.original_image.cuda()

        if gt_image.shape[0] == 1:
            image = gray(image)

        if not res_controller.maximum_cam_res_reached:
            gt_image = VF.gaussian_blur(gt_image.unsqueeze(0), sigma=[0.33, 0.33], kernel_size=[3, 3]).squeeze(0)

        # Loss

        if (opt.masked_image or opt.silhouette_loss) and viewpoint_cam.mask is not None:
            gt_mask = viewpoint_cam.mask.squeeze().cuda()
            gt_mask_bool = viewpoint_cam.mask.bool().squeeze().cuda()
        else:
            gt_mask = None
            gt_mask_bool = None

        if res_controller.c2f_phase or not opt.silhouette_loss:
            if opt.masked_image and gt_mask_bool is not None:
                Ll1 = l1_loss(image[:, gt_mask_bool], gt_image[:, gt_mask_bool])
            else:
                Ll1 = l1_loss(image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=gt_mask_bool))

        if not res_controller.c2f_phase:

            if opt.fft_loss and fft_lowpass_sigma is not None:
                if fft_lowpass_mask is None or gt_image.shape[1:] != fft_lowpass_mask.shape or fft_mask_sigma != fft_lowpass_sigma:
                    fft_mask_sigma = fft_lowpass_sigma
                    fft_lowpass_mask = fftshift(gen_gaussian_ellipse_torch(gt_image.shape[2], gt_image.shape[1], sigma=fft_lowpass_sigma).to(gt_image.device).repeat(1, 3, 1, 1))
                    fft_loss_mask = (fft_lowpass_mask > 0.01)
                    fft_loss_coeffs = fft_lowpass_mask[fft_loss_mask]

                # if not hasattr(viewpoint_cam, "fft"):
                #     viewpoint_cam.fft = fft2(gt_image.unsqueeze(0))
                #
                # fft_gt = viewpoint_cam.fft[fft_loss_mask]

                fft_gt = fft2(gt_image.unsqueeze(0))[fft_loss_mask]
                fft_rendered = fft2(image.unsqueeze(0))[fft_loss_mask]

                ampl_gt = torch.abs(fft_gt)
                ampl = torch.abs(fft_rendered)
                ampl_loss = (torch.abs(ampl_gt - ampl) * fft_loss_coeffs).mean()

                phi_dist = calc_phase_dist(fft_gt, fft_rendered)

                phi_loss = (phi_dist * fft_loss_coeffs).mean()

                loss += opt.lambda_fft_ampl * ampl_loss + opt.lambda_fft_phi * phi_loss


            if opt.silhouette_loss:

                gt_mask_object = (gt_mask > 0.5).repeat(3, 1, 1)

                masked_gt = gt_image * gt_mask + bg[:, None, None] * (1 - gt_mask).squeeze()

                Ll1 = l1_loss(image[gt_mask_object], masked_gt[gt_mask_object])
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, masked_gt))

                if opt.silhouette_loss_type == "bce":
                    silhouette_loss = F.binary_cross_entropy(alpha_img, gt_mask)
                elif opt.silhouette_loss_type == "mse":
                    silhouette_loss = F.mse_loss(alpha_img, gt_mask)
                else:
                    raise NotImplementedError
                loss += opt.lambda_silhouette * silhouette_loss
                # if tb_writer is not None:
                #     tb_writer.add_scalar('loss/silhouette_loss', silhouette_loss, iteration)

            if viewpoint_cam.sky_mask is not None:
                sky_select = (~(viewpoint_cam.sky_mask.to(torch.bool))).cuda()
            else:
                sky_select = None

            if opt.skysphere_loss and sky_select is not None:
                skymask_prob = sky_select.to(torch.float32).clamp(0, 1).clamp(1e-6, 1 - 1e-6)
                rendered_skyness_prob = skyness.clamp(1e-6, 1 - 1e-6)

                splats_skyness_prob = gaussians.get_skysphere.squeeze().clamp(1e-6, 1 - 1e-6)

                skysphere_mask_loss = binary_cross_entropy(rendered_skyness_prob, skymask_prob)

                # real_xyz_world_dist = torch.norm(gaussians.get_xyz, dim=1)
                # splats_dist_gt_100 = (real_xyz_world_dist > 100).squeeze().to(torch.float32).clamp(0, 1).clamp(1e-6, 1 - 1e-6)
                #
                # skysphere_by_distance_loss = binary_cross_entropy(
                #     splats_skyness_prob,
                #     splats_dist_gt_100
                # )

                # minimization of crossentropy with itself = entropy minimization
                skysphere_entropy_loss = binary_cross_entropy(splats_skyness_prob, splats_skyness_prob)


                loss += (opt.lambda_skysphere_mask * skysphere_mask_loss
                         # + opt.lambda_skysphere_dist * skysphere_by_distance_loss
                         + opt.lambda_skysphere_entropy * skysphere_entropy_loss)


            if opt.skysphere_loss and opt.sky_depth_loss:
                update_loss_from_skydepth(gaussians, opt, loss)

            # if opt.sky_depth_loss:
            #     pass
            #
                # rendered_depth_sky = render_pkg['rendered_depth'].reshape(viewpoint_cam.sky_sphere_depthmap.shape)[sky_select]
                # rendered_alpha_sky = render_pkg['rendered_alpha'].reshape(viewpoint_cam.sky_sphere_depthmap.shape)[sky_select]
                # if sky_select is not None and viewpoint_cam.sky_sphere_depthmap is not None:
                #     dmap_rendered = rendered_depth_sky
                #     dmap_ideal = viewpoint_cam.sky_sphere_depthmap[sky_select].cuda()
                #     sky_depth_diff = torch.abs(torch.log(torch.clamp(dmap_rendered, 1)) - torch.log(torch.clamp(dmap_ideal, 1)))
                #     # ещё и на alpha завязать, чтобы оно пыталось "выключать" косячные пиксели через прозрачность,
                #     # и их потом prune зачистит
                #     # чем меньше прозрачность, тем меньше потеря
                #     sky_depth_diff *= torch.clamp(rendered_alpha_sky, 0, 1)
                #
                #     good_diff = sky_depth_diff > 1e-6
                #     l1_sky_depth = torch.clamp(sky_depth_diff[good_diff], 0, 1e10).mean()
                #
                #
                #     loss += opt.lambda_sky_depth * l1_sky_depth


            update_loss_from_splat_shape(gaussians, opt, loss)

            if opt.normal_loss:
                pass

                # rendered_depth = render_pkg["rendered_depth"]
                # rendered_normal = render_pkg['rendered_normal'] * 2 - 1
                # nnn = pseudo_normals_from_depthmap_gradient(rendered_depth.squeeze())

                # if viewpoint_cam.sky_mask is not None:
                #     world_mask = viewpoint_cam.sky_mask.cuda().to(torch.bool)

                # loss += 0.1 * (nnn[:, ~world_mask] - torch.tensor((0, 0, 1), device=nnn.device)[:, None]).norm().mean()
                # loss += 0.001 * (nnn[:, world_mask] - rendered_normal[:, world_mask]).norm().mean()


                # if viewpoint_cam.normal is not None:
                #     normal_gt = viewpoint_cam.normal.cuda()[:, world_mask]
                #
                #     l1_normal = torch.abs(nnn[:, world_mask] - normal_gt[[1, 2, 0], ...]).sum(dim=0).mean()
                #     # cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0)).mean()
                #     # loss += opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal
                #
                #     loss += opt.lambda_l1_normal * l1_normal


            if opt.mono_loss and hasattr(viewpoint_cam, "depth") and viewpoint_cam.depth is not None:

                render_mask = torch.where(alpha_img > 0.5, True, False)
                mask = render_mask
                if opt.skysphere_loss:
                    gt_mask = ~sky_select  # torch.where(viewpoint_cam.mask > 0.5, True, False)
                    mask &= gt_mask

                if torch.count_nonzero(mask) > 10:
                    moonodepth = viewpoint_cam.depth.cuda().unsqueeze(0)
                    rendered_depth = render_pkg["rendered_depth"]

                    depth_mask_mono = moonodepth[mask].clamp(1e-6)
                    depth_mask_render = rendered_depth[mask].clamp(1e-6)



                    if opt.mono_loss_type == "mid":
                        depth_loss = monodisp(1 / depth_mask_mono, 1 / depth_mask_render, 'l1')[-1]

                    elif opt.mono_loss_type == "pearson":

                        # depth_loss = torch.min(
                        #     (1 - pearson_corrcoef(- depth_mask_mono, depth_mask_render)),
                        #     (1 - pearson_corrcoef(1 / (depth_mask_mono + 200.), depth_mask_render))
                        # )

                        disp_mono = 1 / depth_mask_mono
                        disp_render = 1 / depth_mask_render
                        depth_loss = (1 - pearson_corrcoef(disp_render, -disp_mono)).mean()

                    elif opt.mono_loss_type == "grad":
                        depth_loss = loss_depth_grad(depth_mask_mono, depth_mask_render) + loss_depth_grad(depth_mask_mono, depth_mask_render)

                    else:
                        raise NotImplementedError

                    loss += opt.lambda_mono_depth * depth_loss

        else:
            sky_select = None

        if opt.opacity_reset_interval > 0 and opacity_was_reset_at_iter > 0:
            d_iter = iteration - opacity_was_reset_at_iter
            r_iter1 = opt.opacity_reset_interval / 20
            r_iter2 = opt.densification_interval * 2
            r_iter3 = len(res_controller.trainCameras_filter3d) * 2
            regenerating_opacity = d_iter < max(r_iter1, r_iter2, r_iter3)
            opacity_reset_phase = iteration > opt.opacity_reset_interval
        else:
            regenerating_opacity = False
            opacity_reset_phase = False
            d_iter = -1

        if opt.semitransparent_until_iter is not None:
            semi_iter_ok = opt.semitransparent_from_iter <= iteration < opt.semitransparent_until_iter
        else:
            semi_iter_ok = opt.semitransparent_from_iter <= iteration

        if opt.semitransparent_loss and semi_iter_ok and not regenerating_opacity:
            # полупрозрачные сплаты по возможности сделать или полностью прозрачными, или полностью непрозрачными
            # (а потом удалить полностью прозрачные в prune)
            # хорошо подходит для мелких объектов, но очень так-себе для уличных сцен с большой глубиной

            opacity = gaussians.get_opacity.clamp(1e-6, 1-1e-6)
            # minimization of crossentropy with itself = entropy minimization
            semitransparent_loss = binary_cross_entropy(opacity, opacity)
            loss += opt.lambda_semitransparent * semitransparent_loss


        if 0:
            scalingsz = gaussians.get_scaling.max(dim=1).values
            if sky_select is not None:
                world = (gaussians.get_skysphere < 0.4).squeeze()
                scalingsz = scalingsz[world]

            loss += 2 * ((scalingsz - scene.cameras_extent / 200).clamp(0) ** 2).mean()

        if 0:
            loss += 0.5 * (6-radii[radii>0]).float().clamp(0).mean()


        if 0:
            depth = render_pkg["rendered_depth"].squeeze()

            if not hasattr(viewpoint_cam, "canny_mask"):
                import kornia.filters
                magnitude, edges = kornia.filters.canny(gt_image.unsqueeze(0))

                viewpoint_cam.canny_mask = edges.squeeze() > 0.5

            canny_mask = viewpoint_cam.canny_mask

            depth_mask = (depth>0).detach().squeeze()
            nearDepthMean_map = nearMean_map(depth, canny_mask*depth_mask, kernelsize=3)
            loss += l2_loss(nearDepthMean_map[~sky_select], (depth*depth_mask)[~sky_select]) * .1



        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                params = {"Loss": f"{ema_loss_for_log:.{5}f}", "N": f"{gaussians._xyz.shape[0]}"}
                if opt.c2f:
                    params["σ"] = f"{kernel_size:.2f}"
                params["blur"] = res_controller.report
                if fft_lowpass_sigma is not None:
                    params["fft"] = f"{fft_lowpass_sigma:.2f}"
                if opt.semitransparent_loss:
                    params["r_o"] = f"{regenerating_opacity}"
                    params["r_d"] = f"{d_iter}"
                progress_bar.set_postfix(params)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), progress.testing_iterations, scene, render, (pipe, background))
            if iteration in progress.save_gaussians_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            N_unprocessed_fullstacks = N_full_stacks_done - N_full_stacks_processed

            # Densification
            if iteration < opt.densify_until_iter: # and N_unprocessed_fullstacks > 0:

                gaussians.statblock.add_densification_stats(
                    viewspace_point_tensor, visibility_filter,
                    radii=radii, n_touched=n_touched, n_dominated=n_dominated,
                    splat_depths=splat_depths,
                    fx=viewpoint_cam.focal_x,
                    total_px=total_px
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if opt.fft_loss:
                        d = iteration / opt.fft_lowpass_sigma_iter
                        if d <= 1.2:
                            fft_lowpass_sigma = opt.fft_lowpass_sigma_initial + d * (opt.fft_lowpass_sigma_max - opt.fft_lowpass_sigma_initial)
                        else:
                            fft_lowpass_sigma = None

                    res_scale = viewpoint_cam.image_width / 1920
                    size_threshold = 20 * res_scale if opacity_reset_phase and not res_controller.c2f_phase else None
                    min_opacity = 0.05 if not res_controller.c2f_phase else 0.01
                    if regenerating_opacity:
                        min_opacity = 0
                        size_threshold = None

                    outliers_sigma = None
                    # outliers_sigma = 5 if iteration > 8_000 else None
                    # if iteration % (opt.densification_interval*3) != 0:
                    #     # дать 3 раза расклонироваться, чтобы пустоту заполнили
                    #     outliers_sigma = None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold,
                        kl_threshold=opt.kl_threshold, skysphere_radius=opt.skysphere_radius,
                        kill_outliers=outliers_sigma,
                        largest_point_divider=opt.largest_point_divider,
                        smallest_point_divider=opt.smallest_point_divider * (1 + iteration / 1000)
                    )

                    gaussians.propagate_depth_sq_to_filter3D(res_controller.trainCameras_filter3d)

                    torch.cuda.empty_cache()

                if opt.opacity_reset_interval != -1 and (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()
                    opacity_was_reset_at_iter = iteration

                N_full_stacks_processed = N_full_stacks_done

            if iteration % 500 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 500:
                    # don't update in the end of training
                    gaussians.propagate_depth_sq_to_filter3D(res_controller.trainCameras_filter3d)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration in progress.save_checkoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path / f"chkpnt{iteration}.pth")

        scene.was_updated.set()
        scene.lock.release()
        # torch.cuda.synchronize()

    unload_cam_data(scene)



def update_loss_from_splat_shape(gaussians: GaussianModel, opt: OptimizationParams, loss):

    scales = gaussians.get_scaling[(gaussians.get_skysphere < 0.6).squeeze(), :]
    scales_sky = gaussians.get_scaling[(gaussians.get_skysphere >= 0.6).squeeze(), :]

    # flatten loss with anisotropy regularization
    if opt.flatten_loss or opt.flatten_aniso_loss:
        # нормаль должна быть значительно короче остальных осей...

        sorted_scales, sorted_indices = scales.sort(dim=1, descending=True)

        # Split sorted scales into s1 (largest), s2 (second largest), and s3 (smallest)
        s1, s2, s3 = sorted_scales.split(1, dim=1)

        # Compute flatten loss based on s3 (smallest scaling)
        flatten_loss = torch.clamp(s1 / s3 - 5, 0, 1e6).mean()

        if opt.flatten_aniso_loss:
            # ... но при этом сплаты не должны превращаться в иглы, а только в более-менее плоские диски

            # from https://arxiv.org/html/2401.15318v1 Eq. (6)
            # a = opt.aniso_ratio_threshold
            # aniso_loss = torch.clamp(s1 / s2 - a, 0, 1e6).mean()
            iso12_loss = torch.abs(scales[:2, :] - scales[:2, :].mean(dim=1).view(-1, 1)).mean()

            loss += opt.lambda_flatten * flatten_loss + opt.lambda_aniso * iso12_loss
        else:
            loss += opt.lambda_flatten * flatten_loss

    elif opt.isotropy_loss:
        # более-менее ровные  шарики, ну может слегка вытянутые по одной из осей

        # from https://arxiv.org/html/2312.06741v1 Eq. (14)
        #mean_scales = torch.mean(scales, dim=1, keepdim=True)
        #isotropy_loss = torch.norm(scales - mean_scales, p=1, dim=1).mean()
        isotropy_loss = torch.abs(scales - scales.mean(dim=1).view(-1, 1)).mean()

        loss += opt.lambda_iso * isotropy_loss

    # для неба - всегда ослабленная изотропная регуляризация
    if scales_sky.count_nonzero() > 0:
        isotropy_loss_sky = torch.abs(scales_sky - scales_sky.mean(dim=1).view(-1, 1)).mean()
        loss += opt.lambda_iso * isotropy_loss_sky

    return loss


def update_loss_from_skydepth(gaussians: GaussianModel, opt: OptimizationParams, loss):
    # приягиваем небо, отклоняющееся от скайсферы
    sky_by_attr = gaussians.get_skysphere[:, 0] >= 0.6

    real_xyz_sky_dist = torch.norm(gaussians.get_xyz[sky_by_attr], dim=1)
    sky_depth_diff = ((real_xyz_sky_dist - opt.skysphere_radius).abs().clamp(1e-6) ** 0.5)

    if sky_depth_diff.shape[0] > 0:
        l1_sky_depth = sky_depth_diff.mean()
        loss += opt.lambda_sky_depth * l1_sky_depth

    # гасим НЕ небо дальше 1/3 радиуса скайсферы от начала мира
    world_by_attr = gaussians.get_skysphere[:, 0] < 0.4

    opa_world = gaussians.get_opacity[world_by_attr].squeeze()
    real_xyz_world_dist = torch.norm(gaussians.get_xyz[world_by_attr], dim=1)
    world_depth_diff = ((real_xyz_world_dist - opt.skysphere_radius / 3).clamp(0) ** 2)
    world_depth_diff *= opa_world

    if world_depth_diff.shape[0] > 0:
        l1_world_depth = world_depth_diff.mean()
        loss += opt.lambda_sky_depth * l1_world_depth

    return loss


def prepare_output_and_logger(conf: GaussianSplattingConf):
    model_path = conf.model_params.model_path
    if not model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        conf.model_params.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(model_path))
    os.makedirs(model_path, exist_ok = True)

    OmegaConf.save(conf, Path(model_path) / "cfg_args.yaml")
    # with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
    #     cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def main():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", required=True, help="path to the yaml config file")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")

    args, extras = parser.parse_known_args()

    try:
        schema = OmegaConf.structured(GaussianSplattingConf)
        conf_default: GaussianSplattingConf = OmegaConf.load("configs/gaussian-object.yaml")
        conf_default = OmegaConf.merge(schema, conf_default)

        conf_user: GaussianSplattingConf = OmegaConf.load(args.config, )
        conf = OmegaConf.merge(conf_default, conf_user)
    except omegaconf.errors.OmegaConfBaseException as e:
        print(e, file=sys.stderr)
        sys.exit(-1)

    op = conf.optimization_params
    lp = conf.model_params
    pg = conf.progress_params

    if op.iterations not in pg.save_gaussians_iterations:
        pg.save_gaussians_iterations.append(op.iterations)

    if op.iterations not in pg.testing_iterations:
        pg.testing_iterations.append(op.iterations)

    print("Optimizing " + lp.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    from tiny_renderer.minigui import SimpleGUI

    gui = SimpleGUI()

    t0 = threading.Thread(target=t0_fn, args=(gui,))
    t0.start()

    # Start GUI server, configure and run training
    # network_gui = NetworkGUI(args.ip, args.port)
    network_gui = None
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(conf, args.debug_from, network_gui, gui)

    # All done
    print("\nTraining complete.")


def t0_fn(gui):
    gui.run()

if __name__ == "__main__":
    main()
