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

import math
import os, sys
from pathlib import Path
from typing import Optional
from random import randint
import uuid
from argparse import ArgumentParser, Namespace

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim, binary_cross_entropy
from gaussian_renderer import render
from gaussian_renderer.network_gui import NetworkGUI
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from arguments import OptimizationParams, GaussianSplattingConf
from utils.skysphere_utils import add_skysphere_points3d

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(conf: GaussianSplattingConf, debug_from,
             network_gui: Optional[NetworkGUI]):

    opt = conf.optimization_params
    dataset = conf.model_params
    pipe = conf.pipeline_params
    progress = conf.progress_params

    if opt.c2f:
        opt.divide_ratio = 0.7
        splat_sigma = opt.c2f_max_sigma
        c2f_phase = True
    else:
        dataset.N_random_init_pts = -1
        splat_sigma = opt.default_sigma
        c2f_phase = False

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

    if not opt.skysphere_radius > 0 and opt.skysphere_radius_in_cam_extents > 0:
        opt.skysphere_radius = float(scene.cameras_extent * opt.skysphere_radius_in_cam_extents)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    with torch.no_grad():
        if dataset.load_skymask:
            add_skysphere_points3d(scene, gaussians, opt.skysphere_radius)


    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        # if network_gui is not None:
        #     with torch.no_grad():
        #         network_gui.tick(opt, pipe, dataset, gaussians, iteration, background)


        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if not c2f_phase and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        #render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=args.normal_loss)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,
                            return_normal=opt.normal_loss and not c2f_phase,
                            return_skyness=dataset.load_skymask and not c2f_phase, kernel_size=kernel_size)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # # opacity mask
        # if False: # iteration < opt.propagated_iteration_begin and opt.depth_loss:
        #     opacity_mask = render_pkg['rendered_alpha'] > 0.999
        #     opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)
        # else:
        #     opacity_mask = render_pkg['rendered_alpha'] > 0.0
        #     opacity_mask = opacity_mask.unsqueeze(0).repeat(3, 1, 1)

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        # Ll1 = l1_loss(image[opacity_mask], gt_image[opacity_mask])
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=opacity_mask))

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if viewpoint_cam.sky_mask is not None:
            sky_select = (~(viewpoint_cam.sky_mask.to(torch.bool))).cuda()
        else:
            sky_select = None

        if opt.skysphere_loss and sky_select is not None and not c2f_phase:
            skymask_prob = sky_select.to(torch.float32).clamp(0, 1).clamp(1e-6, 1 - 1e-6)
            rendered_skyness_prob = render_pkg["rendered_skyness"].reshape(sky_select.shape).clamp(1e-6, 1 - 1e-6)

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


        if opt.skysphere_loss and opt.sky_depth_loss and not c2f_phase:
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

        if opt.semitransparent_loss and not c2f_phase:
            # полупрозрачные сплаты по возможности сделать или полностью прозрачными, или полностью непрозрачными
            # (а потом удалить полностью прозрачные в prune)
            # хорошо подходит для мелких объектов, но очень так-себе для уличных сцен с большой глубиной

            opacity = gaussians.get_opacity.clamp(1e-6, 1-1e-6)
            # minimization of crossentropy with itself = entropy minimization
            semitransparent_loss = binary_cross_entropy(opacity, opacity)
            loss += opt.lambda_semitransparent * semitransparent_loss

        if opt.normal_loss and not c2f_phase:
            rendered_normal = render_pkg['rendered_normal']
            if viewpoint_cam.normal is not None:
                normal_gt = viewpoint_cam.normal.cuda()
                if viewpoint_cam.sky_mask is not None:
                    filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
                    normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
                loss += opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "σ": f"{splat_sigma:.2f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), progress.testing_iterations, scene, render, (pipe, background))
            if iteration in progress.save_gaussians_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and not c2f_phase and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval and not c2f_phase else None
                    min_opacity = 0.05 if not c2f_phase else 0.01
                    gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold,
                                                kl_threshold=opt.kl_threshold, skysphere_radius=opt.skysphere_radius)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration in progress.save_checkoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path / f"chkpnt{iteration}.pth")


def update_loss_from_splat_shape(gaussians: GaussianModel, opt: OptimizationParams, loss):

    # flatten loss with anisotropy regularization
    if opt.flatten_loss or opt.flatten_aniso_loss:
        # нормаль должна быть значительно короче остальных осей...

        scales = gaussians.get_scaling
        sorted_scales, sorted_indices = scales.sort(dim=1, descending=True)

        # Split sorted scales into s1 (largest), s2 (second largest), and s3 (smallest)
        s1, s2, s3 = sorted_scales.split(1, dim=1)

        # Compute flatten loss based on s3 (smallest scaling)
        min_scale = s3
        flatten_loss = torch.abs(min_scale).mean()

        if opt.flatten_aniso_loss:
            # ... но при этом сплаты не должны превращаться в иглы, а только в более-менее плоские диски

            # from https://arxiv.org/html/2401.15318v1 Eq. (6)
            a = opt.aniso_ratio_threshold
            aniso_loss = torch.clamp(s1 / s2 - a, 0, 1e6).mean()

            loss += opt.lambda_flatten * flatten_loss + opt.lambda_aniso * aniso_loss
        else:
            loss += opt.lambda_flatten * flatten_loss

    elif opt.isotropy_loss:
        # более-менее ровные  шарики, ну может слегка вытянутые по одной из осей
        scales = gaussians.get_scaling

        # from https://arxiv.org/html/2312.06741v1 Eq. (14)
        mean_scales = torch.mean(scales, dim=1, keepdim=True)
        isotropy_loss = torch.norm(scales - mean_scales, p=1, dim=1).mean()

        loss += opt.lambda_iso * isotropy_loss
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

    # Start GUI server, configure and run training
    network_gui = NetworkGUI(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(conf, args.debug_from, network_gui)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
