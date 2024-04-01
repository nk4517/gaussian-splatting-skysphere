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
from pathlib import Path
import random
import json

import torch

from scene.gaussian_model_mip import GaussianModel
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model_w_save import save_gaussians_ply, load_gaussians_ply, init_gaussians_from_pcd
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel,
                 load_iteration: int | str | Path | None = None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = Path(args.model_path)
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(self.model_path / "point_cloud")
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        source_path = Path(args.source_path)

        elif args.waymo_calibs: # and (source_path / "calibs.dump").is_file() and (source_path / "images").is_dir():
            from scene.dataset_readers_waymo import readWaymoExportInfo

            scene_info = readWaymoExportInfo(args.source_path, args.eval,
                                             load_skymask=args.load_skymask, N_random_init_pts=args.N_random_init_pts)
        elif args.dreamer_calibs: # and (source_path / "calibs.dump").is_file() and (source_path / "images").is_dir():
            from scene.dataset_readers_dreamer import readDREAMER
            scene_info = readDREAMER(args.source_path, args.eval,
                                     load_skymask=args.load_skymask, N_random_init_pts=args.N_random_init_pts)
        elif (source_path / "sparse").is_dir():
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                          load_mask=args.load_mask, load_skymask=args.load_skymask,
                                                          load_normal=args.load_normal, load_depth=args.load_depth,
                                                          N_random_init_pts=args.N_random_init_pts)
        elif (source_path / "transforms_train.json").is_file():
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(self.model_path / "input.ply" , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(self.model_path / "cameras.json", 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if isinstance(self.loaded_iter, int):
                ply_fname = self.model_path / "point_cloud" / f"iteration_{self.loaded_iter}" / "point_cloud.ply"
            elif isinstance(self.loaded_iter, str) or isinstance(self.loaded_iter, Path):
                ply_fname = Path(self.loaded_iter)
            else:
                raise NotImplementedError("loaded_iter: " + repr(self.loaded_iter))

            load_gaussians_ply(self.gaussians, ply_fname)
        else:
            init_gaussians_from_pcd(self.gaussians, scene_info.point_cloud, self.cameras_extent * args.spatial_scaling_lr_mult)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        save_gaussians_ply(self.gaussians, os.path.join(point_cloud_path, "point_cloud.ply"))

        have_sky = bool(torch.any(self.gaussians.get_skysphere > 0.6))
        if have_sky:
            save_gaussians_ply(self.gaussians, os.path.join(point_cloud_path, "point_cloud_wo_sky.ply"), save_sky=False)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]