from pathlib import Path

from PIL import Image
import cv2
import numpy as np
import open3d
import torch

from scene.dataset_readers import CameraInfo, fetchPly, getNerfppNorm, SceneInfo
from utils.graphics_utils import focal2fov, BasicPointCloud
from utils.waymo_utils import WaymoCoordsHelper


inv = np.linalg.inv


def readWaymoExportInfo(path: str | Path, eval, llffhold=8, load_skymask=False, SLV_initial_gaussians=True):
    """
    Reads information from a Waymo dataset export.
    The dataset is expected to have images, lidar, and optional skymask folders.
    Each of these folders contains subfolders for different camera angles.
    """

    path = Path(path).absolute()

    # Initialize camera and point cloud information lists
    train_cam_infos = []
    test_cam_infos = []
    all_points = []

    camera_angles_ALL = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    camera_angles = ["FRONT",]

    from utils.pcpr_utils import PCPRRenderer

    waymo_conv = WaymoCoordsHelper(path)

    ids_to_load = range(0, 197)
    ids_loaded = set()

    all_points = []
    all_normals = []
    all_colors = []

    load_ply = not SLV_initial_gaussians > 0

    lidar_path = path / "lidar"

    for frame_id in ids_to_load:

        if load_ply:
            ply_path = lidar_path / f"{frame_id:04}.ply"
            pcd = fetchPly(ply_path)
            points3d = torch.from_numpy(waymo_conv.points_waymo2ocv(waymo_conv.points_frame2fr0(frame_id, pcd.points))).cuda()


        for waymo_cam in camera_angles:

            images_path = path / "images" / waymo_cam

            img_path = images_path / f"{frame_id:04d}.png"

            if not img_path.is_file(): continue
            if frame_id not in ids_to_load: continue
            ids_loaded.add(frame_id)

            masks_path = path / "skymask" / waymo_cam if load_skymask else None

            skymask = None
            if load_skymask and masks_path:
                mask_path = (masks_path / img_path.stem).with_suffix(".npy")
                if mask_path.is_file():
                    #skymask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    skymask = np.load(mask_path)

            w, h, K, d, cam_pose = waymo_conv.get_calib_as_ocv_rel_fr0(frame_id, waymo_cam)

            # assert w == image.shape[1] and h == image.shape[0]
            print("T:", np.round(cam_pose[:3, 3], 3))

            # image_new, K_new = pad_offcenter_img(image, w, h, K)
            # image_new = cv2.cvtColor(image_new, cv2.COLOR_BGRA2RGBA)
            # image_new = Image.fromarray(image_new)

            image_new = Image.open(img_path)
            K_new = K

            w2c = inv(cam_pose)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            fx = K_new[0, 0]
            fy = K_new[1, 1]
            cx = K_new[0, 2]
            cy = K_new[1, 2]
            intrin = np.array((fx, fy, cx, cy), dtype=np.float32)

            new_w = image_new.width
            new_h = image_new.height

            HFov = focal2fov(fx, new_w)
            VFov = focal2fov(fy, new_h)

            # if waymo_cam ==  "FRONT":
            cam_name = img_path.relative_to(path / "images").with_suffix("").as_posix()
            cam_info = CameraInfo(uid=frame_id*10+camera_angles_ALL.index(waymo_cam)+1, R=R, T=T, FovY=VFov, FovX=HFov, image=image_new,
                                  image_path=str(img_path.absolute()), image_name=cam_name, width=new_w, height=new_h,
                                  intrinsics=intrin, sky_mask=skymask)

            #
            #         for waymo_cam, viewpoint_cam in ffzz.items():
            #             viewpoint_cam: Camera
            #
            #             # c2w_xformW = waymo.get_calib_as_ocv_rel_fr0(frame_idx=frame_id, cam=waymo_cam)[-1]
            #             # w2c_xformW = torch.tensor(np.linalg.inv(c2w_xformW)).float()
            #
            #             w2c_xform = torch.eye(4)
            #             w2c_xform[:3, :3] = torch.tensor(viewpoint_cam.R.T) # rasterizer GLM
            #             w2c_xform[:3, 3] = torch.tensor(viewpoint_cam.T)
            #
            #             w2c_xform = torch.inverse(w2c_xform)
            #
            #             w = viewpoint_cam.image_width
            #             h = viewpoint_cam.image_height
            #             K = viewpoint_cam.K
            #

            if load_ply:
                lidar_depthmap, index_map_coarse = PCPRRenderer.render_depthmap(
                    points3d, w, h, torch.tensor(K), torch.tensor(cam_pose), max_splatting_size=0.1)

                # from modules.common_img import turbo_img
                # turbo_img(f"lidar_{frame_id:04d}_{waymo_cam}.png", lidar_depthmap.detach().cpu().numpy())

                lidar_sparsity = 0.015
                lidar_mask = (index_map_coarse != 0).cuda()

                random_mask = torch.rand(lidar_mask.shape, device=lidar_mask.device)
                lidar_mask_to_propagate = lidar_mask & (random_mask >= (1-lidar_sparsity))

                pts_idx_to_propagate = index_map_coarse[lidar_mask_to_propagate]
                pt111 = points3d[pts_idx_to_propagate.cpu()].cpu().numpy()
                # colors111 = np.asarray(image)[..., ::-1][lidar_mask_to_propagate.cpu()].astype(np.float32) / 255

                all_points.append(pt111)
                all_normals.append(np.zeros_like(pt111))
                # all_colors.append(colors111)
                all_colors.append(np.full_like(pt111, fill_value=0.5))

            # if waymo_cam == "FRONT":
            # # Split into training and testing based on llffhold
            if frame_id % llffhold == 0:
                test_cam_infos.append(cam_info)
            # else:
            train_cam_infos.append(cam_info)

    # lidar_path = path / "lidar"
    # for ply_path in lidar_path.glob("*.ply"):
    #     stem = ply_path.stem
    #     if not stem.isdigit():
    #         continue
    #     pcd_id = int(stem.lstrip("0") or "0")
    #     if pcd_id not in ids_loaded:
    #         continue
    #
    #     pcd = fetchPly(ply_path)
    #
    #     all_points.append(waymo_conv.points_frame2fr0(pcd_id, pcd.points))
    #     all_normals.append(pcd.normals)
    #     all_colors.append(pcd.colors)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = r"X:\_ai\_waymo\tensorflow_extractor\colmap_proj-single\ZZinput.ply"

    if load_ply:
        pts = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        normals = np.concatenate(all_normals, axis=0)

        # pp = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pts))
        # pp = pp.voxel_down_sample(0.2)
        #
        # combined = np.ascontiguousarray(pp.points, dtype=np.float32)

        pcd_all = BasicPointCloud(
            points=np.ascontiguousarray(pts, dtype=np.float32),
            normals=np.ascontiguousarray(normals, dtype=np.float32),
            colors=np.ascontiguousarray(colors, dtype=np.float32),
        )
        # pcd_all = BasicPointCloud(points=np.zeros((1, 3)), normals=np.zeros((1,3)), colors=np.zeros((1,3)))

        pp = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pcd_all.points))
        pp.colors = open3d.utility.Vector3dVector(pcd_all.colors)
        open3d.io.write_point_cloud("naka.ply", pp)
        print(1)

    if SLV_initial_gaussians > 0:
        num_pts = SLV_initial_gaussians

        cam_pos = []
        for k in train_cam_infos:
            C = -np.matmul(k.R, k.T)
            cam_pos.append(C)
        cam_pos = np.array(cam_pos)
        min_cam_pos = np.min(cam_pos)
        max_cam_pos = np.max(cam_pos)
        mean_cam_pos = (min_cam_pos + max_cam_pos) / 2.0
        cube_mean = (max_cam_pos - min_cam_pos) * 1.5

        max_coord = (max_cam_pos - min_cam_pos) * 3 - (cube_mean - mean_cam_pos)
        xyz = np.random.random((num_pts, 3)) * max_coord
        print(f"Generating SLV point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3))
        pcd_all = BasicPointCloud(points=xyz, colors=shs, normals=np.zeros((num_pts, 3)))


    scene_info = SceneInfo(point_cloud=pcd_all,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info
