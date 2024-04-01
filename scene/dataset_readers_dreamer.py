from pathlib import Path

from PIL import Image
import cv2
import numpy as np

from scene.dataset_readers import CameraInfo, getNerfppNorm, SceneInfo
from utils.graphics_utils import BasicPointCloud


def _calc_rays(res_x, res_y, K):
    # (0,0,0) -> Z=1.0

    xx, yy = np.meshgrid(
        np.arange(0, res_x),
        np.arange(0, res_y)
    )

    rays = np.dstack((xx, yy)).reshape(-1, 2)
    rays = np.hstack((rays, np.ones((rays.shape[0], 1))))
    rays = np.linalg.inv(K) @ rays.T
    rays = rays.T
    return rays.astype(np.float32)


def readDREAMER(path: str | Path, eval, llffhold=8, load_skymask=False, N_random_init_pts=True):
    path = Path(path).absolute()

    # Initialize camera and point cloud information lists
    train_cam_infos = []
    test_cam_infos = []
    all_points = []

    from utils.pcpr_utils import PCPRRenderer

    ids_to_load = [0,]
    ids_loaded = set()

    all_points = []
    all_normals = []
    all_colors = []

    load_ply = not N_random_init_pts > 0


    for frame_id in ids_to_load:

            images_path = path / "images"

            img_path = images_path / f"{frame_id:04d}.jpg"

            if not img_path.is_file(): continue
            if frame_id not in ids_to_load: continue
            ids_loaded.add(frame_id)

            masks_path = path / "skymask"

            skymask = None
            if load_skymask and masks_path:
                mask_fname = (masks_path / img_path.stem).with_suffix(".png")
                if mask_fname.is_file():
                    skymask = cv2.imread(str(mask_fname), cv2.IMREAD_GRAYSCALE)

            monodepth_path = path / "monodepth"

            monodepth = None
            # if load_depth and masks_path:
            monodepth_fname = (monodepth_path / img_path.stem).with_suffix(".png")
            if monodepth_fname.is_file():
                monodepth = cv2.imread(str(monodepth_fname), cv2.IMREAD_GRAYSCALE)

                monodepth = ((1 - (monodepth / 255)) * 40 + 20).astype(np.float32)


            image = Image.open(img_path)

            # w2c = inv(cam_pose)
            # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            # T = w2c[:3, 3]

            w_orig = image.width
            f = w_orig * 1.5

            h_orig = image.height
            K = np.array([
                [f, 0, w_orig / 2],
                [0, f, h_orig / 2],
                [0, 0, 1]
            ])
            R = np.eye(3)
            T = np.zeros(3)

            cam_name = img_path.relative_to(path / "images").with_suffix("").as_posix()
            cam_info = CameraInfo(uid=frame_id, K=K, R=R, T=T, image=image,
                                  image_path=str(img_path.absolute()), image_name=cam_name, width=w_orig, height=h_orig,
                                  sky_mask=skymask, depth=monodepth)

            # if waymo_cam == "FRONT":
            # # Split into training and testing based on llffhold
            if frame_id % llffhold == 0:
                test_cam_infos.append(cam_info)
            # else:
            train_cam_infos.append(cam_info)

    nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization["radius"] = 20

    # if load_ply:

    # colors = np.concatenate(all_colors, axis=0)
    # normals = np.concatenate(all_normals, axis=0)

    # pp = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pts))
    # pp = pp.voxel_down_sample(0.2)
    #
    # combined = np.ascontiguousarray(pp.points, dtype=np.float32)

    # pcd_all = BasicPointCloud(
    #     points=np.ascontiguousarray(pts, dtype=np.float32),
    #     normals=np.ascontiguousarray(normals, dtype=np.float32),
    #     colors=np.ascontiguousarray(colors, dtype=np.float32),
    # )
    # pcd_all = BasicPointCloud(points=np.zeros((1, 3)), normals=np.zeros((1,3)), colors=np.zeros((1,3)))

    # pp = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pcd_all.points))
    # pp.colors = open3d.utility.Vector3dVector(pcd_all.colors)
    # open3d.io.write_point_cloud("naka.ply", pp)
    # print(1)

    if N_random_init_pts > 0:
        num_pts = N_random_init_pts

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

    else:
        pts = _calc_rays(monodepth.shape[1], monodepth.shape[0], K).reshape(monodepth.shape[0], monodepth.shape[1], 3) * monodepth[..., None]

        sparsity = 0.02
        random_mask = np.random.rand(monodepth.shape[0], monodepth.shape[1]) < sparsity
        random_mask = random_mask[:888, :888]
        random_mask &= np.bool_(skymask[:888, :888])

        pts = pts[:888, :888][random_mask, :].reshape(-1, 3)

        num_pts = pts.shape[0]

        normals = np.zeros((num_pts, 3), dtype=np.float32)
        colors = np.full((num_pts, 3), fill_value=0.5, dtype=np.float32)

        pcd_all = BasicPointCloud(
            points=np.ascontiguousarray(pts, dtype=np.float32),
            normals=np.ascontiguousarray(normals, dtype=np.float32),
            colors=np.ascontiguousarray(colors, dtype=np.float32),
        )


    ply_path = r"X:\_ai\_waymo\tensorflow_extractor\colmap_proj-single\ZZinput.ply"

    scene_info = SceneInfo(point_cloud=pcd_all,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info
