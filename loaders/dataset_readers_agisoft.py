from pathlib import Path

from PIL import Image
import PIL.Image
import cv2
import numpy as np
import open3d
from plyfile import PlyData

from loaders.dataset_readers import CameraInfo, getNerfppNorm, SceneInfo
from utils.graphics_utils import BasicPointCloud


inv = np.linalg.inv

xform = np.array([
    [1,  0,  0, 0],
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [0,  0,  0, 1],
], dtype=np.float32)

# 	xform_pts = np.concatenate((pts1, np.ones((pts1.shape[0], 1))), axis=1) @ xform.T pts_world = xform_pts[:, :3]
# 	xform_normals = np.concatenate((normals1, np.zeros((normals1.shape[0], 1))), axis=1) @ xform.T normals_world = xform_normals[:, :3]

def readAgisoftExportInfo(path: str | Path, eval, llffhold=8, load_skymask=False, N_random_init_pts=None):

    path = Path(path)

    train_cam_infos = []
    test_cam_infos = []

    # cams_to_show = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    camera_angles_ALL = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
    cams_to_show = ["SIDE_LEFT"]

    agisoft_cams = {
        "SIDE_LEFT": 0,
        "FRONT_LEFT": 1,
    }


    calibs, poses = read_agisoft_xml(path / "cameras.xml")

    for frame_id in range(50):

        for waymo_cam in cams_to_show:
            img_path = path / waymo_cam / f"{frame_id:04d}.jpg"

            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            # img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            waymo_to_agisoft_I = agisoft_cams[waymo_cam]

            w, h, K, d = calibs[waymo_to_agisoft_I]

            img = cv2.undistort(img, K, d)

            C2W_xform0 = poses[(frame_id, waymo_to_agisoft_I)]

            # C2W_xform0 = xform @ C2W_xform0

            W2C_xform = inv(C2W_xform0)

            skymask = None

            monodepth = None
            # monodepth_path = path / "monodepth" / waymo_cam / f"{frame_id:04d}.png"
            # if monodepth_path.is_file():
            #     monodepth = cv2.imread(str(monodepth_path), cv2.IMREAD_GRAYSCALE) / 256
            #
            #
            #     def _calc_rays(res_x, res_y, K):
            #         # (0,0,0) -> Z=1.0
            #
            #         xx, yy = np.meshgrid(
            #             np.arange(0, res_x),
            #             np.arange(0, res_y)
            #         )
            #
            #         rays = np.dstack((xx, yy)).reshape(-1, 2)
            #         rays = np.hstack((rays, np.ones((rays.shape[0], 1))))
            #         rays = np.linalg.inv(K) @ rays.T
            #         rays = rays.T
            #         return rays.astype(np.float32).reshape(h, w, 3)
            #
            #     from modules.filters.zmap_curvature import calc_normals_and_curvature_covariance
            #
            #     rays_map = _calc_rays(w, h, K)
            #
            #     points3d_map = (rays_map * (monodepth + 1)[..., None])
            #     curv, normals, dot = calc_normals_and_curvature_covariance(points3d_map, max_grazing_angle_deg=90, min_d=1., max_d=10, half_w=10)
            #     from modules.common_img import turbo_img
            #     import cv2
            #     from math_funcs import ocv_to_gl
            #
            #     frame = ocv_to_gl(normals)
            #     normals_map_bgr = ((frame[:, :] * 128) + 128)
            #     normals_map_bgr[normals_map_bgr[:, :, :] > 255] = 255
            #     normals_map_bgr[normals_map_bgr[:, :, :] < 0] = 0
            #     normals_map_bgr = normals_map_bgr.astype(np.uint8)
            #     cv2.imwrite("normals.png", normals_map_bgr)
            #
            #
            #     turbo_img("curv.png", curv)
            #     print(1)

            R = W2C_xform[:3, :3]
            T = W2C_xform[:3, 3]

            image = PIL.Image.fromarray(img[..., ::-1])


            cam_name = img_path.relative_to(path).with_suffix("").as_posix()
            cam_info = CameraInfo(uid=frame_id*10+camera_angles_ALL.index(waymo_cam)+1, K=K, R=R, T=T, image=image,
                                  image_path=str(img_path.absolute()), image_name=cam_name, width=w, height=h,
                                  sky_mask=skymask, depth=monodepth)

            # if waymo_cam == "FRONT":
            # # Split into training and testing based on llffhold
            if frame_id % llffhold == 0:
                test_cam_infos.append(cam_info)
            # else:
            train_cam_infos.append(cam_info)


    nerf_normalization = getNerfppNorm(train_cam_infos)


    load_ply = not N_random_init_pts > 0

    if load_ply:
        ply_path = path / f"dump.ply"

        plydata = PlyData.read(ply_path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 256

        # xform_pts = np.concatenate((positions, np.ones((positions.shape[0], 1))), axis=1) @ xform.T
        # positions = xform_pts[:, :3]

        pcd_all = BasicPointCloud(
            points=np.ascontiguousarray(positions, dtype=np.float32),
            normals=np.ascontiguousarray(np.zeros_like(positions), dtype=np.float32),
            colors=np.ascontiguousarray(colors, dtype=np.float32),
        )

        pp = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pcd_all.points))
        pp.colors = open3d.utility.Vector3dVector(pcd_all.colors)
        open3d.io.write_point_cloud(str(path / "ZZinput.ply"), pp)
        print(1)

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


    ply_path = str(path / "ZZinput.ply")

    scene_info = SceneInfo(point_cloud=pcd_all,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info


def read_agisoft_xml(p):
    import lxml.etree

    tree = lxml.etree.parse(p)

    sensors = tree.xpath("/document/chunk/sensors/sensor")

    calibs = {}
    for sensor in sensors:
        sensor_id = int(sensor.attrib["id"])

        xpath0 = sensor.xpath("./calibration[@class='adjusted']")
        if not xpath0:
            xpath0 = sensor.xpath("./calibration[@class='initial']")
        calibration = xpath0[0]

        w = calibration.find("resolution").attrib["width"]
        h = calibration.find("resolution").attrib["height"]

        f = calibration.find("f").text
        cx = calibration.find("cx").text
        cy = calibration.find("cy").text
        k1 = calibration.find("k1").text
        k2 = calibration.find("k2").text
        try:
            k3 = calibration.find("k3").text
        except AttributeError:
            k3 = 0
        p1 = calibration.find("p1").text
        p2 = calibration.find("p2").text

        K = np.array([
            [float(f), 0, float(w) / 2 + float(cx)],
            [0, float(f), float(h) / 2 + float(cy)],
            [0, 0, 1]
        ])

        d = np.array([float(v) for v in (k1, k2, p1, p2, k3)])

        calibs[sensor_id] = (int(w), int(h), K, d)

    poses = {}
    for camera in tree.xpath("/document/chunk/cameras/camera"):
        sensor_id = int(camera.attrib["sensor_id"])
        frame_id = int(camera.attrib["label"].lstrip("0") or 0)
        xform = np.array([float(v) for v in camera.xpath("./transform")[0].text.split()]).reshape(4, 4)
        # print(camera)
        poses[(frame_id, sensor_id)] = xform

    return calibs, poses
