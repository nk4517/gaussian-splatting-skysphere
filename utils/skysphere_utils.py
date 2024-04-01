import math

import torch

from scene import Scene, GaussianModel
from scene.cameras import project_points_to_image


def add_skysphere_points3d(scene: Scene, gaussians: GaussianModel, skysphere_radius: float, full_skysphere_points: int = 50_000):
    cameras = scene.getTrainCameras()

    skysphere_pts3d = fibonacci_sphere(samples=full_skysphere_points, radius=skysphere_radius).float().to("cuda")

    populated = torch.zeros(skysphere_pts3d.shape[0], dtype=torch.bool, device="cuda")

    # import open3d
    # pp = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(skysphere_pts3d))
    # open3d.io.write_point_cloud(str("sky.ply"), pp)

    for cam in cameras:
        from scene.cameras import Camera
        cam: Camera

        if cam.sky_mask is not None:
            # sky_depthmap = compute_skysphere_depth_map(
            #     cam.K, cam.image_width, cam.image_height,
            #     cam.R, cam.T,
            #     skysphere_radius=skysphere_radius
            # )

            camera_2d, camera_3d, in_view = project_points_to_image(skysphere_pts3d, cam)

            # in_view_fru = cam.points_inside_frustrum(skysphere_pts3d)

            visible_unpopulated = in_view & ~populated
            visible_unpopulated_xy = camera_2d[visible_unpopulated]
            un_y = visible_unpopulated_xy[:, 1]
            un_x = visible_unpopulated_xy[:, 0]

            sky_ok = ~cam.sky_mask.cuda().to(torch.bool)

            visible_unpopulated_confirmed_sky = sky_ok[un_y, un_x]
            populated[visible_unpopulated] = visible_unpopulated_confirmed_sky

            un2add = torch.zeros_like(visible_unpopulated)
            un2add[visible_unpopulated] = visible_unpopulated_confirmed_sky

            un2add_y = un_y[visible_unpopulated_confirmed_sky]
            un2add_x = un_x[visible_unpopulated_confirmed_sky]

            pts3d_to_add = skysphere_pts3d[un2add]
            colors_to_add = cam.original_image[:, un2add_y, un2add_x].permute(1, 0)

            # turbo_img(f"sky_fib_{cam.uid:04d}.png", skymask_to_propagate.cpu().numpy(), norm_min=0, norm_max=1)

            params = gaussians.params_from_points3d(pts3d_to_add, colors_to_add, None, skyness=1)
            if params is not None:
                gaussians.densification_postfix(*params)


def compute_skysphere_depth_map(K: torch.Tensor, width, height, c2w_R: torch.Tensor, c2w_t: torch.Tensor,
                                skysphere_radius: float = 300, sphere_center: torch.Tensor | None = None):
    """
    Computes depth map for a sphere with an arbitrary center in world coordinates, observed by an arbitrary camera.

    Args:
    - K (torch.Tensor): Camera's intrinsic matrix of size (3, 3).
    - width (int): Image width.
    - height (int): Image height.
    - R (torch.Tensor): Camera rotation matrix of size (3, 3).
    - t (torch.Tensor): Camera translation vector of size (3,).
    - sky_sphere_radius (float): Sky sphere radius.
    - sphere_center_world (np.array): Sphere center in world coordinates.

    Returns:
    - depth_map (torch.Tensor): Sphere's depth map of size (height, width).
    """

    import kornia

    if sphere_center is None:
        sphere_center = torch.zeros(3, dtype=torch.float32)

    K = K.to(torch.float32).cuda()
    c2w_R = c2w_R.to(torch.float32).to(K.device)
    c2w_t = c2w_t.to(torch.float32).to(K.device)
    sphere_center = sphere_center.to(torch.float32).to(K.device)

    # Transform the sphere center to camera coordinates
    R_inv = torch.linalg.inv(c2w_R)
    sphere_center_cam = torch.matmul(R_inv, sphere_center - c2w_t)

    # Camera position in camera coordinates is the origin
    cam_position_cam = torch.tensor([0, 0, 0], device=K.device, dtype=torch.float32)

    # Generating ray coordinate grid
    grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=True).to(K.device)
    grid = kornia.geometry.denormalize_pixel_coordinates(grid, height, width).reshape(-1, 2)
    ones = torch.ones(grid.shape[0], 1, device=K.device)
    normalized_pixels = torch.cat([grid, ones], dim=-1)
    rays_camera = torch.matmul(torch.inverse(K), normalized_pixels.T)

    # Calculating ray intersections with the sphere in camera coordinates
    SCO = (cam_position_cam - sphere_center_cam).unsqueeze(1)
    D = rays_camera
    A = torch.sum(D ** 2, dim=0)
    B = 2 * torch.sum(D * SCO, dim=0)
    C = torch.sum(SCO ** 2, dim=0) - skysphere_radius ** 2
    discriminant = B ** 2 - 4 * A * C
    valid = discriminant >= 0

    depth_map = torch.zeros(height * width, device=K.device, dtype=torch.float32)

    # For rays with valid intersections, calculate the depth
    root1 = (-B[valid] + torch.sqrt(discriminant[valid])) / (2 * A[valid])
    root2 = (-B[valid] - torch.sqrt(discriminant[valid])) / (2 * A[valid])
    positive_root = torch.where(root1 >= 0, root1, torch.where(root2 >= 0, root2, torch.tensor(2 * skysphere_radius, device=K.device)))

    depth_map[valid] = positive_root

    # For rays without valid intersections, set depth equal to the diameter of the sphere
    depth_map[~valid] = 2 * skysphere_radius
    depth_map = depth_map.reshape(height, width)

    return depth_map


def fibonacci_sphere(samples: int = 1000, radius: float = 300):
    """
    Generates 3D points on a sphere using the Fibonacci spiral approach.

    Args:
    - samples (int): Number of points to generate.
    - radius (float): Radius of the sphere.

    Returns:
    - torch.Tensor: 3D points on the sphere's surface of shape (samples, 3).
    """
    indices = torch.arange(0, samples, dtype=torch.float32) + 0.5

    phi = torch.tensor(math.pi * (3. - math.sqrt(5.)), dtype=torch.float32)  # golden angle in radians
    y = 1 - (indices / (samples - 1)) * 2  # y goes from 1 to -1
    radius_sqrt = torch.sqrt(1 - y * y)  # radius at y

    theta = phi * indices  # golden angle increment

    x = torch.cos(theta) * radius_sqrt
    z = torch.sin(theta) * radius_sqrt

    points = torch.stack((radius * x, radius * y, radius * z), dim=-1)

    return points

