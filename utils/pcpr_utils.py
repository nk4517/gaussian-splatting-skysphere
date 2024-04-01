import numpy as np
import torch


def make_pcpr_bundle(src_imgs: list[np.ndarray], src_ks: list[np.ndarray], src_poses: list[np.ndarray]):

    assert len(src_imgs) == len(src_ks) == len(src_poses)
    N_sampling = len(src_imgs)
    max_w = max(img.shape[1] for img in src_imgs)
    max_h = max(img.shape[0] for img in src_imgs)
    pcpr_imgs = torch.zeros((N_sampling, max_h, max_w, 3), dtype=torch.uint8)
    pcpr_sizes = torch.zeros((N_sampling, 2), dtype=torch.int)
    pcpr_Ks = torch.zeros((N_sampling, 3, 3), dtype=torch.float)
    pcpr_poses = torch.zeros((N_sampling, 4, 3), dtype=torch.float)
    for i, (img, img_K, w2c_cur) in enumerate(zip(src_imgs, src_ks, src_poses)):
        h, w = img.shape[:2]

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        pcpr_imgs[i, :h, :w, :] = img
        pcpr_Ks[i] = torch.tensor(img_K)
        pcpr_sizes[i, 0] = w
        pcpr_sizes[i, 1] = h
        pcpr_poses[i] = PCPRRenderer.xform2pcpr(w2c_cur)

    return pcpr_imgs.cuda(), pcpr_sizes.cuda(), pcpr_Ks.cuda(), pcpr_poses.cuda()


class PCPRSourcesImagesBundle:
    def __init__(self, src_imgs: list[np.ndarray], src_ks: list[np.ndarray], src_poses: list[np.ndarray]):

        self.pcpr_imgs, self.pcpr_sizes, self.pcpr_Ks, self.pcpr_poses = make_pcpr_bundle(src_imgs, src_ks, src_poses)


class PCPRRenderer:
    def __init__(self):
        pass

    @staticmethod
    def xform2pcpr(w2c_xform: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(w2c_xform, np.ndarray):
            w2c_xform = torch.from_numpy(w2c_xform)

        c2w_xform_pcpr = torch.inverse(w2c_xform)

        R = c2w_xform_pcpr[:3, :3]
        t = c2w_xform_pcpr[:3, 3]
        C = -torch.matmul(R.t(), t)

        # t=(i, 0, 0) - при увеличении i - движение ВЛЕВО
        # t=(0, i, 0) - при увеличении i - движение ВНИЗ
        # t=(0, 0, i) - при увеличении i - движение НАЗАД
        # axis=(0,0,1), angle=i - вращение вокруг оси "вперёд", по часовой.
        # axis=(0,1,0), angle=i - вращение вокруг вертикальной оси, влево.
        # axis=(1,0,0), angle=i - наклон вниз

        fwd = R[2]  # Вперед (Z)
        up = R[1]  # Вверх (Y)
        right = R[0]  # Вправо (X)

        pcpr_pose = torch.zeros((4, 3), dtype=torch.float32)
        # pose inner: float3 norm, Xaxis, Yaxis, offset;
        pcpr_pose[0] = fwd
        pcpr_pose[1] = right
        pcpr_pose[2] = up
        pcpr_pose[3] = C

        return pcpr_pose


    @staticmethod
    def render_rgb(depth_map: torch.Tensor, dmap_K: np.ndarray, dmap_xform: np.ndarray,
                   bundle: PCPRSourcesImagesBundle):
        import pcpr

        out_rgb = torch.zeros((*depth_map.shape, 3), dtype=torch.uint8, device='cuda')
        tar_pose = PCPRRenderer.xform2pcpr(dmap_xform)

        pcpr.rgb_render(
            torch.from_numpy(dmap_K).float().cuda().contiguous(),
            tar_pose.float().cuda().contiguous(),
            bundle.pcpr_Ks,
            bundle.pcpr_poses,
            bundle.pcpr_sizes,
            bundle.pcpr_imgs,
            depth_map.float().cuda().contiguous(),
            out_rgb,
        )

        return out_rgb

    @classmethod
    def render_depthmap(cls, points3d, width, height, K, w2c_xform, near=0.1, far=100_000, max_splatting_size=0.1):
        import pcpr

        if isinstance(points3d, np.ndarray):
            points3d = torch.from_numpy(points3d)

        tar_intrinsic = torch.tensor(K)
        tar_pose = PCPRRenderer.xform2pcpr(w2c_xform)

        out_depth = torch.zeros((height, width), dtype=torch.float32, device='cuda')
        out_index = torch.zeros((height, width), dtype=torch.int32, device='cuda')

        depth, index = pcpr.forward(
            points3d.float().contiguous().cuda(),
            tar_intrinsic.float().contiguous().cuda(),
            tar_pose.float().contiguous().cuda(),
            out_depth,
            out_index,
            near,
            far,
            max_splatting_size
        )

        return depth, index
