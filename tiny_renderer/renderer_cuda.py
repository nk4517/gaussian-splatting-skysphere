from collections import OrderedDict
import math

from OpenGL import GL as gl

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import numpy as np
import torch

from cuda import cudart as cu

from scene import GaussianModel, Scene
from scene.cameras import Camera, MiniCam
from .util import compile_shaders

from tiny_renderer.colormaps111 import apply_colormap, cm_data_magma_r, cm_data_magma

from tiny_renderer.colormaps111 import cm_data_twilight, cm_data_sunglight


class GaussianRenderBase:
    RENDERMODE_STUB = -1

    render_modes = OrderedDict([
        ("Stub", RENDERMODE_STUB), ])

    default_render_mode = RENDERMODE_STUB


    def __init__(self):
        self.render_mode = 0
        self._reduce_updates = True


    @property
    def reduce_updates(self):
        return self._reduce_updates


    @reduce_updates.setter
    def reduce_updates(self, val):
        self._reduce_updates = val
        self.update_vsync()


    def update_vsync(self):
        print("VSync is not supported")


    def update_gaussian_data(self, gaus: GaussianModel):
        raise NotImplementedError()


    def sort_and_update(self, camera: Camera):
        raise NotImplementedError()


    def set_scale_modifier(self, modifier: float):
        raise NotImplementedError()


    def set_render_mode(self, mod: int):
        raise NotImplementedError()


    def update_camera_pose(self, camera: Camera):
        raise NotImplementedError()


    def update_camera_intrin(self, camera: Camera):
        raise NotImplementedError()


    def draw(self):
        raise NotImplementedError()


    def set_render_reso(self, w, h):
        raise NotImplementedError()


VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec4 fragColor;
smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    gl_Position = positions[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330

smooth in vec2 texcoords;

out vec4 outputColour;

uniform sampler2D texSampler;

void main()
{
    outputColour = texture(texSampler, texcoords);
}
"""


def pseudocolor_from_splat_orientation(scene, viewpoint_camera: Camera | MiniCam):

    from utils.general_utils import build_rotation

    means3D = scene.gaussians.get_xyz
    rot = scene.gaussians.get_rotation
    scales = scene.gaussians.get_scaling

    rotations_mat = build_rotation(rot)
    min_scales = torch.argmin(scales, dim=1)
    indices = torch.arange(min_scales.shape[0])
    normal = rotations_mat[indices, :, min_scales]

    # convert normal direction to the camera; calculate the normal in the camera coordinate
    view_dir = means3D - viewpoint_camera.camera_center[None, ...]
    view_dir /= view_dir.norm(dim=1)[..., None]

    to_inv = (view_dir * normal).sum(dim=-1) < 0
    normal[to_inv] *= -1


    R_w2c = viewpoint_camera.R # GLM
    normal = (R_w2c @ normal.transpose(0, 1)).transpose(0, 1)

    colors = normal / 2 + 0.5

    sky_mask = scene.gaussians.get_skysphere.squeeze(1) > 0.6
    colors[sky_mask, :] = 0

    return colors


def simple_hash(index):
    # Using a simple speudo-hash function to generate a pseudo-random color
    r = (index * 67 % 256) / 255.0
    g = (index * 129 % 256) / 255.0
    b = (index * 193 % 256) / 255.0
    r[index == -1] = 0.5
    g[index == -1] = 0.5
    g[index == -1] = 0.5
    return torch.tensor([r, g, b])


def pseudocolor_from_domination(dominating_splats):

    # Create an empty tensor for the colored image
    #colored_img = torch.zeros_like(dominating_splats, dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 3)

    colored_img = torch.zeros((*dominating_splats.shape, 3), dtype=torch.float32, device=dominating_splats.device)
    colored_img[..., 0] = ((dominating_splats * 67) % 256) / 255
    colored_img[..., 1] = ((dominating_splats * 129) % 256) / 255
    colored_img[..., 2] = ((dominating_splats * 193) % 256) / 255
    colored_img[dominating_splats == -1, :] = 0.5

    # # Assign a color to each pixel based on its splat index
    # for splats, color in colors.items():
    #     mask = dominating_splats == splats
    #     colored_img[mask] = color

    # Permute the tensor to match the expected color channel layout [3, H, W]
    # colored_img = colored_img.permute(2, 0, 1)

    return colored_img


def pseudocolor_from_depth_gradient(depth):
    from utils.general_utils import pseudo_normals_from_depthmap_gradient

    nn = pseudo_normals_from_depthmap_gradient(depth.squeeze())
    img = nn.permute(1, 2, 0)
    return img


class CUDARenderer(GaussianRenderBase):
    RENDERMODE_SKYNESS = -8
    RENDERMODE_LEARNING_GRADIENT = -7
    RENDERMODE_DOMINATION = -6
    RENDERMODE_BLURINESS = -5
    RENDERMODE_DEPTH_IMAGE_GRADIENT = -4
    RENDERMODE_DEPTH_SPEUDOCOLOR = -3
    RENDERMODE_DEPTH_FROM_RASTERIZER = -1
    RENDERMODE_NORMALS_PSEUDOCOLORS = -2

    render_modes = OrderedDict([
        ("Skyness", RENDERMODE_SKYNESS),
        ("Learning gradient", RENDERMODE_LEARNING_GRADIENT),
        ("Domination", RENDERMODE_DOMINATION),
        ("Bluriness", RENDERMODE_BLURINESS),
        ("Depth (pseudoRGB)", RENDERMODE_DEPTH_SPEUDOCOLOR),
        ("Depth (Rasterizer)", RENDERMODE_DEPTH_FROM_RASTERIZER),
        ("Normals", RENDERMODE_NORMALS_PSEUDOCOLORS),
        ("Depth-Gradient", RENDERMODE_DEPTH_IMAGE_GRADIENT),

        ("SH:0", 0),
        ("SH:0~1", 1),
        ("SH:0~2", 2),
        ("SH:0~3 (default)", 3)])

    default_render_mode = 3


    def __init__(self):
        super().__init__()
        self.need_rerender = True
        self.width = 100
        self.height = 100

    # def update_vsync(self):
    #     if wglSwapIntervalEXT is not None:
    #         wglSwapIntervalEXT(1 if self.reduce_updates else 0)
    #     else:
    #         print("VSync is not supported")


    def init_gl(self):
        # gl.glViewport(0, 0, w, h)
        self.program = compile_shaders(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)

        # setup cuda
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError(
                "OpenGL context may be running on integrated graphics"
            )

        self.vao = gl.glGenVertexArrays(1)
        self.tex = None
        self.set_gl_texture(self.width, self.height)

        # gl.glDisable(gl.GL_CULL_FACE)
        # gl.glEnable(gl.GL_BLEND)
        # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # self.need_rerender = True
        # self.update_vsync()

        self._scale_modifier = 1.0



    def sort_and_update(self, camera: Camera):
        self.need_rerender = True


    def set_scale_modifier(self, modifier):
        self.need_rerender = True
        self._scale_modifier = float(modifier)


    def set_render_mode(self, mod: int):
        print("set_render_mod", mod)
        self.render_mode = mod
        self.need_rerender = True


    def set_gl_texture(self, h, w):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to register opengl texture")


    def set_render_reso(self, w, h):
        self.need_rerender = True
        self.width = int(w)
        self.height = int(h)
        self.set_gl_texture(h, w)


    def update_camera_pose(self, camera: Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["viewmatrix"] = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings["campos"] = torch.tensor(camera.position).float().cuda()
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()
        self.raster_settings["projmatrix_raw"] = torch.tensor(proj.T).float().cuda()


    def update_camera_intrin(self, camera: Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()
        self.raster_settings["projmatrix_raw"] = torch.tensor(proj.T).float().cuda()
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings["tanfovx"] = hfovx
        self.raster_settings["tanfovy"] = hfovy


    def draw(self):
        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)


    def rasterize(self, scene: Scene, viewpoint_camera: Camera | MiniCam):
        if scene is None or viewpoint_camera is None:
            return

        if not self.need_rerender:
            return

        override_colors = None
        skyness = scene.gaussians.get_skysphere
        if self.render_mode == self.RENDERMODE_NORMALS_PSEUDOCOLORS:
            override_colors = pseudocolor_from_splat_orientation(scene, viewpoint_camera)

        elif self.render_mode == self.RENDERMODE_SKYNESS:
            override_colors = torch.full((skyness.shape[0], 3), 0.5, device=skyness.device)
            override_colors[..., 0] = 1 - skyness.squeeze(1)
            override_colors[..., 1] = (skyness.squeeze(1) - 0.5).abs()
            override_colors[..., 2] = skyness.squeeze(1)

        color, radii, depth, alpha, n_touched, splat_depths, n_dominated, dominating_splat = self.render111(scene, viewpoint_camera, override_colors=override_colors)

        from utils.loss_utils import normalize

        if self.render_mode == self.RENDERMODE_DEPTH_IMAGE_GRADIENT:
            img = pseudocolor_from_depth_gradient(depth)

        elif self.render_mode == self.RENDERMODE_DOMINATION:
            #img = pseudocolor_from_domination(dominating_splat)
            img = scene.gaussians.statblock.associated_color[dominating_splat, :]
            img[dominating_splat == -1, :] = 0.5

        elif self.render_mode == self.RENDERMODE_DOMINATING_DEPTH:
            #img = pseudocolor_from_domination(dominating_splat)
            img = scene.gaussians.statblock.min_depthF[dominating_splat]
            img = apply_colormap(img, cm_data_sunglight).contiguous()



        elif self.render_mode == self.RENDERMODE_DEPTH_FROM_RASTERIZER:

            sky_splats = (scene.gaussians.get_skysphere > 0.66).squeeze(1)
            sky_invdepth = scene.gaussians.statblock.min_depthF_iteration[sky_splats].mean() * 0.8

            invdepth_mask = depth < (sky_invdepth * scene.getTrainCameras()[0].focal_x)
            d1: torch.Tensor = depth[invdepth_mask]
            q1 = d1.quantile(0.05)
            q2 = d1.quantile(0.95)

            d2 = ((depth - q1) / (q2 - q1)).clamp(0, 1)

            img = apply_colormap(d2, cm_data_sunglight).contiguous()


        elif self.render_mode == self.RENDERMODE_BLURINESS:
            # accumed = scene.gaussians.xyz_gradient_accum / scene.gaussians.n_touched_accum
            # accumed[~accumed.isfinite()] = 0
            # accumed /= accumed.max()
            # pse1 = (accumed).repeat(1, 3).contiguous()

            # area = 2 * math.pi * (radii**2)
            # n_total = (viewpoint_camera.image_width * viewpoint_camera.image_height)
            # pct = n_touched / n_total

            #naka = normalize(n_touched.float().unsqueeze(0)).unsqueeze(0)

            # dist = torch.norm(scene.gaussians.get_xyz - viewpoint_camera.camera_center[None, ...], dim=1)
            # pse1 = normalize(dist.clamp(1e-6, 1e6).square().unsqueeze(0)).permute(1, 0).repeat(1, 3)

            sky_mask = skyness.squeeze(1) > 0.6

            sq = scene.gaussians.statblock.filter3d_sq.sqrt().unsqueeze(1)

            if sq.shape[0]:

                sq[sky_mask, :] = 0

                pse1 = sq
                pse1[sky_mask, :] = 0

                q1 = pse1.quantile(0.05)
                q2 = pse1.quantile(0.95)

                pse1 = ((pse1 - q1) / (q2 - q1)).clamp(0, 1)

                pse1 = apply_colormap(pse1, cm_data_sunglight).contiguous()

                # pse1 = pse1.repeat(1, 3).contiguous()

                color, _, _, _, _, _, _, _ = self.render111(scene, viewpoint_camera, override_colors=pse1, overmax_opacity=True)
                img = color.permute(1, 2, 0)

            else:
                img = color.permute(1, 2, 0)

        elif self.render_mode == self.RENDERMODE_LEARNING_GRADIENT:

            denom = scene.gaussians.statblock.n_touched_accum
            wpx = scene.gaussians.statblock.xyz_gradient_accum / denom  # scene.gaussians.statblock.gradWpx()
            wpx[~wpx.isfinite()] = 0

            q1 = wpx.quantile(0.05)
            q2 = wpx.quantile(0.95)

            pse1 = ((wpx - q1) / (q2 - q1)).clamp(0, 1)

            pse1 = apply_colormap(pse1, cm_data_magma).contiguous()

            color, _, _, _, _, _, _, _ = self.render111(scene, viewpoint_camera, override_colors=pse1, overmax_opacity=False)
            img = color.permute(1, 2, 0)

        else:
            img = color.permute(1, 2, 0)

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
            img = img.contiguous()
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = img.repeat(1, 1, 3)
            img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
            img = img.contiguous()

        self.transfert1(img)

        self.need_rerender = False


    def render111(self, scene, viewpoint_camera, override_colors=None, overmax_opacity=False):
        with torch.no_grad():

            bg_color = torch.Tensor([0.4, 0.3, 0.4]).float().cuda()
            kernel_size = 0.1

            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

            raster_settings = GaussianRasterizationSettings(
                image_height=self.height,
                image_width=self.width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                kernel_size=kernel_size,
                bg=bg_color,
                scale_modifier=self._scale_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform,
                projmatrix_raw=viewpoint_camera.projection_matrix,
                sh_degree=scene.gaussians.active_sh_degree,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                depth_threshold=None,
                debug=False
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            if override_colors is not None:
                sh = None
                colors = override_colors.contiguous()
            else:
                sh = scene.gaussians.get_features.contiguous()
                colors = None

            means3D = scene.gaussians.get_xyz
            rot = scene.gaussians.get_rotation
            scal, opa = scene.gaussians.get_scal_opa_w_3D
            # scal = scene.gaussians.get_scaling
            # opa = scene.gaussians.get_opacity

            if overmax_opacity or self._scale_modifier <= 0.02:
                opa = torch.full_like(scene.gaussians._opacity, fill_value=1e3)

            if self._scale_modifier <= 0.02:
                scal = scene.gaussians.get_scaling
            # else:
            #     scal, opa = scene.gaussians.get_scal_opa_w_3D
            #     # opa = scene.gaussians.get_opacity_with_3D_filter
            #     # scal = scene.gaussians.get_scaling_with_3D_filter

            cov3D_precomp = None

            color, radii, depth, alpha, n_touched, splat_depths, n_dominated, dominating_splat = rasterizer(
                means3D=means3D.contiguous(),
                means2D=None,
                shs=sh,
                colors_precomp=colors,
                opacities=opa.contiguous(),
                scales=scal.contiguous(),
                rotations=rot.contiguous(),
                cov3D_precomp=cov3D_precomp
            )

        return color, radii, depth, alpha, n_touched, splat_depths, n_dominated, dominating_splat


    def transfert1(self, img_rgba: torch.Tensor):

        height, width = img_rgba.shape[:2]
        # transfer
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img_rgba.data_ptr(),
            4 * 4 * width,
            4 * 4 * width,
            height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")
        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")


