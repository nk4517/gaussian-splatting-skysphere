from collections import OrderedDict
import math

from OpenGL import GL as gl

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import numpy as np
import torch

from cuda import cudart as cu

from scene import GaussianModel, Scene
from scene.cameras import Camera
from .util import compile_shaders


class GaussianRenderBase:
    RENDERMODE_STUB = -1

    render_modes = OrderedDict([
        ("Stub", RENDERMODE_STUB), ])

    default_render_mode = RENDERMODE_STUB


    def __init__(self):
        self.gaussians = None
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

class CUDARenderer(GaussianRenderBase):
    RENDERMODE_DEPTH_IMAGE_GRADIENT = -4
    RENDERMODE_DEPTH_SPEUDOCOLOR = -3
    RENDERMODE_DEPTH_FROM_RASTERIZER = -1
    RENDERMODE_NORMALS_PSEUDOCOLORS = -2

    render_modes = OrderedDict([
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

        bg_color = torch.Tensor([0., 0., 0]).float().cuda()
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
            projmatrix_raw=viewpoint_camera.full_proj_transform,
            sh_degree=scene.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        with torch.no_grad():

            sh = scene.gaussians.get_features
            colors = None

            color, radii, depth, alpha, _ = rasterizer(
                means3D=scene.gaussians.get_xyz,
                means2D=None,
                shs=sh,
                colors_precomp=colors,
                opacities=scene.gaussians.get_opacity,
                scales=scene.gaussians.get_scaling,
                rotations=scene.gaussians.get_rotation,
                cov3D_precomp=None
            )

        img = color.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()

        self.transfert1(img)


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


