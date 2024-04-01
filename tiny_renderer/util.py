from enum import Enum
import json
import math
from pathlib import Path
from typing import List

from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import numpy as np
import glm

import torch



import pyquaternion as pq


class Projection111(Enum):
    BALL = 0
    PLANE = 1
    HYPERBOLIC = 2


class Arcball1:

    def __init__(self):

        self.prev_cx, self.prev_cy, self.prev_theta = None, None, None
        self.q_prev = pq.Quaternion()

        self.prev_touch_id0 = self.prev_touch_id1 = None

        self.prev_theta = None
        self.prev_proj3d = None

        self.SCALE_BALL = 10 #* 0.25
        # self.SCALE_PINCH = 1
        self.SCALE_PINCH = self.SCALE_BALL / 8


        self.sphere_radius = 0.8
        self.RELATIVE = True
        self.SCALE_BALL = 1


    #  Define a function to project a screen point onto the sphere or the plane
    def project_point_sphere_plane(self, nx, ny):
        # # Normalize the screen coordinates to [-1, 1]
        # nx = 2.0 * x / screen_width - 1.0
        # ny = 2.0 * y / screen_height - 1.0

        # Compute the squared distance from the origin
        d2 = nx * nx + ny * ny

        # If the point is inside the sphere, project it onto the sphere
        if d2 < self.sphere_radius * self.sphere_radius:
            # Use Pythagoras theorem to compute the z coordinate
            nz = math.sqrt(self.sphere_radius * self.sphere_radius - d2)
            # Return the normalized vector
            return np.array([nx, ny, nz]) / self.sphere_radius, Projection111.BALL

        # If the point is outside the sphere, project it onto the plane
        else:
            # Return the normalized vector with z = 0
            return np.array([nx, ny, 0.0]) / math.sqrt(d2), Projection111.PLANE

    def project_point_sphere_hyperbolic(self, nx, ny):
        # from https://www.khronos.org/opengl/wiki/Object_Mouse_Trackball

        # Compute the squared distance from the origin
        d2 = nx * nx + ny * ny

        # If the point is inside the sphere, project it onto the sphere
        if d2 < self.sphere_radius * self.sphere_radius / 2:
            # Use Pythagoras theorem to compute the z coordinate
            nz = math.sqrt(self.sphere_radius * self.sphere_radius - d2)
            # Return the normalized vector
            return np.array([nx, ny, nz]) / self.sphere_radius, Projection111.BALL

        # If the point is outside the sphere, project it onto the hyperbolic sheet
        else:
            # Compute the z coordinate using the equation of the hyperbolic sheet
            nz = (self.sphere_radius * self.sphere_radius / 2) / math.sqrt(d2)
            # Return the normalized vector
            return np.array([nx, ny, nz]) / math.sqrt(d2 + nz * nz), Projection111.PLANE


    def project_point_to_plane(self, nx, ny):
        d2 = nx * nx + ny * ny
        return np.array([nx, ny, 0.0]) / math.sqrt(d2)


    def compute_rotation(self, p0, p1):
        axis = np.cross(p0, p1)
        dot = np.dot(p0, p1)
        angle = math.acos(dot)
        return axis, angle

    project_point = project_point_sphere_hyperbolic

    def reset_prev(self):
        self.prev_proj3d = None
        self.prev_theta = None

    def update_quat(self, c_x, c_y, abs_theta, angle_c = 0):
        if self.prev_proj3d is None and c_x is not None:
            self.prev_proj3d, _ = self.project_point(c_x, c_y)
            return None, None

        if self.prev_theta is None and abs_theta is not None:
            self.prev_theta = abs_theta
            return None, None

        if c_x is not None and self.prev_proj3d is not None:
            proj3d_cur, mode = self.project_point(c_x, c_y)
            dd = np.linalg.norm((proj3d_cur - self.prev_proj3d))

            if self.RELATIVE and mode != Projection111.BALL:
                dd = 0

        else:
            dd = 0

        if abs_theta is not None and self.prev_theta is not None:
            angle_c = abs_theta - self.prev_theta

        if dd < 1e-6 and abs(angle_c) < 1e-6:
            return None, None

        q = pq.Quaternion(self.q_prev)

        if dd != 0:
            axis_s, angle_s = self.compute_rotation(self.prev_proj3d, proj3d_cur)
            angle_s *= self.SCALE_BALL  # чувствительность побольше

            # print("angle_s", math.degrees(angle_s))
            q_s = pq.Quaternion(axis=axis_s, angle=angle_s)
            q = (q_s * q).normalised

        if angle_c != 0:
            angle_c *= self.SCALE_PINCH  # а вот тут чувствительность поменьше

            axis_c = (0, 0, 1)
            # print("angle_c", math.degrees(angle_c))
            q_c = pq.Quaternion(axis=axis_c, angle=angle_c)
            q = (q_c * q).normalised

        q_diff = q * self.q_prev.inverse

        self.q_prev = q

        if dd != 0:
            self.prev_proj3d = proj3d_cur

        if abs_theta is not None:
            self.prev_theta = abs_theta

        return q, q_diff

    def get_object_matrix4(self, q: pq.Quaternion):
        inverse_q = q.inverse
        matrix4 = inverse_q.transformation_matrix
        return matrix4



class Camera:

    ARCBALL_NONE = 0
    ARCBALL_CENTER_LOCKED = 1
    ARCBALL_RIM_LOCKED = 2
    ARCBALL_ABSOLUTE = 3

    def __init__(self, h, w):
        self.znear = 0.01
        self.zfar = 1000
        self.h = h
        self.w = w
        self.fovy = np.pi / 6
        self.position = np.array([0.0, 0.0, 3.0]).astype(np.float32)
        self.target = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        self.up = np.array([0.0, -1.0, 0.0]).astype(np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0
        
        self.is_pose_dirty = True
        self.is_intrin_dirty = True
        
        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True
        
        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False
        
        self.rot_sensitivity = 0.02
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03
        self.target_dist = 3.

        self.arcball = Arcball1()
        self.arcball.sphere_radius = 3
        self.start_x = self.start_y = None
        self.arc_state = Camera.ARCBALL_NONE
        self.arcball.RELATIVE = True
        self.arcball.SCALE_BALL = 4


    @property
    def fov_deg(self):
        return math.degrees(self.fovy)

    @fov_deg.setter
    def fov_deg(self, value):
        self.fovy = math.radians(value)
        self.is_intrin_dirty = True

    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def get_view_matrix(self):
        return np.array(glm.lookAt(
            self.position.astype(np.float32),
            self.target.astype(np.float32),
            self.up.astype(np.float32)
        ))

    def get_project_matrix(self):
        # htanx, htany, focal = self.get_htanfovxy_focal()
        # f_n = self.zfar - self.znear
        # proj_mat = np.array([
        #     1 / htanx, 0, 0, 0,
        #     0, 1 / htany, 0, 0,
        #     0, 0, self.zfar / f_n, - 2 * self.zfar * self.znear / f_n,
        #     0, 0, 1, 0
        # ])
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        USE_QUAT = False

        if self.is_leftmouse_pressed and not USE_QUAT:
            self.yaw += xoffset * self.rot_sensitivity
            self.pitch += yoffset * self.rot_sensitivity

            self.pitch = np.clip(self.pitch, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)

            front = np.array([np.cos(self.yaw) * np.cos(self.pitch),
                            np.sin(self.pitch), np.sin(self.yaw) *
                            np.cos(self.pitch)])
            front = self._global_rot_mat() @ front.reshape(3, 1)
            front = front[:, 0]
            self.position[:] = - front * np.linalg.norm(self.position - self.target) + self.target

            self.is_pose_dirty = True

            return

        if self.is_leftmouse_pressed and USE_QUAT:
            aspect = self.w/self.h

            rfax = xpos / (self.w / 2) - 1
            rfay = (ypos / (self.h / 2) - 1) / aspect

            if self.arc_state == Camera.ARCBALL_NONE:
                self.arcball.reset_prev()
                self.start_x = xpos
                self.start_y = ypos

                d = np.linalg.norm((rfax, rfay))
                # print(d)
                #
                if d > 0.9:
                    self.arc_state = Camera.ARCBALL_RIM_LOCKED
                else:
                    self.arc_state = Camera.ARCBALL_CENTER_LOCKED

                return
            else:

                rel_x = (xpos - self.start_x) / (self.w / 2)
                rel_y = ((ypos - self.start_y) / (self.h / 2))

                if self.arc_state == Camera.ARCBALL_RIM_LOCKED:
                    theta = - math.atan2(rfax, rfay)
                    q, q_diff = self.arcball.update_quat(None, None, theta)
                else:
                    q, q_diff = self.arcball.update_quat(rel_x, rel_y, None)


                if q_diff is not None:
                    xform_mat = q.inverse.transformation_matrix.astype(np.float32)
                    xform_mat /= xform_mat[3, 3]
                    R = xform_mat[:3, :3]

                    # вперёд - это Z, вверх - -Y
                    self.up = -R[:, 1]
                    self.target = self.position + R[:, 2] * self.target_dist

                self.is_pose_dirty = True

        else:
            if self.arc_state != Camera.ARCBALL_NONE:
                self.arc_state = Camera.ARCBALL_NONE
                print(self.arc_state)

        if self.is_rightmouse_pressed:
            front = self.target - self.position
            front = front / np.linalg.norm(front)
            right = np.cross(self.up, front)
            self.position += right * xoffset * self.trans_sensitivity
            self.target += right * xoffset * self.trans_sensitivity
            cam_up = np.cross(right, front)
            self.position += cam_up * yoffset * self.trans_sensitivity
            self.target += cam_up * yoffset * self.trans_sensitivity
            
            self.is_pose_dirty = True
        
    def process_wheel(self, dx, dy):
        if self.arc_state == Camera.ARCBALL_CENTER_LOCKED:
            angle_c = -dy/33.
            q, q_diff = self.arcball.update_quat(None, None, None, angle_c=angle_c)
            if q is not None:
                xform_mat = q.transformation_matrix  # self.arcball.get_object_matrix4(q_diff)
                xform_mat /= xform_mat[3, 3]
                R = np.linalg.inv(xform_mat[:3, :3]).astype(np.float32)

                # rotated_camera_to_target = glm.vec3(rot_mat * glm.vec4(camera_to_target, 1.0))
                # вперёд - это Z, вверх - -Y
                self.up = -R[:, 1]
                self.target = self.position + R[:, 2] * self.target_dist

                self.is_pose_dirty = True

            return

        front = self.target - self.position
        front = front / np.linalg.norm(front)
        self.position += front * dy * self.zoom_sensitivity
        self.target += front * dy * self.zoom_sensitivity
        self.is_pose_dirty = True
        
    def process_roll_key(self, d):
        front = self.target - self.position
        right = np.cross(front, self.up)
        new_up = self.up + right * (d * self.roll_sensitivity / np.linalg.norm(right))
        self.up = new_up / np.linalg.norm(new_up)
        self.is_pose_dirty = True

    def flip_ground(self):
        self.up = -self.up
        self.is_pose_dirty = True

    def update_target_distance(self):
        _dir = self.target - self.position
        _dir = _dir / np.linalg.norm(_dir)
        self.target = self.position + _dir * self.target_dist
        
    def update_resolution(self, height, width):
        self.h = max(height, 1)
        self.w = max(width, 1)
        self.is_intrin_dirty = True


def load_shaders(vs, fs):
    vertex_shader = open(vs, 'r').read()        
    fragment_shader = open(fs, 'r').read()

    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader


def compile_shaders(vertex_shader, fragment_shader):
    active_shader = shaders.compileProgram(
        shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
    )
    return active_shader


def set_attributes(program, keys, values, vao=None, buffer_ids=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_ids is None:
        buffer_ids = [None] * len(keys)
    for i, (key, value, b) in enumerate(zip(keys, values, buffer_ids)):
        if b is None:
            b = glGenBuffers(1)
            buffer_ids[i] = b
        glBindBuffer(GL_ARRAY_BUFFER, b)
        glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
        length = value.shape[-1]
        pos = glGetAttribLocation(program, key)
        glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
        glEnableVertexAttribArray(pos)
    
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_ids

def set_attribute(program, key, value, vao=None, buffer_id=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
    glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    length = value.shape[-1]
    pos = glGetAttribLocation(program, key)
    glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(pos)
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_id

def set_attribute_instanced(program, key, value, instance_stride=1, vao=None, buffer_id=None):
    glUseProgram(program)
    if vao is None:
        vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
    glBufferData(GL_ARRAY_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    length = value.shape[-1]
    pos = glGetAttribLocation(program, key)
    glVertexAttribPointer(pos, length, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(pos)
    glVertexAttribDivisor(pos, instance_stride)
    glBindBuffer(GL_ARRAY_BUFFER,0)
    return vao, buffer_id

def set_storage_buffer_data(program, key, value: np.ndarray, bind_idx, vao=None, buffer_id=None):
    glUseProgram(program)
    # if vao is None:  # TODO: if this is really unnecessary?
    #     vao = glGenVertexArrays(1)
    if vao is not None:
        glBindVertexArray(vao)
    
    if buffer_id is None:
        buffer_id = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, value.nbytes, value.reshape(-1), GL_STATIC_DRAW)
    # pos = glGetProgramResourceIndex(program, GL_SHADER_STORAGE_BLOCK, key)  # TODO: ???
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bind_idx, buffer_id)
    # glShaderStorageBlockBinding(program, pos, pos)  # TODO: ???
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return buffer_id

def set_faces_tovao(vao, faces: np.ndarray):
    # faces
    glBindVertexArray(vao)
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    return element_buffer

def set_gl_bindings(vertices, faces):
    # vertices
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    # vertex_buffer = glGenVertexArrays(1)
    vertex_buffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 4, GL_FLOAT, False, 0, None)
    glEnableVertexAttribArray(0)

    # faces
    element_buffer = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
    # glVertexAttribPointer(1, 3, GL_FLOAT, False, 36, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(1)
    # glVertexAttribPointer(2, 3, GL_FLOAT, False, 36, ctypes.c_void_p(12))
    # glEnableVertexAttribArray(2)

def set_uniform_mat4(shader, content, name):
    glUseProgram(shader)
    if isinstance(content, glm.mat4):
        content = np.array(content).astype(np.float32)
    else:
        content = content.T
    glUniformMatrix4fv(
        glGetUniformLocation(shader, name), 
        1,
        GL_FALSE,
        content.astype(np.float32)
    )

def set_uniform_1f(shader, content, name):
    glUseProgram(shader)
    glUniform1f(
        glGetUniformLocation(shader, name), 
        content,
    )

def set_uniform_1int(shader, content, name):
    glUseProgram(shader)
    glUniform1i(
        glGetUniformLocation(shader, name), 
        content
    )

def set_uniform_v3f(shader, contents, name):
    glUseProgram(shader)
    glUniform3fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )

def set_uniform_v3(shader, contents, name):
    glUseProgram(shader)
    glUniform3f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1], contents[2]
    )

def set_uniform_v1f(shader, contents, name):
    glUseProgram(shader)
    glUniform1fv(
        glGetUniformLocation(shader, name),
        len(contents),
        contents
    )
    
def set_uniform_v2(shader, contents, name):
    glUseProgram(shader)
    glUniform2f(
        glGetUniformLocation(shader, name),
        contents[0], contents[1]
    )

def set_texture2d(img, texid=None):
    h, w, c = img.shape
    assert img.dtype == np.uint8
    if texid is None:
        texid = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,   
        GL_RGB, GL_UNSIGNED_BYTE, img
    )
    glActiveTexture(GL_TEXTURE0)  # can be removed
    # glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    return texid

def update_texture2d(img, texid, offset):
    x1, y1 = offset
    h, w = img.shape[:2]
    glBindTexture(GL_TEXTURE_2D, texid)
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, x1, y1, w, h,
        GL_RGB, GL_UNSIGNED_BYTE, img
    )

class cvCam:
    def __init__(self, w, h, K, W2C_xform, cam_name="-"):
        self.name = cam_name

        self.w = w
        self.h = h

        self.K = K
        self.W2C_xform = W2C_xform.astype(np.float32)

    @property
    def R(self):
        R = self.W2C_xform[:3, :3]
        R /= np.linalg.det(R) # там и scale бывает
        return R

    @property
    def t(self):
        return self.W2C_xform[:3, 3]

    @property
    def C(self):
        return -np.linalg.inv(self.R.T) @ self.t

    def __repr__(self):
        import pyquaternion as pq
        pq = pq.Quaternion(matrix=self.W2C_xform, rtol=1e-6, atol=1e-6)
        return f"<cvCam C={np.round(self.C, 2)}, R axis={np.round(pq.axis, 2)} angle={np.rad2deg(pq.angle)}>"

    def clone(self):
        return cvCam(self.w, self.h, np.copy(self.K), np.copy(self.W2C_xform), self.name)


def try_read_cameras(ply_fullpath: str | Path) -> List[cvCam]:
    # есть шансы, что .ply - это часть модели, и там есть камеры
    # point_cloud/iteration_XXX/point_cloud.ply
    # cameras.json
    ply_fullpath = Path(ply_fullpath).absolute()
    cameras_json_fpath1 = (ply_fullpath.parent.parent.parent / "cameras.json")
    cameras_json_fpath2 = (ply_fullpath.parent / "cameras.json")
    if cameras_json_fpath1.is_file():
        cameras_json_fpath = cameras_json_fpath1
    elif cameras_json_fpath2.is_file():
        cameras_json_fpath = cameras_json_fpath2
    else:
        raise RuntimeError("no cameras.json found")

    with open(cameras_json_fpath, "rt", encoding="utf-8") as fl:
        cameras_data = json.load(fl)

    cvcams = []

    for cam_info in cameras_data:

        cx = cam_info["width"] / 2
        cy = cam_info["height"] / 2

        K = np.eye(3, dtype=np.float32)
        K[0, 0] = cam_info["fx"]
        K[1, 1] = cam_info["fy"]
        K[0, 2] = cx
        K[1, 2] = cy

        R = np.array(cam_info["rotation"], dtype=np.float32)
        C = np.array(cam_info["position"], dtype=np.float32)
        t = -(R.T @ C)

        W2C_xform = np.eye(4)
        W2C_xform[:3, :3] = R
        W2C_xform[:3, 3] = t

        cvcam = cvCam(cam_info["width"], cam_info["height"], K, W2C_xform, cam_name=cam_info["img_name"])

        assert np.allclose(cvcam.C, C, rtol=1e-4, atol=1e-4)
        assert np.allclose(cvcam.R, R, rtol=1e-4, atol=1e-4)
        assert np.allclose(cvcam.t, t, rtol=1e-4, atol=1e-4)

        cvcams.append(cvcam)

    return cvcams

    # like that
    # {
    #     'id': 0,
    #     'img_name': '0000_cam0',
    #     'width': 1907,
    #     'height': 1272,
    #     'position': [
    #         -7.227473846344491,
    #         0.36855735187729166,
    #         -1.2154362267399894,
    #     ],
    #     'rotation': [
    #         [
    #             0.153722715673349,
    #             0.035638178198492965,
    #             0.987471137269694,
    #         ],
    #         [
    #             0.1414186871240709,
    #             0.9882679490055079,
    #             -0.057682024067740166,
    #         ],
    #         [
    #             -0.9779417577842325,
    #             0.1485139091908218,
    #             0.14687932856173289,
    #         ],
    #     ],
    #     'fy': 2205.0608258885777,
    #     'fx': 2182.50387833004,
    # },


def apply_cam_info(camera: Camera, cam_info: cvCam, apply_fov=False):
    C = cam_info.C
    R = cam_info.R
    forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    camera.position[:] = C
    camera_direction = R @ forward
    camera.target[:] = camera.position + camera_direction * camera.target_dist
    camera.up = R[:3, 1] * -1
    camera.pitch = - np.arcsin(camera_direction[1])
    camera.yaw = - np.arctan2(camera_direction[2], camera_direction[0])

    if apply_fov:
        fy = cam_info.K[1, 1]
        VFov = 2 * np.arctan(cam_info.h / (2 * fy))
        camera.fovy = VFov


    camera.is_pose_dirty = True
    camera.arcball.q_prev = pq.Quaternion(matrix=R, rtol=1e-4, atol=1e-4).inverse


def quat_to_R(q: torch.Tensor):
    q /= torch.norm(q, dim=1)[..., None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R
