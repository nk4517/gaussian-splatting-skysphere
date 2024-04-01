import threading
from typing import Optional

import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer

from scene.cameras import Camera, MiniCam, DynamicMiniCamKRT, MiniCamKRT
from tiny_renderer.renderer_cuda import CUDARenderer
from scene import Scene


MENU_HEIGHT = 18
MENU_WIDTH = 160 + 6


class CamNaka:
    def __init__(self):
        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False
        self.prev_x = None
        self.prev_y = None
        self.zoom = 1

        self.viewpoint_camera: Camera | MiniCam | None = None


    def cursor_pos_callback(self, window, xpos, ypos, *args, **kwargs):
        if imgui.get_io().want_capture_mouse:
            self.is_leftmouse_pressed = False
            self.is_rightmouse_pressed = False
            return

        if self.viewpoint_camera:
            if self.is_rightmouse_pressed:
                sens = 45
                if self.prev_x is not None:
                    dx = xpos - self.prev_x
                    # print(dx)
                    self.viewpoint_camera.T[0] += dx / sens / self.zoom

                self.prev_x = xpos

                if self.prev_y is not None:
                    dy = ypos - self.prev_y
                    # print(dy)
                    self.viewpoint_camera.T[1] += dy / sens / self.zoom

                self.prev_y = ypos

                self.viewpoint_camera.init_derived()

            elif self.is_leftmouse_pressed:
                if not hasattr(self.viewpoint_camera, "R_initial"):
                    self.viewpoint_camera.R_initial = self.viewpoint_camera.R
                    self.viewpoint_camera.T_initial = self.viewpoint_camera.T

                import pyquaternion as pq

                if self.prev_x is not None:
                    dx = xpos - self.prev_x

                    self.yaw -= dx / 10

                self.prev_x = xpos

                if self.prev_y is not None:
                    dy = ypos - self.prev_y

                    self.pitch -= dy / 10

                self.prev_y = ypos

                rr1 = pq.Quaternion(axis=(0, 1, 0), angle=math.radians(self.yaw)).transformation_matrix
                rr2 = pq.Quaternion(axis=(1, 0, 0), angle=math.radians(self.pitch)).transformation_matrix
                rr1 = torch.tensor(rr1, dtype=torch.float, device=self.viewpoint_camera.R_initial.device)
                rr2 = torch.tensor(rr2, dtype=torch.float, device=self.viewpoint_camera.R_initial.device)

                rt = torch.eye(4, dtype=torch.float, device=self.viewpoint_camera.R_initial.device)
                rt[:3, :3] = self.viewpoint_camera.R_initial
                rt[:3, 3] = self.viewpoint_camera.T_initial

                xx = (rr1.inverse() @ (rr2 @ rt))

                self.viewpoint_camera.R = xx[:3, :3]
                self.viewpoint_camera.T = xx[:3, 3]
                self.viewpoint_camera.init_derived()


            else:
                self.prev_x = None
                self.prev_y = None



    def mouse_button_callback(self, window, button, action, mod, *args, **kwargs):
        if imgui.get_io().want_capture_mouse:
            self.is_leftmouse_pressed = False
            self.is_rightmouse_pressed = False
            return

        pressed = action == glfw.PRESS
        self.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
        self.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)



    def wheel_callback(self, window, dx, dy, *args, **kwargs):
        if imgui.get_io().want_capture_mouse:
            return
        # print(dx, dy)
        if self.viewpoint_camera:
            fx = self.viewpoint_camera.K[0, 0]
            fy = self.viewpoint_camera.K[1, 1]
            if dy < 0:
                fx *= 1.05
                fy *= 1.05
                self.zoom *= 1.05
            else:
                fx /= 1.05
                fy /= 1.05
                self.zoom /= 1.05

            self.viewpoint_camera.K[0, 0] = fx
            self.viewpoint_camera.K[1, 1] = fy

            self.viewpoint_camera.init_derived()


    def key_callback(self, window, key, scancode, action, mods, *args, **kwargs):
        # print(key, scancode, action, mods)
        pass



class SimpleGUI(CamNaka):
    def __init__(self, scene_lock: Optional[threading.RLock] = None):
        super().__init__()

        self.scene_lock = scene_lock
        self.scene: Scene | None = None

        self.g_renderer = CUDARenderer()
        self.cam_idx = 0

        self.window = None
        self.impl = None

        self.show_control_window = False



    def init(self):
        if not glfw.init():
            print("Could not initialize OpenGL context")
            exit(1)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        initial_width = 640 + MENU_WIDTH
        initial_height = 480 + MENU_HEIGHT

        self.window = glfw.create_window(initial_width, initial_height, "Simple GUI", None, None)
        if not self.window:
            glfw.terminate()
            print("Could not initialize Window")
            exit(1)

        glfw.make_context_current(self.window)
        imgui.create_context()

        glfw.set_cursor_pos_callback(self.window, self.cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.wheel_callback)
        glfw.set_key_callback(self.window, self.key_callback)

        glfw.set_window_size_callback(self.window, self.window_resize_callback)

        # if args.hidpi:
        #     imgui.get_io().font_global_scale = 1.5

        self.impl = GlfwRenderer(self.window, attach_callbacks=False)

        self.g_renderer.init_gl()
        self.window_resize_callback(self.window, initial_width, initial_height)

        glfw.swap_interval(1)




    def window_resize_callback(self, window, width, height, *args, **kwargs):
        # print(f"window_resize_callback: {width}, {height}")
        #gl.glViewport(0, 0, width-MENU_WIDTH, height)
        #self.g_renderer.set_render_reso(width-MENU_WIDTH, height)
        # g_camera.update_resolution(height, width)
        # g_renderer.set_render_reso(width, height)
        self.upd_cam_viewport(self.viewpoint_camera)

    def show_controls(self):

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item("Quit", None, False, True)
                if clicked_quit:
                    glfw.set_window_should_close(self.window, True)
                imgui.end_menu()

            clicked, self.show_control_window = imgui.menu_item("Show Control", None, self.show_control_window)

            clicked_square, selected_square = imgui.menu_item("Make Square", None, False, True)
            if clicked_square:
                glfw.set_window_size(self.window, 500, 500)

            imgui.end_main_menu_bar()

        if self.show_control_window:
            self.show_control_window = imgui.show_metrics_window()

        w, h = glfw.get_framebuffer_size(self.window)

        imgui.set_next_window_position(w-160-6, 0)
        imgui.set_window_size(160, h)
        # imgui.set_next_window_content_size(140, 460)

        imgui.begin("Control Window", False, flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)
        # imgui.text("This is the Control window")

        changed, scale_modifier = imgui.slider_float(
            "", self.g_renderer._scale_modifier, 0.01, 1, "scale=%.2f"
        )
        imgui.same_line()
        if imgui.button(label="reset"):
            scale_modifier = 1.
            changed = True

        if changed:
            self.g_renderer.set_scale_modifier(scale_modifier)

        if self.scene is not None:
            self.cams_widget()

        imgui.end()


    def cams_widget(self):
        cam_scale = list(self.scene.train_cameras.keys())[0]
        ks = self.scene.train_cameras[cam_scale]

        cams = [(c.image_name, i) for i, c in enumerate(ks)]
        cams.sort()
        cnames = [c for c, i in cams]
        cids = [i for c, i in cams]

        imgui.begin_group()
        imgui.text("Cam:")
        prev5_pressed = imgui.button("<<")
        imgui.same_line()
        prev_pressed = imgui.button("<")
        imgui.same_line()
        next_pressed = imgui.button(">")
        imgui.same_line()
        next5_pressed = imgui.button(">>")
        cam_selected, self.cam_idx = imgui.combo("cam_idx", self.cam_idx, cnames)
        imgui.end_group()

        if prev_pressed:
            self.cam_idx -= 1
            cam_selected = True
        elif prev5_pressed:
            self.cam_idx -= 5
            cam_selected = True
        elif next_pressed:
            self.cam_idx += 1
            cam_selected = True
        elif next5_pressed:
            self.cam_idx += 5
            cam_selected = True
        if self.cam_idx < 0:
            self.cam_idx += len(ks)
        self.cam_idx %= len(ks)
        if cam_selected:
            cam: Camera = ks[cids[self.cam_idx]]
            self.upd_cam_viewport(cam)


    def upd_cam_viewport(self, cam):
        if cam is None:
            return

        self.zoom = 1
        self.yaw = 0
        self.pitch = 0

        w, h = glfw.get_window_size(self.window)
        w -= MENU_WIDTH
        h -= MENU_HEIGHT

        cam_scale = min(w / cam.image_width, h / cam.image_height)

        upscale = float(cam_scale)

        w1 = round(cam.image_width * upscale)
        h1 = round(cam.image_height * upscale)

        K_new = cam.K.clone()
        K_new[0, 0] *= upscale
        K_new[1, 1] *= upscale
        K_new[0, 2] *= upscale
        K_new[1, 2] *= upscale

        self.viewpoint_camera = MiniCamKRT(K_new, cam.R.clone(), cam.T.clone(), w1, h1)
        self.g_renderer.set_render_reso(w1, h1)
        gl.glViewport(0, 0, w1, h1)


    def main_loop(self):
        while not glfw.window_should_close(self.window):

            glfw.poll_events()
            # glfw.wait_events()
            self.impl.process_inputs()
            imgui.new_frame()

            gl.glClearColor(0.5, 0.5, 0.5, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            self.show_controls()

            if self.scene_lock is not None:
                try:
                    self.scene_lock.acquire(timeout=1/20)

                    if self.scene is not None:
                        if self.viewpoint_camera is None:
                            k = list(self.scene.train_cameras.keys())[0]
                            cam: Camera = self.scene.train_cameras[k][self.cam_idx]

                            self.upd_cam_viewport(cam)

                        self.g_renderer.rasterize(self.scene, self.viewpoint_camera)
                except Exception as e:
                    print(e)
                finally:
                    if self.scene_lock._is_owned():
                        self.scene_lock.release()

            self.g_renderer.draw()

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)


    def cleanup(self):
        self.impl.shutdown()
        glfw.terminate()


    def run(self):
        self.init()
        self.main_loop()
        self.cleanup()

    def setScene(self, scene: Scene):
        self.scene = scene


if __name__ == "__main__":
    gui = SimpleGUI()
    gui.run()