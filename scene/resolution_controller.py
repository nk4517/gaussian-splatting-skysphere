import math

from arguments import GaussianSplattingConf
from scene import Scene
from scene.cameras import MiniCamKRT


# проще уж посчитать от итерации.
# начало - 300, конец 0.3, точка перегиба - 1500я итерация, и дальше длинный хвост до x10 где-то до 5000й и по затухающей

def inverted_logistic(x, max_val, min_val, midpoint = 4500):
    k = -0.001  # Slope at the inflection point
    x0 = midpoint  # The x-value of the sigmoid's midpoint
    L = max_val - min_val  # The curve's maximum value minus its minimum value

    # Inverted logistic function
    y = L / (1 + math.exp(-k * (x - x0))) + min_val

    return y


class ResolutionController:
    def __init__(self, scene: Scene, conf: GaussianSplattingConf):

        self.scene = scene

        self.avail_res = scene.train_cameras.keys()

        self.final_downscale = conf.optimization_params.c2f_final_downscale
        self.initial_downscale = conf.optimization_params.c2f_initial_downscale
        self.inflection_iteration = conf.optimization_params.c2f_inflection_iteration

        if self.initial_downscale == -1:
            self.initial_downscale = 3/min(self.avail_res)

        if self.final_downscale == -1:
            self.final_downscale = 1/max(self.avail_res)

        self.cur_downscale_fullres = self.initial_downscale
        self.cur_cam_res = 1e-6
        self.cur_downscale_rel2cam = self.cur_downscale_fullres * self.cur_cam_res

        self.trainCameras_filter3d = []

        self.maximum_cam_res_reached = False

        self.filter3d_updated_downscale = 1e-6

        self.switch_to_next_res_if_need()



    def start(self):
        if not self.maximum_cam_res_reached:
            self.switch_to_next_res_if_need()


    def switch_to_next_res_if_need(self):
        if self.cur_downscale_rel2cam >= 1.5:
            return False

        # нужно переключаться на камеру с большим разрешением
        ava = [v for v in self.avail_res if v > self.cur_cam_res]
        if ava:
            self.cur_cam_res = min(ava)
            self.cur_downscale_rel2cam = self.cur_downscale_fullres * self.cur_cam_res

            self.trainCameras_filter3d = self.scene.getTrainCameras(self.cur_cam_res).copy()
            return True

        if self.cur_downscale_rel2cam < 1:
            self.cur_downscale_fullres = 1
            self.cur_downscale_rel2cam = 1
            self.maximum_cam_res_reached = True
            return False


    @property
    def cam_res_down(self):
        return self.cur_cam_res / max(self.avail_res)

    @property
    def c2f_phase(self):
        return False
        return self.cur_downscale_fullres > 8

    def update_blur(self, iteration):
        if self.maximum_cam_res_reached:
            return

        new_blur_fullres = inverted_logistic(iteration, self.initial_downscale, self.final_downscale, midpoint=self.inflection_iteration)
        new_rel_blur = new_blur_fullres * self.cur_cam_res
        if new_rel_blur < 1:
            # переключение на следующую камеру делается по новому стаку, до этого просто ждём
            self.cur_downscale_rel2cam = 1
            self.cur_downscale_fullres = 1 / self.cur_cam_res
        else:
            self.cur_downscale_fullres = new_blur_fullres
            self.cur_downscale_rel2cam = new_rel_blur

    def update_resolution_if_need(self):
        if not self.maximum_cam_res_reached:
            if self.switch_to_next_res_if_need():
                return True

    @property
    def report(self):
        return f"{self.filter3d_updated_downscale:.2f} / {self.filter3d_updated_downscale * self.cur_cam_res:.2f} @ 1/{1 / self.cur_cam_res:.0f}"

def unload_cam_data(scene: Scene):
    for res in tuple(scene.train_cameras.keys()):
        unload_res(scene, res)


def unload_res(scene: Scene, res):
    if res in scene.train_cameras:
        minicams = []
        for cam in scene.train_cameras[res]:
            minicams.append(MiniCamKRT(cam.K, cam.R, cam.T, cam.image_width, cam.image_height,
                                       zfar=cam.zfar, znear=cam.znear, uid=cam.uid, image_name=cam.image_name))
        scene.train_cameras[res] = minicams

    if res in scene.test_cameras:
        minicams_t = []
        for cam in scene.test_cameras[res]:
            minicams_t.append(MiniCamKRT(cam.K, cam.R, cam.T, cam.image_width, cam.image_height,
                                       zfar=cam.zfar, znear=cam.znear, uid=cam.uid, image_name=cam.image_name))
        scene.test_cameras[res] = minicams_t
