#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import socket
import json
from time import monotonic, sleep
import traceback
from typing import Optional, Tuple, Any
import numpy as np
import torch

from gaussian_renderer import render
from scene.cameras import MiniCam


class NetworkGUI:

    def __init__(self, host="127.0.0.1", port=6009):
        self.host = host
        self.port = port
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connection: Optional[socket.socket] = None
        self.address = None
        self.t_net_last_sent = 0
        self.bind_and_listen()

    def bind_and_listen(self):
        while True:
            try:
                self.listener.bind((self.host, self.port))
                break
            except OSError as e:
                if e.errno == 98:  # Port is already in use.
                    print(f"Port {self.port} is already in use. Trying the next port...")
                    self.port += 1
                else:
                    print(f"Could not bind to port {self.port}: {e}")
                    break

        self.listener.listen()
        self.listener.settimeout(0)  # Set non-blocking mode
        print(f"Listening on {self.host}:{self.port}")

    def _tick(self):
        if self.connection:
            return
        try:
            self.connection, self.address = self.listener.accept()
            print(f"\nConnected from {self.address}")
            self.connection.settimeout(None)
        except (socket.timeout, BlockingIOError):
            pass
        except Exception as e:
            if self.connection:
                self.connection.close()
            self.connection = None
            traceback.print_exc()

    def tick(self, opt, pipe, dataset, gaussians, iteration, background):
        self._tick()

        if not self.connection:
            return

        if monotonic() - self.t_net_last_sent < 1 / 75:
            return

        prev_sh, prev_ocnv = pipe.convert_SHs_python, pipe.compute_cov3D_python

        do_training = True
        keep_alive = False

        while self.connection:
            # pipe.convert_SHs_python
            try:
                d = self.receive()
                if d is not None:
                    custom_cam, do_training, pseudo_SHs_skyness, pipe.compute_cov3D_python, keep_alive, scaling_modifer = d

                    if custom_cam is not None:

                        with torch.no_grad():
                            override_color = None
                            if pseudo_SHs_skyness:
                                skyness = gaussians.get_skysphere
                                override_color = torch.zeros_like(skyness).repeat(1, 3)
                                override_color[:, 2] = skyness.squeeze()
                                override_color[:, 0] = 1 - skyness.squeeze()

                            image = render(custom_cam, gaussians, pipe, background, scaling_modifer, override_color)["render"]
                            image_npy = (torch.clamp(image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

                    else:
                        image_npy = None

                    self.send(image_npy, dataset.source_path)

                training_done = iteration >= int(opt.iterations)

                if (training_done and keep_alive) or not do_training:
                    sleep(1 / 60)
                    continue

                break

            except Exception as e:
                print(e)
                break

        pipe.convert_SHs_python, pipe.compute_cov3D_python = prev_sh, prev_ocnv

    def _read(self) -> Any:
        if not self.connection:
            return None

        message = None
        try:
            self.connection.settimeout(0)
            b = self.connection.recv(4)
            self.connection.settimeout(None)
            if len(b) < 4:
                b += self.connection.recv(4-len(b))
            message_length = int.from_bytes(b, 'little')

            if message_length:
                message = self.connection.recv(message_length)

        except (socket.timeout, BlockingIOError):
            return None

        except Exception as e:
            if self.connection:
                self.connection.close()
            self.connection = None

        if message:
            return json.loads(message.decode("utf-8"))


    def receive(self) -> Tuple[Optional[Any], bool, bool, bool, bool, float]:
        message = self._read()
        if message:

            width = message["resolution_x"]
            height = message["resolution_y"]

            do_training = bool(message["train"])
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]

            if width != 0 and height != 0:
                try:
                    fovy = message["fov_y"]
                    fovx = message["fov_x"]
                    znear = message["z_near"]
                    zfar = message["z_far"]
                    world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
                    world_view_transform[:, 1] = -world_view_transform[:, 1]
                    world_view_transform[:, 2] = -world_view_transform[:, 2]
                    full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
                    full_proj_transform[:, 1] = -full_proj_transform[:, 1]
                    custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
                except Exception as e:
                    print("")
                    traceback.print_exc()
                    raise e

            else:
                custom_cam = None

            return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier


    def send(self, image: np.ndarray, model_path_b: str):
        if self.connection:
            try:
                if image is not None:
                    img = memoryview(image)
                    self.connection.settimeout(None)
                    self.connection.sendall(img)

                model_path_b = model_path_b.encode("utf-8")
                self.connection.sendall(len(model_path_b).to_bytes(4, 'little'))
                self.connection.sendall(model_path_b)
                self.t_net_last_sent = monotonic()
            except Exception as e:
                print(e)
                self.connection.close()
                self.connection = None

    def close_connection(self):
        if self.connection:
            self.connection.close()
            self.connection = None


