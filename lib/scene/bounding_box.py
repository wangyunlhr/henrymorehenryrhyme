import math

import numpy as np
import torch
from lib.utils.general_utils import matrix_to_quaternion, build_rotation

from lib.utils.general_utils import build_rotation
class BoundingBox:
    def __init__(self, object_type, object_id, size):
        self.object_type = object_type
        self.object_id = object_id

        if isinstance(size, np.ndarray):
            size = torch.from_numpy(size)
        size = size.float().cuda()
        self.size = size  # size_x, size_y, size_z

        self.min_xyz, self.max_xyz = -self.size / 2.0, self.size / 2
        self.frame = {}

    def add_frame_waymo(
        self, frame, metadata, ego2world
    ):  # center_x, center_y, center_z, yaw
        pos = [float(metadata[1]), float(metadata[2]), float(metadata[3])]
        theta = float(metadata[7])

        if isinstance(ego2world, np.ndarray):
            ego2world = torch.from_numpy(ego2world)
        ego2world = ego2world.float().cuda()

        pos = torch.tensor(pos).float().cuda()
        T = ego2world[:3, :3] @ pos + ego2world[:3, 3]

        R = (
            torch.tensor(
                [
                    [math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            .float()
            .cuda()
        )
        R = ego2world[:3, :3] @ R

        quaternion: torch.Tensor = matrix_to_quaternion(R)
        quaternion = quaternion / torch.norm(quaternion)
        quaternion = quaternion.unsqueeze(0)

        dT = torch.zeros(3).float().cuda()
        dR = torch.eye(3).float().cuda()
        self.frame[frame] = (T, quaternion, dT, dR)

    def add_frame_kitti(self, frame, transform):
        if isinstance(transform, np.ndarray):
            transform = torch.from_numpy(transform)
        transform = transform.float().cuda()

        pos = transform[:3, 3]
        U, S, V = torch.linalg.svd(transform[:3, :3].cpu())  #!gpu修改到cpu上svd计算
        U = U.cuda()
        S = S.cuda()
        V = V.cuda()

        self.size = torch.max(torch.stack([S, self.size]), dim=0).values
        self.min_xyz, self.max_xyz = -self.size / 2.0, self.size / 2

        quaternion = matrix_to_quaternion(U[:3, :3])
        quaternion = quaternion / torch.norm(quaternion)
        quaternion = quaternion.unsqueeze(0)

        dT = torch.zeros(3).float().cuda()
        dR = torch.eye(3).float().cuda()
        self.frame[frame] = (pos, quaternion, dT, dR)

    #! 检查
    def get_corners(self, frame, to_numpy=True):
        if frame not in self.frame:
            raise KeyError(f"Frame {frame} not found in bounding box {self.object_id}")

        T, quat, dT, dR = self.frame[frame]  # T: (3,), quat: (1,4)

        # 四元数 -> 旋转矩阵 (3,3)
        R = build_rotation(quat)[0]  # world 下旋转

        # 局部坐标系下的 min/max（以 box 中心为原点的 box）
        min_x, min_y, min_z = self.min_xyz
        max_x, max_y, max_z = self.max_xyz

        # 按照 KITTI 约定次序构造 8 个角点 (local frame)
        local_corners = torch.stack([
            torch.tensor([max_x, max_y, min_z], device=T.device),  # 0 front-left-bottom
            torch.tensor([max_x, min_y, min_z], device=T.device),  # 1 front-right-bottom
            torch.tensor([min_x, min_y, min_z], device=T.device),  # 2 back-right-bottom
            torch.tensor([min_x, max_y, min_z], device=T.device),  # 3 back-left-bottom

            torch.tensor([max_x, max_y, max_z], device=T.device),  # 4 front-left-top
            torch.tensor([max_x, min_y, max_z], device=T.device),  # 5 front-right-top
            torch.tensor([min_x, min_y, max_z], device=T.device),  # 6 back-right-top
            torch.tensor([min_x, max_y, max_z], device=T.device),  # 7 back-left-top
        ], dim=0)  # (8,3)

        # 变到世界系
        world_corners = (R @ local_corners.T).T + T[None, :]

        if to_numpy:
            return world_corners.detach().cpu().numpy()
        return world_corners
