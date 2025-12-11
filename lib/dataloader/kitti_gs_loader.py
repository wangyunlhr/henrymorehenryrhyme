import os
import random
from typing import Dict

import numpy as np
import open3d as o3d
import torch
from lib.scene import BoundingBox, GaussianModel, LiDARSensor, Scene, dataset_readers
from lib.utils import general_utils
from lib.utils.camera_utils import cameraList_from_camInfos
from lib.utils.general_utils import build_rotation
from lib.utils.graphics_utils import BasicPointCloud
from PIL import Image


class SceneLidar_all(Scene):
    def __init__(self, waymo_raw_pkg, shuffle=True, resize_ratio=1, test=False, frames = None):


        self.loaded_iter = None
        self.camera_extent = 0
        self.gaussians_assets = [
            GaussianModel(
                2, 3, extent=self.camera_extent
            )
        ]

        lidar: Dict[int, LiDARSensor] = waymo_raw_pkg[0]
        bboxes: Dict[str, BoundingBox] = waymo_raw_pkg[1]
        frame_range = frames 

        self.train_lidar = lidar

        print("[Loaded] background guassians")

        # initialize objects with bounding boxes
        if True:
            obj_ids = list(bboxes.keys())
            for obj_id in obj_ids:
                bbox = bboxes[obj_id]
                # general_utils.fill_zeros_with_previous_nonzero(
                #     range(frame_range[0], frame_range[1] + 1), bbox.frame
                # )
                _, first_frame, last_frame = general_utils.fill_zeros_with_previous_nonzero_new(
                   range(frame_range[0], frame_range[1] + 1), bbox.frame
                ) #! 只填补物体出现第一帧到最后一帧之间的跳帧
                abs_velocities = []
                for frame in range(first_frame, last_frame):
                    velocity = bbox.frame[frame + 1][0] - bbox.frame[frame][0]
                    abs_velocities.append(torch.norm(velocity).item())
                avg_velocity = torch.tensor(abs_velocities).mean().item()

                if avg_velocity > 0.01 and bbox.object_type == 1:
                    extent = (
                        torch.norm(bbox.size, keepdim=False).item()
                        * 4
                    ) 
                    dynamic_flag = True
                else:
                    extent = (
                        torch.norm(bbox.size, keepdim=False).item()
                        * 4
                    ) 
                    dynamic_flag = False
                    #!修改成全部的box都保存，但是只有动态车，行人，自行车人进行warp
                gaussian_model = GaussianModel(
                    2,
                    3,
                    extent=extent,
                    bounding_box=bbox,
                    dynamic_flag=dynamic_flag,
                )
                gaussian_model.tmp_points_intensities_list = []
                self.gaussians_assets.append(gaussian_model)

    def accumulate_frame(self, frame_list, target_frame):
        # initialize bkgd points
        all_points = []
        all_intensity = []
        all_normal = []
        all_label = [] #! 0为背景,其他为各自的类别

        # for frame in range(frame_range[0], frame_range[1] + 1): #! 转换成只有训练帧
        #! 清空gaussian_model.tmp_points_intensities_list
        for gaussian_model in self.gaussians_assets[1:]:
            gaussian_model.tmp_points_intensities_list = []

        for frame in frame_list:
            # lidar_pts, lidar_intensity, _ ,_, = self.train_lidar.inverse_projection(frame) #! 全部世界坐标系下点云
            lidar_pts, lidar_intensity = self.train_lidar.direct_point[frame], self.train_lidar.direct_intensity[frame]#! kitti直接添加点云
            sensor2world = self.train_lidar.sensor2world[frame]
            lidar_pts = lidar_pts @ sensor2world[:3, :3].T + sensor2world[:3, 3] #! 全部世界坐标系下点云

            points_lidar = o3d.geometry.PointCloud()
            points_lidar.points = o3d.utility.Vector3dVector(
                lidar_pts.cpu().numpy().astype(np.float64)
            )
            points_lidar.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=6)
            )
            normals = torch.from_numpy(np.asarray(points_lidar.normals)).float()
            for gaussian_model in self.gaussians_assets[1:]: #!处理不同的box
                bbox = gaussian_model.bounding_box
                #!只过滤动态物体，只有动态物体用box对齐
                if not gaussian_model.dynamic_flag:
                    continue
                if frame not in bbox.frame.keys(): #如果这个box在这帧没有，就跳过
                    continue
                T = bbox.frame[frame][0].cpu()
                R = build_rotation(bbox.frame[frame][1])[0].cpu() #! box to world
                points_in_local = (lidar_pts - T) @ R.inverse().T
                normals_in_local = normals @ R.inverse().T
                mask = (torch.abs(points_in_local) < bbox.size.cpu() / 2).all(dim=1)
                gaussian_model.tmp_points_intensities_list.append( #! label只标记了动态物体，因为只有动态物体需要进行累积，所以构建了gs
                    (
                        points_in_local[mask],
                        lidar_intensity[mask][..., None],
                        normals_in_local[mask],
                        (bbox.object_type * torch.ones(points_in_local[mask].shape[0]))[..., None],
                    )
                )

                lidar_pts, lidar_intensity = lidar_pts[~mask], lidar_intensity[~mask]
                normals = normals[~mask]

                

            all_points.append(lidar_pts)
            all_intensity.append(lidar_intensity)
            all_normal.append(normals)


        all_points = torch.cat(all_points, dim=0)  #! 静态区域的点和normal还是在世界坐标系下
        all_intensity = torch.cat(all_intensity, dim=0)[..., None]
        all_normal = torch.cat(all_normal, dim=0)
        all_label = torch.zeros(all_points.shape[0])[..., None] #! 0为背景,其他为各自的类别


        #!静态区域添加点云
        self.gaussians_assets[0].xyz_set(all_points, all_intensity, all_normal, all_label)
        # # initialize objects points
        # points_num = args.model.obj_pt_num
        for j, gaussian_model in enumerate(self.gaussians_assets[1:]):
            if not gaussian_model.tmp_points_intensities_list:
                continue
            print(f"[Loaded] object {j+1} points for frame {target_frame} ")
            points = torch.cat(
                [point for point, _, _, _ in gaussian_model.tmp_points_intensities_list],
                dim=0,
            )
            intensities = torch.cat(
                [
                    intensitie
                    for _, intensitie, _, _ in gaussian_model.tmp_points_intensities_list
                ],
                dim=0,
            )
            normals = torch.cat(
                [normal for _, _, normal, _ in gaussian_model.tmp_points_intensities_list],
                dim=0,
            )
            labels = torch.cat(
                [label for _, _, _, label in gaussian_model.tmp_points_intensities_list],
                dim=0,
            )

            gaussian_model.xyz_set(points, intensities, normals, labels) #动态点云加载
            #! 动态区域也不过滤
        #! warp到目标帧
        all_means3D = []
        all_intensitys = []
        all_normals3D = []
        all_labels3D = []

        for i, gaussian_model in enumerate(self.gaussians_assets):
            if i >= 1:
                if not gaussian_model.tmp_points_intensities_list:
                    continue
                bbox = gaussian_model.bounding_box
                if target_frame not in bbox.frame.keys(): #如果这个box在这帧没有，就跳过
                    continue
            # means3D = gaussian_model.get_world_xyz(target_frame)
            means3D, normals3D = gaussian_model.get_world_xyz_and_normals(target_frame)
            if i == 0:
                means3D = means3D.cuda()
                normals3D = normals3D.cuda()
            all_means3D.append(means3D)
            all_intensitys.append(gaussian_model._intensity)
            all_normals3D.append(normals3D)
            all_labels3D.append(gaussian_model._labels)

        all_means3D = torch.cat(all_means3D, dim=0)
        all_intensitys = torch.cat(all_intensitys, dim=0) #!全部是世界坐标系下的结果，才能够对齐
        all_normals3D = torch.cat(all_normals3D, dim=0)
        all_labels3D = torch.cat(all_labels3D, dim=0)

        assert all_intensitys.max()< 1.1
        return all_means3D, all_intensitys, all_normals3D, all_labels3D


