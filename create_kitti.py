# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import argparse
import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_USE_CUDA_DSA"] = "1"
import torch
import yaml
from lib import dataloader
from lib.arguments import parse
from lib.gaussian_renderer import raytracing
from lib.scene import Scene
from lib.scene.unet import UNet
from lib.utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from lib.utils.console_utils import *
from lib.utils.image_utils import mse, psnr
from lib.utils.loss_utils import (
    BinaryCrossEntropyLoss,
    BinaryFocalLoss,
    l1_loss,
    l2_loss,
    ssim,
)
from lib.utils.record_utils import make_recorder
from ruamel.yaml import YAML
from tqdm import tqdm
from lib.utils.kitti_utils import kitti_points_to_range_image
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from lib.utils.image_utils import color_mapping
colormap_ = 20
import imageio
import open3d as o3d

from lib.utils.o3d_view import MyVisualizer
import multiprocessing
from pathlib import Path

from multiprocessing import Pool, current_process
from pathlib import Path
import fire, time, h5py
import pickle

def set_seed(seed):
    """
    Useless function, result still have a 1e-7 difference.
    Need to test problem in optix.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def inverse_transform_point(Pw, T_ego2world):
    R = T_ego2world[:3, :3]
    t = T_ego2world[:3, 3]
    Pe = (Pw - t) @ R      # 等价于 R^T (Pw - t)
    return Pe 


def world_to_lidar_points_normals(Nw: np.ndarray, T_w_l: np.ndarray) -> np.ndarray:
    """
    将世界坐标系下的法线 Nw 变换到 LiDAR 坐标系（刚体：仅旋转）。
    约定：Nw 为 (N,3) 行向量，并采用右乘；T_w_l 为 LiDAR->World 外参 [R_wl | t_wl]。

    Args:
        Nw: (N, 3) 世界系法线（不要求单位长度）
        T_w_l: (4, 4) 外参矩阵，LiDAR->World

    Returns:
        Nl: (N, 3) LiDAR 系法线（单位长度）
    """
    R = T_w_l[:3, :3]               # R_w<-l
    Nl = Nw @ R                     # 行向量右乘：n_l = n_w * R
    norm = np.linalg.norm(Nl, axis=1, keepdims=True)
    Nl = Nl / (norm + 1e-12)        # 归一化
    return Nl


def process_one_kitti_chunk(args) :

    def create_group_data(group, save_dict):
        group.create_dataset('pano_cat_return1_2', data=(save_dict['pano_cat_return1_2']).astype(np.float32)) #! 
        # group.create_dataset('pano_cat_1', data=(save_dict['pano_cat_1']).astype(np.float32))
        group.create_dataset('gt_cat', data=(save_dict['gt_cat']).astype(np.float32))
        group.create_dataset('sensor_center', data=(save_dict['sensor_center']).astype(np.float32))
        group.create_dataset('sensor2world', data=(save_dict['sensor2world'].numpy()).astype(np.float64))


    seq, chunk_start, chunk_end, output_dir, data_dir, last_flag, start_flag = args[0], args[1], args[2], args[3], args[4], args[5], args[6]

    if last_flag:
        scene, bboxes = dataloader.load_scene_all(data_dir, [chunk_start-3, chunk_end-1], seq, test=False)
    else:
        scene, bboxes = dataloader.load_scene_all(data_dir, [chunk_start, chunk_end], seq, test=False)

    if start_flag: #!单独处理第一帧
        name_start = f'{(chunk_start):05d}'
        target_frame = chunk_start
        all_means3D, all_intensitys, all_normals3D, all_labels3D = scene.accumulate_frame([target_frame+1, target_frame +2], target_frame)
        sensor2world = scene.train_lidar.sensor2world[target_frame]
        sensor_center = scene.train_lidar.sensor_center[target_frame].cpu().numpy()
        #!gt数据
        gt_depth = scene.train_lidar.get_depth(target_frame)
        gt_intensity = scene.train_lidar.get_intensity(target_frame)
        gt_mask = scene.train_lidar.get_mask(target_frame)
        gt_cat = np.concatenate([gt_depth[:,:, None], gt_intensity[:,:, None], gt_mask[:,:, None]], axis=-1)

        point_acc_sensor = inverse_transform_point(all_means3D.detach().cpu().numpy(), sensor2world.numpy())


        all_normals3D_sensor = world_to_lidar_points_normals(all_normals3D.detach().cpu().numpy(), sensor2world.numpy())
        all_information = np.concatenate([all_intensitys.detach().cpu().numpy(), all_normals3D_sensor, all_labels3D.detach().cpu().numpy()], axis=-1)
        generate_range_image = kitti_points_to_range_image(point_acc_sensor, all_information)

        save_dict = {'pano_cat_return1_2': generate_range_image, 'gt_cat': gt_cat, 
            'sensor_center': sensor_center, 'sensor2world': sensor2world}


        key = str(target_frame)  # 你创建组时就是 f.create_group(str(timestamp))
        with h5py.File(output_dir+f'{seq}_{name_start}_{(chunk_end - 1):05d}.h5', 'a') as f:
            # if key in f:                # 根目录下有该组
            #     del f[key]
            group = f.create_group(str(target_frame))
            create_group_data(group, save_dict)
    else:
        name_start = f'{(chunk_start + 1):05d}'

    for target_frame in range(chunk_start + 1, chunk_end):
        print('target_frame', target_frame)
        if target_frame == chunk_end - 1 and last_flag:
            all_means3D, all_intensitys, all_normals3D, all_labels3D = scene.accumulate_frame([target_frame-1, target_frame -2], target_frame)
        else:
            all_means3D, all_intensitys, all_normals3D, all_labels3D = scene.accumulate_frame([target_frame-1, target_frame +1], target_frame)
        sensor2world = scene.train_lidar.sensor2world[target_frame]
        sensor_center = scene.train_lidar.sensor_center[target_frame].cpu().numpy()
        #!gt数据
        gt_depth = scene.train_lidar.get_depth(target_frame)
        gt_intensity = scene.train_lidar.get_intensity(target_frame)
        gt_mask = scene.train_lidar.get_mask(target_frame)
        gt_cat = np.concatenate([gt_depth[:,:, None], gt_intensity[:,:, None], gt_mask[:,:, None]], axis=-1)

        point_acc_sensor = inverse_transform_point(all_means3D.detach().cpu().numpy(), sensor2world.numpy())


        all_normals3D_sensor = world_to_lidar_points_normals(all_normals3D.detach().cpu().numpy(), sensor2world.numpy())
        all_information = np.concatenate([all_intensitys.detach().cpu().numpy(), all_normals3D_sensor, all_labels3D.detach().cpu().numpy()], axis=-1)
        generate_range_image = kitti_points_to_range_image(point_acc_sensor, all_information)

        save_dict = {'pano_cat_return1_2': generate_range_image, 'gt_cat': gt_cat, 
            'sensor_center': sensor_center, 'sensor2world': sensor2world}


        key = str(target_frame)  # 你创建组时就是 f.create_group(str(timestamp))

        with h5py.File(output_dir+f'{seq}_{name_start}_{(chunk_end - 1):05d}.h5', 'a') as f:
            # if key in f:                # 根目录下有该组
            #     del f[key]
            group = f.create_group(str(target_frame))
            create_group_data(group, save_dict)



def debug_check(args):

    seq, chunk_start, chunk_end, output_dir, data_dir, last_flag = args[0], args[1], args[2], args[3], args[4], args[5]

    scene, bboxes = dataloader.load_scene_all(data_dir, [chunk_start, chunk_end], seq, test=False)
    target_frame = args.frame_length[0] + 1
    all_means3D, all_intensitys, all_normals3D, all_labels3D = scene.accumulate_frame([target_frame-1, target_frame +1], target_frame)
    sensor2world = scene.train_lidar.sensor2world[target_frame]
    gt_depth = scene.train_lidar.get_depth(target_frame)
    point_acc_sensor = inverse_transform_point(all_means3D.detach().cpu().numpy(), sensor2world.numpy())
    generate_range_image = kitti_points_to_range_image(point_acc_sensor, all_intensitys.detach().cpu().numpy())

    #! 发现物体错位检查box transform问题
    # viz = MyVisualizer(view_file=None, window_title="BBox Demo")

    # for frame in range(args.frame_length[0], args.frame_length[1]+1):
    #     corner_list = []
    #     for box_id in bboxes:
    #         if frame in bboxes[box_id].frame.keys():
    #             corner = bboxes[box_id].get_corners(frame, to_numpy=True)  # (8,3) cuda tensor
    #             corner_list.append(corner)

    #     sensor2world = scene.train_lidar.sensor2world[frame]
    #     ##corner转换到sensor
    #     boxes_sensor = []
    #     for corner_world in corner_list:_
    #         corner_sensor = inverse_transform_point(corner_world, sensor2world.numpy())
    #         boxes_sensor.append(corner_sensor)

    #     print(f'target{frame}', np.array(boxes_sensor).shape)
    #     point_gt = scene.train_lidar.direct_point[frame]
    #     point_gt_sensor = point_gt.detach().cpu().numpy()
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(point_gt_sensor)
    #     pcd.paint_uniform_color([0.5, 0.5, 0.5])


    #     # 这里用 update 而不是 show，因为你 update 里处理了 bbox_vertices_list
    #     print(frame,"="*10)
    #     viz.update(
    #         assets=[pcd],
    #         bbox_vertices_list=np.array(boxes_sensor) if len(boxes_sensor) > 0 else None,  # 变成 (1,8,3)
    #         bbox_vertices_list_green=None,  # 或者也传一个绿色框的 list
    #         bbox_ids=None,
    #         clear=True,
    #         name="checkbox"
    #     )

    #!检查变换前后复原问题 是正常的
    # point_gt_sensor = scene.train_lidar.direct_point[target_frame]
    # point_gt_sensor_2range_image = kitti_points_to_range_image(point_gt_sensor.cpu().numpy(), torch.ones_like(point_gt_sensor)[:,0].cpu().numpy())
    # point_gt_sensor_restore = scene.train_lidar.range2point(target_frame, point_gt_sensor_2range_image[:,:,0])
    # point_gt_sensor_restore_sensor = inverse_transform_point(point_gt_sensor_restore.reshape(-1,3).detach().cpu().numpy(), sensor2world.numpy())


    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/checkorigt_{target_frame}.ply", point)
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor_restore_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/checkrestoregt_{target_frame}.ply", point)



    #! check the range image
    # gt_depth_vis = (
    #     color_mapping(gt_depth.cpu().numpy(), colormap_) * 255
    # ).astype(np.uint8)
    # os.makedirs('./scale_vis', exist_ok=True)
    # image_path = f'./scale_vis/{target_frame:4d}_gt.jpg'
    # imageio.imwrite(image_path, gt_depth_vis)
    # generate_range_image_vis = (
    #     color_mapping(generate_range_image[:,:,0], colormap_) * 255
    # ).astype(np.uint8)
    # image_path = f'./scale_vis/{target_frame:4d}_generate_range_image.jpg'
    # imageio.imwrite(image_path, generate_range_image_vis)


    # #! check point 直接点云，在ego坐标系下
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_acc_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/pointacc_{target_frame}.ply", point)

    # point_gt = scene.train_lidar.direct_point[target_frame]
    # point_gt_sensor = point_gt.detach().cpu().numpy()
    # # point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/dpointgt_{target_frame}.ply", point)

    # point_gt = scene.train_lidar.direct_point[target_frame-1]
    # point_gt_sensor = point_gt.detach().cpu().numpy()
    # # point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/dpointgt_{target_frame-1}.ply", point)

    # point_gt = scene.train_lidar.direct_point[target_frame+1]
    # point_gt_sensor = point_gt.detach().cpu().numpy()
    # # point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/dpointgt_{target_frame+1}.ply", point)

    #! rangeimage 转换的点云
    # point_gt,_ = scene.train_lidar.inverse_projection(target_frame)
    # sensor2world = scene.train_lidar.sensor2world[target_frame]
    # point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/newpointgt_{target_frame}.ply", point)

    # point_gt,_ = scene.train_lidar.inverse_projection(target_frame-1)
    # sensor2world = scene.train_lidar.sensor2world[target_frame-1]
    # point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/newpointgt_{target_frame-1}.ply", point)

    # point_gt,_ = scene.train_lidar.inverse_projection(target_frame+1)
    # sensor2world = scene.train_lidar.sensor2world[target_frame+1]
    # point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/newpointgt_{target_frame+1}.ply", point)

    # point_gt,_ = scene.train_lidar.inverse_projection(target_frame-2)
    # sensor2world = scene.train_lidar.sensor2world[target_frame-2]
    # point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
    # point = o3d.geometry.PointCloud()
    # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
    # o3d.io.write_point_cloud(f"./scale_vis/newpointgt_{target_frame-2}.ply", point)


def build_kitti_task_list(data_dir, seq_list, output_dir, chunk_size=202, overlap=1):

    assert 0 <= overlap < chunk_size, "overlap 必须小于 chunk_size 且非负，否则会死循环"

    tasks = []
    for seq in seq_list:
        full_seq = f"2013_05_28_drive_{seq}_sync"
        data_dir_seq = os.path.join(
            data_dir, "data_3d_raw", full_seq, "velodyne_points", "data"
        )
        # 帧数 N，索引范围 [0, N-1]
        current_seq_frames = len(os.listdir(data_dir_seq))

        # 只取 .bin 文件
        bin_files = [f for f in os.listdir(data_dir_seq) if f.endswith(".bin")]

        # 防止空目录
        if not bin_files:
            raise RuntimeError(f"No .bin files found in {data_dir_seq}")

        # 文件名转成整数编号
        indices = [int(os.path.splitext(f)[0]) for f in bin_files]

        # 路径中最小的为 frame_s，最大的为 frame_e+1（右开区间）
        frame_s = min(indices)
        frame_e = max(indices) + 1   # 范围 [frame_s, frame_e)


        cur = frame_s
        while cur < frame_e:
            chunk_start = cur
            chunk_end   = min(chunk_start + chunk_size, frame_e)  # [start, end)
            if chunk_end == frame_e: #! 最后一帧单独处理一下
                last_flag = True
            else:
                last_flag = False
            
            if chunk_start == frame_s: #! 第一段单独处理一下
                start_flag = True
            else:
                start_flag = False

            tasks.append((seq, chunk_start, chunk_end, output_dir, data_dir, last_flag, start_flag))
            if last_flag:
                break

            # 下一段的起点 = 本段终点 - overlap
            cur = chunk_end - overlap

            # 防止因为 overlap 导致死循环
            if cur <= chunk_start:
                break
    return tasks




def process_kitti_all(data_dir, output_dir, seq_list, start_id, clip_size, nproc):
    """
    多进程处理整个 KITTI 数据集
    data_dir: 类似 args.source_dir，里面有 0000, 0001, ...
    output_dir: 结果保存目录
    nproc: 进程数
    """
    os.makedirs(output_dir, exist_ok=True)


    tasks = build_kitti_task_list(data_dir, seq_list, output_dir, chunk_size=202, overlap=1)
    sorted_tasks = sorted(tasks, key=lambda x: (x[0], x[1]))[start_id:start_id+clip_size]


    print(f"Using {nproc} processes to process {len(tasks)} (seq, frame) pairs.")

    #! for debug
    # t = sorted_tasks[-1]
    # process_one_kitti_chunk(t)
    # if nproc <= 1:
    for t in tqdm(sorted_tasks):
        process_one_kitti_chunk(t)
    # else: 
    #     with Pool(processes=nproc) as p:
    #         list(tqdm(p.imap_unordered(process_one_kitti_chunk, sorted_tasks[:2]),
    #                   total=len(tasks), ncols=100))


def main_kitti(
    data_dir: str = "/data1/dataset/KITTI-360/",
    output_dir: str = "/data0/dataset/KITTI-360_h5/0002/",
    seq_list: List[str] = ['0002'],
    start_id: int = 0,
    clip_size: int = 20,
    nproc: int = (multiprocessing.cpu_count() - 1),
):
    # data_dir 可以和 args.source_dir 相同，也可以单独传
    process_kitti_all(data_dir, output_dir, seq_list, start_id, clip_size, nproc)

def create_train_reading_index(data_dir: Path, pkl_file_name: str = 'index_train.pkl'):
    """
    data_dir/
        seq_id/
            clip_id_XXX.h5
    h5 内部的 key 为 timestamp
    """

    data_index = []

    # 遍历第一级：seq
    for seq_name in tqdm(os.listdir(data_dir), ncols=100, desc="seq"):
        if seq_name == '0000':
            continue
        seq_path = data_dir + seq_name

        # 遍历第二级：clip(.h5)
        for clip_name in os.listdir(seq_path):
            if not clip_name.endswith(".h5"):
                continue

            clip_path = seq_path + '/' + clip_name
            # 场景 id 可以按你需要来拼，这里示例：seq + '_' + clip
            scene_id = f"{clip_name.split('.')[0]}"

            timestamps = []
            with h5py.File(clip_path, 'r') as f:
                timestamps.extend(f.keys())

            # key 是字符串的 timestamp，这里转成 int 排序
            timestamps.sort(key=lambda x: int(x))

            for timestamp in timestamps:
                data_index.append([seq_name, scene_id, timestamp])

    with open(f'./{pkl_file_name}', "wb") as f:
        pickle.dump(data_index, f)

    return data_index



full_list = [
    2360, 2370, 2380, 2390,
    4960, 4970, 4980, 4990,
    8130, 8140, 8150, 8160,
    10210, 10220, 10230, 10240,
    10760, 10770, 10780, 10790,
    11410, 11420, 11430, 11440,
    1551, 1564, 1577, 1590,
    1741, 1754, 1767, 1780,
    1921, 1934, 1947, 1960,
    3366, 3379, 3392, 3405
]


def create_test_reading_index(data_dir: Path, pkl_file_name: str = 'index_test.pkl'):
    """
    data_dir/
        seq_id/
            clip_id_XXX.h5
    h5 内部的 key 为 timestamp
    """

    data_index = []

    # 遍历第一级：seq

    seq_name = '0000'

    seq_path = data_dir + seq_name

    # 遍历第二级：clip(.h5)
    for clip_name in os.listdir(seq_path):
        if not clip_name.endswith(".h5"):
            continue

        clip_path = seq_path + '/' + clip_name
        # 场景 id 可以按你需要来拼，这里示例：seq + '_' + clip
        scene_id = f"{clip_name.split('.')[0]}"

        timestamps = []
        with h5py.File(clip_path, 'r') as f:
            for timestamp in f.keys():
                if int(timestamp) in full_list:
                    timestamps.append(timestamp)
        # key 是字符串的 timestamp，这里转成 int 排序
        if len(timestamps) == 0:
            continue
        timestamps.sort(key=lambda x: int(x))

        for timestamp in timestamps:
            data_index.append([seq_name, scene_id, timestamp])

    with open(f'./{pkl_file_name}', "wb") as f:
        pickle.dump(data_index, f)

    return data_index

if __name__ == "__main__":
    seq_list_all = sorted(['0000','0002','0003','0004','0005','0006','0007','0009','0010'])
    #! 生成数据
    # main_kitti()
    seq = seq_list_all[:1] #!不同的seq,要修改多次
    main_kitti(
        data_dir = "/data1/dataset/KITTI-360/", #!原始路径
        output_dir = f"/data0/dataset/debug/{seq[0]}/", 
        seq_list = seq,
        start_id = 0, #!要修改多次0,20,40.....
        clip_size = 20,
        nproc = (multiprocessing.cpu_count() - 1),
    )
    #! 生成索引
    # create_train_reading_index(data_dir = '/data0/dataset/KITTI-360_h5/', pkl_file_name = 'index_train.pkl')
    # create_test_reading_index(data_dir = '/data0/dataset/KITTI-360_h5/', pkl_file_name = 'index_test.pkl')



#! old version
# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = argparse.ArgumentParser(description="launch args")
#     parser.add_argument("-dc", "--data_config_path", type=str, help="config path")
#     parser.add_argument("-ec", "--exp_config_path", type=str, help="config path")
#     parser.add_argument("-m", "--model", type=str, help="the path to a checkpoint")
#     parser.add_argument(
#         "-r",
#         "--only_refine",
#         action="store_true",
#         help="skip the training. only refine the model. E.g. load a checkpoint and only refine the unet to fit the checkpoint",
#     )
#     launch_args = parser.parse_args()

#     args = parse(launch_args.exp_config_path)
#     args = parse(launch_args.data_config_path, args)
#     args.model_path = launch_args.model
#     args.only_refine = launch_args.only_refine

#     if not os.path.exists(args.model_dir):
#         os.makedirs(args.model_dir)

#     if args.seed is not None:
#         set_seed(args.seed)

#     # Start GUI server, configure and run training
#     torch.autograd.set_detect_anomaly(args.detect_anomaly)
#     training(args)

#     # All done
#     print(blue("\nTraining complete."))
