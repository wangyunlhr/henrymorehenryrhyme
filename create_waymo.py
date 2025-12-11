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
import imageio
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# NOTE(2023/02/29): it's really important to set this! otherwise, the point cloud will be wrong. really wried.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUDA_USE_CUDA_DSA"] = "1"
import torch
import yaml
from lib import dataloader
# from lib.arguments import parse
# from lib.gaussian_renderer import raytracing
# from lib.scene import Scene
# from lib.scene.unet import UNet
# from lib.utils.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from lib.utils.console_utils import *
# from lib.utils.image_utils import mse, psnr
# from lib.utils.loss_utils import (
#     BinaryCrossEntropyLoss,
#     BinaryFocalLoss,
#     l1_loss,
#     l2_loss,
#     ssim,
# )
from lib.utils.record_utils import make_recorder
from ruamel.yaml import YAML
from tqdm import tqdm
from convert import lidar_to_pano, lidar_to_pano_with_beams, lidar_to_pano_with_beams_expand_scale, downsample_rangeimage_2x2_min
from lib.utils.image_utils import color_mapping
import open3d as o3d    
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import multiprocessing
from multiprocessing import Pool, current_process
from pathlib import Path
import fire, time, h5py
import pickle

colormap_ = 20

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




def process_one_frame(element, target_frame):
    # first_iter = 0

    # color = cv2.COLORMAP_JET

    # element = dataloader.load_scene(args.source_dir, args, test=False)

    lidar_H, lidar_W = 64, 2650  
    # for target_frame in range(1, element.train_lidar.num_frames):
    #! 保存点云累积点云
    all_means3D, all_intensitys, all_normals3D, all_labels3D = element.accumulate_frame([target_frame-2, target_frame + 2], target_frame)
    acc_all_information = np.concatenate([all_means3D.detach().cpu().numpy(), all_intensitys.detach().cpu().numpy(),\
                                            all_normals3D.detach().cpu().numpy(), all_labels3D.detach().cpu().numpy()], axis=-1) #C 3,1,3,1
    #!世界坐标系下acc_all_information, 其中labels是动态物体的标签，其余为0
    sensor2world = element.train_lidar.sensor2world[target_frame]
    #!1.转换成2D存储
    point_acc_sensor = inverse_transform_point(all_means3D.detach().cpu().numpy(), sensor2world.numpy())
    all_normals3D_sensor = world_to_lidar_points_normals(all_normals3D.detach().cpu().numpy(), sensor2world.numpy())
    all_information = np.concatenate([all_intensitys.detach().cpu().numpy(), all_normals3D_sensor, all_labels3D.detach().cpu().numpy()], axis=-1)
#     lidar_H, lidar_W = 64, 2650  
    pano_acc, pano_intensity = lidar_to_pano_with_beams(point_acc_sensor, all_information, lidar_H, lidar_W, element.train_lidar.inclination_bounds, \
                                                            element.train_lidar.angle_offset, element.train_lidar.pixel_offset, max_depth=80)
    
    pano_cat_return1_2 = np.concatenate([pano_acc[..., None], pano_intensity], axis=-1).astype(np.float32)


    #只用第一帧的点云
    all_means3D_1, all_intensitys_1, all_normals3D_1, all_labels3D_1 = element.accumulate_frame_return1([target_frame-2, target_frame +2], target_frame)
    acc_all_information_1 = np.concatenate([all_means3D_1.detach().cpu().numpy(), all_intensitys_1.detach().cpu().numpy(),\
                                            all_normals3D_1.detach().cpu().numpy(), all_labels3D_1.detach().cpu().numpy()], axis=-1) #C 3,1,3,1
    #!世界坐标系下acc_all_information, 其中labels是动态物体的标签，其余为0
    #!1.转换成2D存储
    point_acc_sensor_1 = inverse_transform_point(all_means3D_1.detach().cpu().numpy(), sensor2world.numpy())
    all_normals3D_sensor_1 = world_to_lidar_points_normals(all_normals3D_1.detach().cpu().numpy(), sensor2world.numpy())
    all_information_1 = np.concatenate([all_intensitys_1.detach().cpu().numpy(), all_normals3D_sensor_1, all_labels3D_1.detach().cpu().numpy()], axis=-1)
#     lidar_H, lidar_W = 64, 2650  
    pano_acc_1, pano_intensity_1 = lidar_to_pano_with_beams(point_acc_sensor_1, all_information_1, lidar_H, lidar_W, element.train_lidar.inclination_bounds, \
                                                            element.train_lidar.angle_offset, element.train_lidar.pixel_offset, max_depth=80)
    
    pano_cat_1 = np.concatenate([pano_acc_1[..., None], pano_intensity_1], axis=-1).astype(np.float32)



    gt_mask = element.train_lidar.get_mask(target_frame) #! h*w 只有range1
    gt_depth = element.train_lidar.get_depth(target_frame)
    #!1.1 为了验证lidar_to_pano的正确性
    _, _, gt_point_frame1, gt_intensity_frame1 = element.train_lidar.inverse_projection(target_frame)
    gt_point_frame1_sensor = inverse_transform_point(gt_point_frame1.detach().cpu().numpy(), sensor2world.numpy())
    pano_transpointgt_1, pano_transintensitygt_1 = lidar_to_pano_with_beams(gt_point_frame1_sensor, gt_intensity_frame1[:,None].detach().cpu().numpy(), lidar_H, lidar_W, element.train_lidar.inclination_bounds, \
                                                            element.train_lidar.angle_offset, element.train_lidar.pixel_offset, max_depth=80)

    gt_intensity = element.train_lidar.get_intensity(target_frame)
    gt_cat = np.concatenate([gt_depth[:,:, None], gt_intensity[:,:, None], gt_mask[:,:, None]], axis=-1)
    #!1.2如果不进行坐标系的反复转换
    _, _, gt_point_frame1_noworld, gt_intensity_frame1_noworld = element.train_lidar.inverse_projection_sensor(target_frame)
    pano_transpointgt_1_noworld, pano_transintensitygt_1_now = lidar_to_pano_with_beams(gt_point_frame1_noworld.detach().cpu().numpy(), gt_intensity_frame1_noworld[:,None].detach().cpu().numpy(), lidar_H, lidar_W, element.train_lidar.inclination_bounds, \
                                                            element.train_lidar.angle_offset, element.train_lidar.pixel_offset, max_depth=80)
    #!2.保存pose等矩阵信息
    pano_transpointgt_1_noworld_scale2, pano_transintensitygt_1_now_scale2 = lidar_to_pano_with_beams_expand_scale(gt_point_frame1_noworld.detach().cpu().numpy(), gt_intensity_frame1_noworld[:,None].detach().cpu().numpy(), lidar_H, lidar_W, element.train_lidar.inclination_bounds, \
                                                            element.train_lidar.angle_offset, element.train_lidar.pixel_offset, 2, 2, max_depth=80)
    
    pano_transpointgt_1_noworld_scale2_vis = (
        color_mapping(pano_transpointgt_1_noworld_scale2, colormap_) * 255
    ).astype(np.uint8)
    image_path = f'./scale_vis/{target_frame:4d}_scale2.jpg'
    imageio.imwrite(image_path, pano_transpointgt_1_noworld_scale2_vis)
    gt_depth_vis = (
        color_mapping(gt_depth.cpu().numpy(), colormap_) * 255
    ).astype(np.uint8)
    image_path = f'./scale_vis/{target_frame:4d}_gt.jpg'
    imageio.imwrite(image_path, gt_depth_vis)
    pano_downsample2, pano_intensity_downsample2 = downsample_rangeimage_2x2_min(pano_transpointgt_1_noworld_scale2, pano_transintensitygt_1_now_scale2)
    pano_downsample2_vis = (    
        color_mapping(pano_downsample2, colormap_) * 255
    ).astype(np.uint8)
    image_path = f'./scale_vis/{target_frame:4d}_scale2_downsample2.jpg'
    imageio.imwrite(image_path, pano_downsample2_vis)

    sensor_center = element.train_lidar.sensor_center[target_frame].cpu().numpy()
    angle_offset = element.train_lidar.angle_offset
    pixel_offset = element.train_lidar.pixel_offset
    ir = np.array(element.train_lidar.inclination_bounds)
    return {'pano_cat_return1_2': pano_cat_return1_2, 'pano_cat_1': pano_cat_1, 'gt_cat': gt_cat, 
            'angle_offset': angle_offset, 'pixel_offset': pixel_offset, 'ir': ir,
            'sensor_center': sensor_center, 'sensor2world': sensor2world}


def process_log(filename, output_dir: Path, n = None) :

    def create_group_data(group, save_dict):
        group.create_dataset('pano_cat_return1_2', data=(save_dict['pano_cat_return1_2']).astype(np.float32)) #! 保存return 1and2
        # group.create_dataset('pano_cat_1', data=(save_dict['pano_cat_1']).astype(np.float32))
        group.create_dataset('gt_cat', data=(save_dict['gt_cat']).astype(np.float32))
        group.create_dataset('ir', data=(save_dict['ir']).astype(np.float32))
        group.create_dataset('sensor_center', data=(save_dict['sensor_center']).astype(np.float32))
        group.create_dataset('sensor2world', data=(save_dict['sensor2world'].numpy()).astype(np.float64))
        group.attrs['angle_offset'] = np.float32(save_dict['angle_offset'])
        group.attrs['pixel_offset'] = np.float32(save_dict['pixel_offset'])

    element = dataloader.load_scene(filename, test=False)
    scene_id = filename.split('/')[-1].split('.')[0]
    for target_frame in range(5, element.train_lidar.num_frames-5):
        if target_frame not in [158, 168, 178]: #! 只弄val
            continue
        print("target_frame",target_frame)
        save_dict = process_one_frame(element, target_frame)

        key = str(target_frame)  # 你创建组时就是 f.create_group(str(timestamp))

        with h5py.File(output_dir/f'{scene_id}.h5', 'a') as f:
            # if key in f:                # 根目录下有该组
            #     del f[key]
            group = f.create_group(str(target_frame))
            create_group_data(group, save_dict)


def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)

def process_logs(data_dir: Path, output_dir: Path, nproc: int):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
       Args:
         data_dir: Argoverse 2.0 directory
         output_dir: Output directory.
    """
    if not (data_dir).exists():
        print(f'{data_dir} not found')
        return
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".tfrecord")]
    files.sort()  # 字典序
    create_120 = files[290:300]
    #! 保存路径到txt文件中
    # create_300_400 = files[400:500]
    # out_txt = "create_400_500.txt"
    # with open(out_txt, "w", encoding="utf-8") as f:
    #     f.write("\n".join(create_300_400) + "\n")  # 最后一行加换行更规范
    print('create: !!!!!! 300')
    logs = [os.path.join(data_dir, f) for f in create_120]
    #! 只是为了快速生成
    # logs = ["/data1/dataset/lidar-rt/save_forh5/waymo_perception_v1.4.3/dynamic/1/segment-1083056852838271990_4080_000_4100_000_with_camera_labels.tfrecord"]
    logs = ["/data1/dataset/lidar-rt/waymo_perception_v1.4.3/dynamic/1/segment-1083056852838271990_4080_000_4100_000_with_camera_labels.tfrecord"]
    args = sorted([(log, output_dir) for log in logs])
    print(f'Using {nproc} processes to process {len(args)} logs.')
    
    # for debug
    for x in tqdm(args):
        proc(x, ignore_current_process=True)
        # break

    # if nproc <= 1:
    #     for x in tqdm(args):
    #         proc(x, ignore_current_process=True)
    # else:
    #     with Pool(processes=nproc) as p:
    #         res = list(tqdm(p.imap_unordered(proc, args), total=len(args), ncols=100))


def create_reading_index(data_dir: Path, flow_inside_check=False):
    pkl_file_name = 'index_total.pkl' if not flow_inside_check else 'index_flow.pkl'
    start_time = time.time()
    data_index = []
    for file_name in tqdm(os.listdir(data_dir), ncols=100):
        if not file_name.endswith(".h5"):
            continue
        scene_id = file_name.split(".")[0]
        timestamps = []
        with h5py.File(data_dir/file_name, 'r') as f:
            if flow_inside_check:
                for key in f.keys():
                    if 'flow' in f[key]:
                        # print(f"Found flow in {scene_id} at {key}")
                        timestamps.append(key)
            else:
                timestamps.extend(f.keys())
        timestamps.sort(key=lambda x: int(x)) # make sure the timestamps are in order
        for timestamp in timestamps:
            data_index.append([scene_id, timestamp])



#! from txt读取路径

def _load_file_list(txt_path: Path):
    """从txt读取文件列表；忽略空行/注释行；去重；保留顺序。"""
    seen = set()
    files = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s).expanduser()
            # 允许相对路径（相对txt所在目录）
            if not p.is_absolute():
                p = (txt_path.parent / p).resolve()
            if p not in seen:
                seen.add(p)
                files.append(p)
    return files

def create_reading_index_split_from_txt(
    files_txt: Path,
    flow_inside_check: bool = False,
    test_digit: int = 7,
    train_pkl: str = "index_train.pkl",
    test_pkl: str  = "index_test.pkl",
    strict_suffix: bool = True,
):
    """
    从 files_txt（每行一个 .h5 路径，支持相对/绝对）读取文件清单，
    按时间戳最后一位是否等于 test_digit 划分为 test / train 并各自保存为 pkl。
    """
    t0 = time.time()
    index_train, index_test = [], []

    file_list = _load_file_list(Path(files_txt))
    if not file_list:
        raise FileNotFoundError(f"空的文件清单：{files_txt}")

    for p in tqdm(file_list, ncols=100, desc="Reading .h5 list"):
        if strict_suffix and p.suffix.lower() != ".h5":
            # 跳过非 .h5（或你也可以改为 continue+告警）
            continue
        if not p.exists():
            # 不存在则跳过
            continue

        scene_id = p.stem  # 文件名去掉后缀
        with h5py.File(p, "r") as f:
            ts_iter = [key for key in f.keys() if str(key).isdigit()]

        ts_iter.sort(key=lambda x: int(x))

        for ts in ts_iter:
            last_digit = int(ts) % 10
            pair = (scene_id, str(ts))
            if last_digit == test_digit:
                index_test.append(pair)
            else:
                index_train.append(pair)

    #（可选）全局排序
    index_train.sort(key=lambda x: (x[0], int(x[1])))
    index_test.sort(key=lambda x: (x[0], int(x[1])))

    # pkl 输出到 txt 同目录下
    out_dir = Path(files_txt).parent
    with open(out_dir / train_pkl, "wb") as f:
        pickle.dump(index_train, f)
    with open(out_dir / test_pkl, "wb") as f:
        pickle.dump(index_test, f)

    print(f"[Done] train: {len(index_train)}, test: {len(index_test)}, "
          f"saved to: {out_dir}, elapsed: {time.time()-t0:.2f}s")





def main(
    data_dir: str = "/data0/dataset/waymo_1_4_3/",
    mode: str = "training",
    output_dir: str ="/data1/dataset/debug/",
    nproc: int = (multiprocessing.cpu_count() - 1),
    create_index_only: bool = False, #! 只存储pkl文件
): #! 修改不同数量的累积帧，注意检查
    output_dir_ = Path(output_dir) / mode
    if create_index_only:
        # create_reading_index(Path(output_dir_)) #！
        create_reading_index_split_from_txt('/data0/code/LiDAR-RT-main_waymodata/files_for_train_part.txt')
        return
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(Path(data_dir) / mode, output_dir_, nproc)
    # create_reading_index(output_dir_)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")


#     # point = o3d.geometry.PointCloud()
#     # point.points = o3d.utility.Vector3dVector(point_acc_sensor)
#     # o3d.io.write_point_cloud(f"./11111pointacc_{frame}.ply", point)

#     # point_gt,_,_,_ = element.train_lidar.inverse_projection(frame)
#     # point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
#     # point = o3d.geometry.PointCloud()
#     # point.points = o3d.utility.Vector3dVector(point_gt_sensor)
#     # o3d.io.write_point_cloud(f"./11111pointgt_{frame}.ply", point)

#     # point_all = []
#     # for frame in list(range(frame-3, frame)) + list(range(frame+1, frame+3)):
#     #     point_gt,_,_,_ = element.train_lidar.inverse_projection(frame)
#     #     point_gt_sensor = inverse_transform_point(point_gt.detach().cpu().numpy(), sensor2world.numpy())
#     #     point_all.append(point_gt_sensor)
#     # point_all = np.concatenate(point_all, axis=0)
#     # point = o3d.geometry.PointCloud()
#     # point.points = o3d.utility.Vector3dVector(point_all)
#     # o3d.io.write_point_cloud(f"./11111pointall.ply", point)


# #!  保存数据
#     # train_frames = element.train_lidar.train_frames
#     # eval_frames = element.train_lidar.eval_frames
#     # frame_s, frame_e = args.frame_length[0], args.frame_length[1]
#     # lidar_H, lidar_W = 64, 2650  
#     # for frame in range(frame_s, frame_e):
#     #     if frame == frame_s: #跳过开头，从第一帧开始
#     #         continue
#     #     #! 保存变换信息
#     #     ir = np.array(element.train_lidar.inclination_bounds)
#     #     sensor_center = element.train_lidar.sensor_center[frame].cpu().numpy()
#     #     sensor2world = element.train_lidar.sensor2world[frame].cpu().numpy()
#     #     angle_offset = element.train_lidar.angle_offset
#     #     pixel_offset = element.train_lidar.pixel_offset

#         #! 保存点云
#     #     point_acc, intensity_acc = element.accumulate_frame([frame-1, frame +1], frame)
#     #     sensor2world = element.train_lidar.sensor2world[frame]

#     #     point_acc_sensor = inverse_transform_point(point_acc.detach().cpu().numpy(), sensor2world.numpy())

#     #     #! gt
#     #     gt_mask = element.train_lidar.get_mask(frame) #! h*w
#         # gt_depth = element.train_lidar.get_depth(target_frame)
#     #     gt_intensity = element.train_lidar.get_intensity(frame)

#     #     gt_cat = np.concatenate([gt_depth[:,:, None], gt_intensity[:,:, None], gt_mask[:,:, None]], axis=-1)


#     #     lidar_H, lidar_W = 64, 2650  
#     #     pano_acc, pano_intensity = lidar_to_pano_with_beams(point_acc_sensor, intensity_acc, lidar_H, lidar_W, element.train_lidar.inclination_bounds, element.train_lidar.angle_offset, element.train_lidar.pixel_offset, max_depth=80)
#     #     acc_cat= np.concatenate([pano_acc[:,:, None], pano_intensity[:,:, None]], axis=-1).astype(np.float32)
#     #     #!保存图像
#         # pano_vis = (
#         #         color_mapping(pano_acc, colormap_) * 255
#         #     ).astype(np.uint8)
#         # pano_image = cv2.cvtColor(pano_vis, cv2.COLOR_BGR2RGB)
#         # gt_vis = (
#         #         color_mapping(gt_depth.cpu().numpy(), colormap_) * 255
#         #     ).astype(np.uint8)
#         # gt_image = cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB)
#         # if frame in train_frames:
#         #     tag = "train"
#         # else: 
#         #     tag = "eval"
#         # data_type = 'waymo'
#     #     os.makedirs(os.path.join(f"./dataset_create/{data_type}/dynamic/1/{tag}/"), exist_ok=True)
#         # image_path = os.path.join(
#         #     f"./acc.jpg"
#         # )
#         # imageio.imwrite(image_path, pano_image)
#         # gt_image_path = os.path.join(
#         #     f"./gt.jpg"
#         # )
#         # imageio.imwrite(gt_image_path, gt_image)
#     #     np.save(f"./dataset_create/{data_type}/dynamic/1/{tag}/{frame}_acc.npy", acc_cat)  
#     #     np.save(f"./dataset_create/{data_type}/dynamic/1/{tag}/{frame}_gt.npy", gt_cat) 

#         # np.savez(f"./dataset_create/{data_type}/dynamic/1/{tag}/{frame}_information", ir = ir, sensor_center = sensor_center, sensor2world = sensor2world, angle_offset = angle_offset, pixel_offset = pixel_offset) 

#     # print("Done dataset create")
#     gaussians_assets = scene.gaussians_assets
#     scene.training_setup(args.opt)
#     log = {
#         "depth_mse": [],
#         "points_num": [],
#         "clone_sum": [],
#         "split_sum": [],
#         "prune_scale_sum": [],
#         "prune_opacity_sum": [],
#     }
#     scene_id = str(args.scene_id) if isinstance(args.scene_id, int) else args.scene_id
#     output_dir = os.path.join(
#         args.model_dir, args.task_name, args.exp_name, "scene_" + scene_id
#     )
#     record_dir = os.path.join(output_dir, "records")
#     recorder = make_recorder(args, record_dir)
#     print(
#         blue(
#             f"Task: {args.task_name}, Experiment: {args.exp_name}, Scene: {args.scene_id}"
#         )
#     )
#     print("Output dir: ", output_dir)

#     if args.model_path:
#         (model_params, first_iter) = torch.load(args.model_path)
#         scene.restore(model_params, args.opt)
#         with open(os.path.join(output_dir, "logs/log.json"), "r") as json_file:
#             log = json.load(json_file)
#     print("Continuing from iteration ", first_iter)

#     # bg_color = [1, 1, 1] if args.model.white_background else [0, 0, 0]
#     background = torch.tensor(
#         [0, 0, 1], device="cuda"
#     ).float()  # background (intensity, hit prob, drop prob)

#     BFLoss = BinaryFocalLoss()
#     BCELoss = BinaryCrossEntropyLoss()
#     frame_stack = []

#     iter_start = torch.cuda.Event(enable_timing=True)
#     iter_end = torch.cuda.Event(enable_timing=True)

#     ema_loss_for_log = 0.0
#     progress_bar = tqdm(
#         initial=first_iter, total=args.opt.iterations, desc="Training progress"
#     )
#     first_iter += 1

#     end = time.time()
#     frame_s, frame_e = args.frame_length[0], args.frame_length[1]
#     render_cams = []
#     best_mix_metric = 0
    

#     for iteration in range(first_iter, args.opt.iterations + 1):
#         if args.only_refine:
#             break
#         iter_start.record()
#         recorder.step += 1

#         scene.update_learning_rate(iteration)

#         # Every 1000 its we increase the levels of SH up to a maximum degree
#         if iteration % 1000 == 0:
#             scene.oneupSHdegree()

#         # Pick a random frame #! 改成遍历fram
#         if not frame_stack:
#             frame_stack = list(scene.train_lidar.train_frames)
#             random.shuffle(frame_stack)
#         frame = frame_stack.pop()
#         data_time = time.time() - end
#         # for frame in scene.train_lidar.eval_frames:
#             # Render
#         if args.pipe.debug_from and (iteration - 1) == args.pipe.debug_from:
#             args.pipe.debug = True

#         render_pkg = raytracing(
#             frame, gaussians_assets, scene.train_lidar, background, args
#         )
#         batch_time = time.time() - end
#         depth = render_pkg["depth"]
#         intensity = render_pkg["intensity"]
#         raydrop_prob = render_pkg["raydrop"]
#         means3d = render_pkg["means3D"]  #!用于保存 world坐标系

        
#         #! 保存真实点云
#         # point = o3d.geometry.PointCloud()
#         # point.points = o3d.utility.Vector3dVector(point_gt_one_sensor)
#         # o3d.io.write_point_cloud(f"./point_gt_one_sensor_{frame}.ply", point)
#         # point.points = o3d.utility.Vector3dVector(check_point_2_sensor )    
#         # o3d.io.write_point_cloud(f"./check_point_2_sensor _{frame}.ply", point)

#         # lidar_H, lidar_W = 64, 2650              # 举例（按你的数据设）
#         # fov_up, fov_total = 2.3, 20.3          # 举例（度），根据你的雷达设
#         # lidar_K = (fov_up, fov_total)
#         # pano_1 = lidar_to_pano(point_gt_one_sensor, lidar_H, lidar_W, lidar_K,max_depth=80)

#         acc_wet = render_pkg["accum_gaussian_weight"]

#         H, W = depth.shape[0], depth.shape[1]

#         gt_mask = scene.train_lidar.get_mask(frame).cuda()

#         # === Depth loss ===
#         depth = depth.squeeze(-1)
#         gt_depth = scene.train_lidar.get_depth(frame).cuda()
#         loss_depth = args.opt.lambda_depth_l1 * l1_loss(
#             depth[gt_mask], gt_depth[gt_mask]
#         )

#         # === Intensity loss ===
#         intensity = intensity.squeeze(-1)
#         gt_intensity = scene.train_lidar.get_intensity(frame).cuda()
#         loss_intensity = (
#             args.opt.lambda_intensity_l1
#             * l1_loss(intensity[gt_mask], gt_intensity[gt_mask])
#             + args.opt.lambda_intensity_l2
#             * l2_loss(intensity[gt_mask], gt_intensity[gt_mask])
#             + args.opt.lambda_intensity_dssim
#             * (
#                 1
#                 - ssim(
#                     (intensity * gt_mask).unsqueeze(0),
#                     (gt_intensity * gt_mask).unsqueeze(0),
#                 )
#             )
#         )

#         # === Raydrop loss ===
#         raydrop_prob = raydrop_prob.reshape(-1, 1)
#         labels_idx = (
#             ~gt_mask
#         )  # (1, h, w) notice: hit is true (1). apply ~ to make idx 0 represent hit
#         labels = labels_idx.reshape(-1, 1)  # (h*w, 1)

#         loss_raydrop = args.opt.lambda_raydrop_bce * BCELoss(labels, preds=raydrop_prob)

#         # === CD loss ===
#         chamLoss = chamfer_3DDist()
#         gt_pts = scene.train_lidar.inverse_projection_with_range(
#             frame, gt_depth, gt_mask
#         )
#         pred_pts = scene.train_lidar.inverse_projection_with_range(
#             frame, depth, gt_mask
#         )

#         dist1, dist2, _, _ = chamLoss(pred_pts[None, ...], gt_pts[None, ...])
#         chamfer_loss = (dist1 + dist2).mean() * 0.5
#         loss_cd = args.opt.lambda_cd * chamfer_loss

#         # === regularization loss ===
#         loss_reg = 0
#         for gaussians in gaussians_assets:
#             loss_reg += args.opt.lambda_reg * gaussians.box_reg_loss()

#         loss = loss_depth + loss_intensity + loss_raydrop + loss_cd + loss_reg
#         loss.backward()

#         with torch.no_grad():
#             densify_info = scene.optimize(
#                 args, iteration, means3d.grad, acc_wet, None, None
#             )

#             points_num = 0
#             for i in gaussians_assets:
#                 points_num += i.get_local_xyz.shape[0]
#             depth_mse = mse(depth[gt_mask], gt_depth[gt_mask]).mean().item()
#             clone_sum = (
#                 densify_info[0] + log["clone_sum"][-1]
#                 if log["clone_sum"]
#                 else densify_info[0]
#             )
#             split_sum = (
#                 densify_info[1] + log["split_sum"][-1]
#                 if log["split_sum"]
#                 else densify_info[1]
#             )
#             prune_scale_sum = (
#                 densify_info[2] + log["prune_scale_sum"][-1]
#                 if log["prune_scale_sum"]
#                 else densify_info[2]
#             )
#             prune_opacity_sum = (
#                 densify_info[3] + log["prune_opacity_sum"][-1]
#                 if log["prune_opacity_sum"]
#                 else densify_info[3]
#             )
#             log["depth_mse"].append(depth_mse)
#             log["points_num"].append(points_num)
#             log["clone_sum"].append(clone_sum)
#             log["split_sum"].append(split_sum)
#             log["prune_scale_sum"].append(prune_scale_sum)
#             log["prune_opacity_sum"].append(prune_opacity_sum)

#             # prepare loss stats for tensorboard record
#             loss_stats = {
#                 "all_loss": loss,
#                 "depth_loss": loss_depth,
#                 "intensity_loss": loss_intensity,
#                 "ema_loss": 0.4 * loss + 0.6 * ema_loss_for_log,
#                 "points_num": torch.tensor(points_num).float(),
#                 "depth_mse": torch.tensor(depth_mse).float(),
#             }

#             reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
#             recorder.update_loss_stats(reduced_losses)

#             end = time.time()
#             recorder.batch_time.update(batch_time)
#             recorder.data_time.update(data_time)
#             recorder.record("train")

#             if iteration % args.visual_interval == 0:
#                 render_pkg = raytracing(
#                     frame_s, gaussians_assets, scene.train_lidar, background, args
#                 )
#                 rendered_depth = render_pkg["depth"]
#                 rendered_intensity = render_pkg["intensity"]

#                 rendered_depth = (rendered_depth - rendered_depth.min()) / (
#                     rendered_depth.max() - rendered_depth.min()
#                 )
#                 rendered_depth = rendered_depth.cpu().numpy()
#                 rendered_depth = np.uint8(rendered_depth * 255)
#                 rendered_depth = cv2.applyColorMap(rendered_depth, color)

#                 rendered_intensity = rendered_intensity.clamp(0, 1)
#                 rendered_intensity = (rendered_intensity - rendered_intensity.min()) / (
#                     rendered_intensity.max() - rendered_intensity.min()
#                 )
#                 rendered_intensity = rendered_intensity.cpu().numpy()
#                 rendered_intensity = np.uint8(rendered_intensity * 255)
#                 rendered_intensity = cv2.applyColorMap(rendered_intensity, color)

#                 concat_image = np.concatenate(
#                     [rendered_depth, rendered_intensity], axis=0
#                 )
#                 rgb_image = concat_image
#                 os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
#                 cv2.imwrite(
#                     os.path.join(output_dir, "images", str(iteration) + ".png"),
#                     rgb_image,
#                 )
#                 render_cams.append(rgb_image)

#             # Progress bar
#             ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
#             if iteration % 10 == 0:
#                 progress_bar.set_postfix(
#                     {
#                         "Loss": f"{ema_loss_for_log.item():.{5}f}",
#                         # "L_all": f"{loss.item():.{5}f}",
#                         # "L_depth": f"{loss_depth.item():.{5}f}",
#                         # "L_intensity": f"{loss_intensity.item():.{5}f}",
#                         # "L_raydrop": f"{loss_raydrop.item():.{5}f}",
#                         "points": f"{points_num}",
#                         "exp": args.exp_name,
#                         "scene": args.scene_id,
#                     }
#                 )
#                 progress_bar.update(10)
#             if iteration == args.opt.iterations:
#                 progress_bar.close()

#             # Log and save
#             if iteration in args.saving_iterations:
#                 progress_bar.write("\n[ITER {}] Saving Gaussians".format(iteration))
#                 scene.save(iteration, "model_it_" + str(iteration))

#             if iteration % args.testing_iterations == 0:
#                 if iteration >= args.saving_iterations[0] - 3000:
#                     mix_metric = 0
#                     for frame in scene.train_lidar.eval_frames:
#                         render_pkg = raytracing(
#                             frame, gaussians_assets, scene.train_lidar, background, args
#                         )
#                         depth = render_pkg["depth"].detach()
#                         intensity = render_pkg["intensity"].detach()
#                         raydrop_prob = render_pkg["raydrop"].detach()
#                         mask = raydrop_prob < 0.5

#                         gt_depth = scene.train_lidar.get_depth(frame).cuda()
#                         gt_intensity = scene.train_lidar.get_intensity(frame).cuda()
#                         gt_mask = scene.train_lidar.get_mask(frame).cuda()
#                         psnr_depth = (
#                             psnr(
#                                 depth[..., 0] * mask[..., 0] / 80,
#                                 gt_depth * gt_mask / 80,
#                             )
#                             .mean()
#                             .item()
#                         )
#                         intensity = intensity.clamp(0, 1)
#                         gt_intensity = gt_intensity.clamp(0, 1)
#                         psnr_intensity = (
#                             psnr(
#                                 intensity[..., 0] * mask[..., 0], gt_intensity * gt_mask
#                             )
#                             .mean()
#                             .item()
#                         )
#                         mix_metric += psnr_depth + psnr_intensity
#                     mix_metric /= len(scene.train_lidar.eval_frames)
#                     print(mix_metric, best_mix_metric)
#                     if mix_metric > best_mix_metric:
#                         for file in os.listdir(scene.model_save_dir):
#                             if file.endswith(".pth") and "ckpt_it_" in file:
#                                 os.remove(os.path.join(scene.model_save_dir, file))
#                         best_mix_metric = mix_metric
#                         scene.save(iteration, "ckpt_it_" + str(iteration) + "_good")
#                 else:
#                     previous_checkpoint_nopfix = os.path.join(
#                         scene.model_save_dir,
#                         "ckpt_it_" + str(iteration - args.testing_iterations) + ".pth",
#                     )
#                     if os.path.exists(previous_checkpoint_nopfix):
#                         os.remove(previous_checkpoint_nopfix)

#                     progress_bar.write(
#                         "\n[ITER {}] Saving Checkpoint".format(iteration)
#                     )
#                     scene.save(iteration, "ckpt_it_" + str(iteration))

#                 logging(log, output_dir)

#         iter_end.record()

#     if args.refine.use_refine:
#         print(output_dir)
#         in_channels = 9 if args.refine.use_spatial else 3
#         unet = UNet(in_channels=in_channels, out_channels=1).cuda()
#         unet_optimizer = torch.optim.Adam(unet.parameters(), lr=0.001)
#         for epoch in tqdm(range(0, args.refine.epochs), desc="Refine raydrop"):
#             for iter in range(0, args.refine.batch_size):
#                 if not frame_stack:
#                     frame_stack = list(scene.train_lidar.train_frames)
#                     random.shuffle(frame_stack)
#                 frame = frame_stack.pop()

#                 render_pkg = raytracing(
#                     frame, gaussians_assets, scene.train_lidar, background, args
#                 )
#                 depth = render_pkg["depth"].detach()
#                 intensity = render_pkg["intensity"].detach()
#                 raydrop_prob = render_pkg["raydrop"].detach()

#                 H, W = depth.shape[0], depth.shape[1]
#                 input_depth = depth.reshape(1, H, W)
#                 input_intensity = intensity.reshape(1, H, W)
#                 input_raydrop = raydrop_prob.reshape(1, H, W)
#                 raydrop_prob = torch.cat(
#                     [input_raydrop, input_intensity, input_depth], dim=0
#                 )
#                 if args.refine.use_spatial:
#                     ray_o, ray_d = scene.train_lidar.get_range_rays(frame)
#                     raydrop_prob = torch.cat(
#                         [raydrop_prob, ray_o.permute(2, 0, 1), ray_d.permute(2, 0, 1)],
#                         dim=0,
#                     )
#                 raydrop_prob = raydrop_prob.unsqueeze(0)
#                 if args.refine.use_rot:
#                     rot = torch.randint(0, W, (1,))
#                     raydrop_prob = torch.cat(
#                         [raydrop_prob[:, :, :, rot:], raydrop_prob[:, :, :, :rot]],
#                         dim=-1,
#                     )
#                 raydrop_prob = unet(raydrop_prob)

#                 raydrop_prob = raydrop_prob.reshape(-1, 1)

#                 gt_mask = scene.train_lidar.get_mask(frame).cuda()
#                 labels_idx = (
#                     ~gt_mask
#                 )  # (1, h, w) notice: hit is true (1). apply ~ to make idx 0 represent hit
#                 if args.refine.use_rot:
#                     labels_idx = torch.cat(
#                         [labels_idx[:, rot:], labels_idx[:, :rot]], dim=-1
#                     )
#                 labels = labels_idx.reshape(-1, 1)  # (h*w, 1)
#                 loss_raydrop = args.refine.lambda_raydrop_bce * BCELoss(
#                     labels, preds=raydrop_prob
#                 )

#                 loss_raydrop.backward()

#             unet_optimizer.step()
#             unet_optimizer.zero_grad()

#         torch.save(unet.state_dict(), os.path.join(output_dir, "models", "unet.pth"))


# def logging(log, output_dir):
#     indices = range(len(log["depth_mse"]))

#     fig, ax1 = plt.subplots(figsize=(8, 6))
#     color = "tab:blue"
#     ax1.set_ylabel("Depth MSE", color=color)
#     ax1.plot(indices, log["depth_mse"], color=color)
#     ax1.tick_params(axis="y", labelcolor=color)
#     ax2 = ax1.twinx()
#     color = "tab:red"

#     ax2.set_ylabel("Points Num", color=color)
#     clone_sum = np.array(log["clone_sum"])
#     split_sum = np.array(log["split_sum"])
#     prune_scale_sum = np.array(log["prune_scale_sum"])
#     prune_opacity_sum = np.array(log["prune_opacity_sum"])

#     plt.fill_between(indices, 0, clone_sum, label="clone_sum", color="blue", alpha=0.5)
#     plt.fill_between(
#         indices,
#         clone_sum,
#         clone_sum + split_sum,
#         label="split_sum",
#         color="green",
#         alpha=0.5,
#     )
#     plt.fill_between(
#         indices,
#         clone_sum + split_sum,
#         clone_sum + split_sum + prune_scale_sum,
#         label="prune_scale_sum",
#         color="red",
#         alpha=0.5,
#     )
#     plt.fill_between(
#         indices,
#         clone_sum + split_sum + prune_scale_sum,
#         clone_sum + split_sum + prune_scale_sum + prune_opacity_sum,
#         label="prune_opacity_sum",
#         color="yellow",
#         alpha=0.5,
#     )

#     ax2.plot(indices, log["points_num"], color=color)
#     ax2.tick_params(axis="y", labelcolor=color)

#     log_dir = os.path.join(output_dir, "logs")
#     os.makedirs(log_dir, exist_ok=True)
#     plt.savefig(os.path.join(log_dir, "log.png"))
#     plt.close()
#     with open(os.path.join(log_dir, "log.json"), "w") as json_file:
#         json.dump(log, json_file, indent=4)


# if __name__ == "__main__":
    # # Set up command line argument parser
    # parser = argparse.ArgumentParser(description="launch args")
    # parser.add_argument("-dc", "--data_config_path", type=str, help="config path")
    # parser.add_argument("-ec", "--exp_config_path", type=str, help="config path")
    # parser.add_argument("-m", "--model", type=str, help="the path to a checkpoint")
    # parser.add_argument(
    #     "-r",
    #     "--only_refine",
    #     action="store_true",
    #     help="skip the training. only refine the model. E.g. load a checkpoint and only refine the unet to fit the checkpoint",
    # )
    # launch_args = parser.parse_args()

    # args = parse(launch_args.exp_config_path)
    # args = parse(launch_args.data_config_path, args)
    # args.model_path = launch_args.model
    # args.only_refine = launch_args.only_refine

    # if not os.path.exists(args.model_dir):
    #     os.makedirs(args.model_dir)

    # if args.seed is not None:
    #     set_seed(args.seed)

    # # Start GUI server, configure and run training
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(args)

    # # All done
    # print(blue("\nTraining complete."))
