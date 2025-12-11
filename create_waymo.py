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
    all_means3D, all_intensitys, all_normals3D, all_labels3D = element.accumulate_frame([target_frame-1, target_frame + 1], target_frame)
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
    all_means3D_1, all_intensitys_1, all_normals3D_1, all_labels3D_1 = element.accumulate_frame_return1([target_frame-1, target_frame +1], target_frame)
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
    # _, _, gt_point_frame1, gt_intensity_frame1 = element.train_lidar.inverse_projection(target_frame)
    # gt_point_frame1_sensor = inverse_transform_point(gt_point_frame1.detach().cpu().numpy(), sensor2world.numpy())
    # pano_transpointgt_1, pano_transintensitygt_1 = lidar_to_pano_with_beams(gt_point_frame1_sensor, gt_intensity_frame1[:,None].detach().cpu().numpy(), lidar_H, lidar_W, element.train_lidar.inclination_bounds, \
    #                                                         element.train_lidar.angle_offset, element.train_lidar.pixel_offset, max_depth=80)

    gt_intensity = element.train_lidar.get_intensity(target_frame)
    gt_cat = np.concatenate([gt_depth[:,:, None], gt_intensity[:,:, None], gt_mask[:,:, None]], axis=-1)
    # #!1.2如果不进行坐标系的反复转换
    # _, _, gt_point_frame1_noworld, gt_intensity_frame1_noworld = element.train_lidar.inverse_projection_sensor(target_frame)
    # pano_transpointgt_1_noworld, pano_transintensitygt_1_now = lidar_to_pano_with_beams(gt_point_frame1_noworld.detach().cpu().numpy(), gt_intensity_frame1_noworld[:,None].detach().cpu().numpy(), lidar_H, lidar_W, element.train_lidar.inclination_bounds, \
    #                                                         element.train_lidar.angle_offset, element.train_lidar.pixel_offset, max_depth=80)
    # #!2.保存pose等矩阵信息
    # pano_transpointgt_1_noworld_scale2, pano_transintensitygt_1_now_scale2 = lidar_to_pano_with_beams_expand_scale(gt_point_frame1_noworld.detach().cpu().numpy(), gt_intensity_frame1_noworld[:,None].detach().cpu().numpy(), lidar_H, lidar_W, element.train_lidar.inclination_bounds, \
    #                                                         element.train_lidar.angle_offset, element.train_lidar.pixel_offset, 2, 2, max_depth=80)
    
    # pano_transpointgt_1_noworld_scale2_vis = (
    #     color_mapping(pano_transpointgt_1_noworld_scale2, colormap_) * 255
    # ).astype(np.uint8)
    # image_path = f'./scale_vis/{target_frame:4d}_scale2.jpg'
    # imageio.imwrite(image_path, pano_transpointgt_1_noworld_scale2_vis)
    # gt_depth_vis = (
    #     color_mapping(gt_depth.cpu().numpy(), colormap_) * 255
    # ).astype(np.uint8)
    # image_path = f'./scale_vis/{target_frame:4d}_gt.jpg'
    # imageio.imwrite(image_path, gt_depth_vis)
    # pano_downsample2, pano_intensity_downsample2 = downsample_rangeimage_2x2_min(pano_transpointgt_1_noworld_scale2, pano_transintensitygt_1_now_scale2)
    # pano_downsample2_vis = (    
    #     color_mapping(pano_downsample2, colormap_) * 255
    # ).astype(np.uint8)
    # image_path = f'./scale_vis/{target_frame:4d}_scale2_downsample2.jpg'
    # imageio.imwrite(image_path, pano_downsample2_vis)

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

    element = dataloader.load_scene(filename, args=None, test=False)
    scene_id = filename.split('/')[-1].split('.')[0]
    for target_frame in range(1, element.train_lidar.num_frames-1):
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

def process_logs(data_dir: Path, output_dir: Path, start_idx: int , clip_len: int, nproc: int):
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
    create_120 = files[start_idx:start_idx + clip_len]
    #! 保存路径到txt文件中
    # create_300_400 = files[400:500]
    # out_txt = "create_400_500.txt"
    # with open(out_txt, "w", encoding="utf-8") as f:
    #     f.write("\n".join(create_300_400) + "\n")  # 最后一行加换行更规范

    logs = [os.path.join(data_dir, f) for f in create_120]

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
    data_dir: str = "/data0/dataset/waymo_1_4_3/", #!原始路径
    mode: str = "training",
    output_dir: str ="/data1/dataset/debug/", #!输出路径
    start_idx: int = 0, #!要修改多次
    clip_len: int = 120, #!要修改多次
    nproc: int = (multiprocessing.cpu_count() - 1),
    create_index_only: bool = False, #! 只存储pkl文件
): #! 修改不同数量的累积帧，注意检查
    output_dir_ = Path(output_dir) / mode
    if create_index_only:
        # create_reading_index(Path(output_dir_)) #！
        create_reading_index_split_from_txt('/data0/code/LiDAR-RT-main_waymodata/files_for_train_part.txt')
        return
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(Path(data_dir) / mode, output_dir_, start_idx, clip_len, nproc)
    # create_reading_index(output_dir_)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")


