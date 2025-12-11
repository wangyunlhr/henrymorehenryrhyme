import os

import numpy
import torch
from lib.dataloader import kitti_loader, waymo_loader
from lib.dataloader.gs_loader import SceneLidar
from lib.dataloader.kitti_gs_loader import SceneLidar_all
from lib.utils.console_utils import *


def load_scene(data_dir, args, test=False):
    if "waymo" in data_dir:
        print(blue("\n====== [Loading] Waymo Open Dataset ======"))
        lidars, bboxes = waymo_loader.load_waymo_raw(data_dir, args)
    elif "KITTI" in data_dir:
        print(blue("\n====== [Loading] KITTI Dataset ======"))
        lidars, bboxes = kitti_loader.load_kitti_raw(data_dir, args)
    else:
        raise ValueError("Error: invalid dataset")

    print(blue("------------"))
    scene = SceneLidar(args, (lidars, bboxes), test=test)
    return scene

def load_scene_all(data_dir, frame_length, seq, test=False):
    if "waymo" in data_dir:
        print(blue("\n====== [Loading] Waymo Open Dataset ======"))
        lidars, bboxes = waymo_loader.load_waymo_raw(data_dir, args)
    elif "KITTI" in data_dir:
        print(blue("\n====== [Loading] KITTI Dataset ======"))
        lidars, bboxes, frames = kitti_loader.load_kitti_raw_all(data_dir, frame_length, seq = seq)
    else:
        raise ValueError("Error: invalid dataset")

    print(blue("------------"))
    scene = SceneLidar_all((lidars, bboxes), test=test, frames=frames)
    return scene, bboxes
