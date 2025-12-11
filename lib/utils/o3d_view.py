'''
# @date: 2023-1-26 16:38
# @author: Qingwen Zhang  (https://kin-zhang.github.io/)
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# @detail:
#  1. Play the data you want in open3d, and save the view control to json file.
#  2. Use the json file to view the data again.
#  3. Save the screen shot and view file for later check and animation.
# 
# code gits: https://gist.github.com/Kin-Zhang/77e8aa77a998f1a4f7495357843f24ef
# 
# CHANGELOG:
# 2024-08-23 21:41(Qingwen): remove totally on view setting from scratch but use open3d>=0.18.0 version for set_view from json text func.
# 2024-04-15 12:06(Qingwen): show a example json text. add hex_to_rgb, color_map_hex, color_map (for color points if needed)
# 2024-01-27 0:41(Qingwen): update MyVisualizer class, reference from kiss-icp: https://github.com/PRBonn/kiss-icp/blob/main/python/kiss_icp/tools/visualizer.py
'''

import open3d as o3d
import os, time
from typing import List, Callable
from functools import partial
import numpy as np
import torch

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

color_map_hex = ['#a6cee3', '#de2d26', '#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00',\
                 '#cab2d6','#6a3d9a','#ffff99','#b15928', '#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3',\
                 '#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']

color_map = [hex_to_rgb(color) for color in color_map_hex]

        
class MyVisualizer:
    def __init__(self, view_file=None, window_title="Default", save_folder="./logs/imgs"):
        self.params = None
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name=window_title)
        self.view_file = view_file

        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True
        self.save_img_folder = save_folder
        os.makedirs(self.save_img_folder, exist_ok=True)
        self.static_assets = []
        print(
            f"\n{window_title.capitalize()} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t[ESC/Q] to exit\n"
            "\t    [P] to save screen and viewpoint\n"
            "\t    [N] to step\n"
        )
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback(["P"], self._save_screen)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["N"], self._next_frame)

    def show(self, assets: List):
        self.vis.clear_geometries()

        for asset in assets:
            self.vis.add_geometry(asset)
            if self.view_file is not None:
                self.vis.set_view_status(open(self.view_file).read())

        self.vis.update_renderer()
        self.vis.poll_events()
        self.vis.run()
        self.vis.destroy_window()
    
    #! 添加固定资产，不更新
    def add_static_asset(self, asset):
        """添加静态资产，并存储在 self.static_assets 中"""
        self.static_assets.append(asset)
        self.vis.add_geometry(asset)

    def create_bbox_from_vertices(self, vertices):
        """根据8个角点创建 Bounding Box 的 LineSet"""
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧边
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)  # 设置角点
        line_set.lines = o3d.utility.Vector2iVector(lines)  # 连接边
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # 设为红色
        
        return line_set


    def create_bbox_from_vertices_green(self, vertices):
        """根据8个角点创建 Bounding Box 的 LineSet"""
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧边
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)  # 设置角点
        line_set.lines = o3d.utility.Vector2iVector(lines)  # 连接边
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(lines))])  # 设为绿色
        
        return line_set



    def create_text_mesh(self, text, position, font_size=0.2):
        """创建文本的三角面片表示 (3D 文字标签)"""
        text_3d = o3d.geometry.TriangleMesh.create_sphere(radius=font_size * 0.1)  # 用小球代替文字
        text_3d.translate(position)  # 将文字放在指定位置
        text_3d.paint_uniform_color([0, 1, 1])  # 设为绿色
        return text_3d


    def update(self, assets: List, bbox_vertices_list: np.ndarray = None, bbox_vertices_list_green: np.ndarray = None, bbox_ids: list = None, clear: bool = True, name: str = 'example'):
        # print("Updating Visualizer...")
        # self._save_screen(self.vis, name)
        # from IPython import embed; embed()
            # 在更新前保存当前的相机参数

        if clear:
            self.vis.clear_geometries()
            for static_asset in self.static_assets:
                self.vis.add_geometry(static_asset, reset_bounding_box=False) 

        for asset in assets:
            self.vis.add_geometry(asset, reset_bounding_box=False)
            self.vis.update_geometry(asset)

        # 处理 N * 8 * 3 的 bounding box 数据
        if bbox_vertices_list is not None:
            assert len(bbox_vertices_list.shape) == 3 and bbox_vertices_list.shape[1] == 8 and bbox_vertices_list.shape[2] == 3, \
                "bbox_vertices_list must be (N, 8, 3) in shape"
            
            for i, bbox_vertices in enumerate(bbox_vertices_list):
                bbox = self.create_bbox_from_vertices(bbox_vertices)
                self.vis.add_geometry(bbox, reset_bounding_box=False)

        if bbox_vertices_list_green is not None:
            assert len(bbox_vertices_list_green.shape) == 3 and bbox_vertices_list_green.shape[1] == 8 and bbox_vertices_list_green.shape[2] == 3, \
                "bbox_vertices_list_green must be (N, 8, 3) in shape"

            for i, bbox_vertices in enumerate(bbox_vertices_list_green):
                bbox = self.create_bbox_from_vertices_green(bbox_vertices)
                self.vis.add_geometry(bbox, reset_bounding_box=False)

                # 计算 ID 位置 (顶面中心)
                # from IPython import embed; embed()
                # top_center = torch.mean(bbox_vertices, axis=0) + torch.tensor([0,0,3])
                
                # # 创建文本 3D 标识
                # if bbox_ids is not None and i < len(bbox_ids):
                #     print(f"Creating text for bbox {i} with ID {bbox_ids[i]}")
                #     text_mesh = self.create_text_mesh(str(bbox_ids[i]), top_center)
                #     self.vis.add_geometry(text_mesh, reset_bounding_box=False)

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            if self.view_file is not None:
                self.vis.set_view_status(open(self.view_file).read())
            self.reset_bounding_box = False

        self.vis.update_renderer()

        while self.block_vis:
            self.vis.poll_events()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _quit(self, vis):
        print("Destroying Visualizer. Thanks for using ^v^.")
        vis.destroy_window()
        os._exit(0)

    def _save_screen(self, vis, name):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        png_file = f"{self.save_img_folder}/{name}.png"
        view_json_file = f"{self.save_img_folder}/ScreenView_{name}.json"
        # from IPython import embed; embed()
        with open(view_json_file, 'w') as f:
            f.write(vis.get_view_status())
        vis.capture_screen_image(png_file)
        print(f"ScreenShot saved to: {png_file}, Please check it.")

if __name__ == "__main__":
    json_content = """{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 3.9660897254943848, 2.427476167678833, 2.55859375 ],
			"boundingbox_min" : [ 0.55859375, 0.83203125, 0.56663715839385986 ],
			"field_of_view" : 60.0,
			"front" : [ 0.27236083595988803, -0.25567329763523589, -0.92760484038816615 ],
			"lookat" : [ 2.4114965637897101, 1.8070288935660688, 1.5662280268112718 ],
			"up" : [ -0.072779625398507866, -0.96676294585190281, 0.24509698622097265 ],
			"zoom" : 0.47999999999999976
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
"""
    # write to json file
    view_json_file = "view.json"
    with open(view_json_file, 'w') as f:
        f.write(json_content)
    sample_ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.path)

    viz = MyVisualizer(view_json_file, window_title="Qingwen's View")
    viz.show([pcd])