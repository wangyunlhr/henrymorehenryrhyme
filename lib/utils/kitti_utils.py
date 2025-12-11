import numpy as np
import math
def LiDAR_2_Pano_KITTI(
    local_points_with_intensities, lidar_H, lidar_W, intrinsics, max_depth=80.0
):
    pano, intensities = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=intrinsics,
        max_depth=max_depth,
    )
    range_view = np.zeros((lidar_H, lidar_W, 3))
    range_view[:, :, 1] = intensities
    range_view[:, :, 2] = pano
    return range_view

def lidar_to_pano_with_intensities(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    lidar_K: int,
    max_depth=80,
):
    """
    Convert lidar frame to pano frame with intensities.
    Lidar points are in local coordinates.

    Args:
        local_points: (N, 4), float32, in lidar frame, with intensities.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
        intensities: (H, W), float32.
    """
    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3]
    fov_up, fov = lidar_K
    fov_down = fov - fov_up

    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W))
    for local_points, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_points
        beta = np.pi - np.arctan2(y, x)
        alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        c = int(round(beta / (2 * np.pi / lidar_W)))
        r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))

        # Check out-of-bounds.
        if r >= lidar_H or r < 0 or c >= lidar_W or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity

    return pano, intensities


def lidar_to_pano(
    local_points: np.ndarray, lidar_H: int, lidar_W: int, lidar_K: int, max_dpeth=80
):
    """
    Convert lidar frame to pano frame. Lidar points are in local coordinates.

    Args:
        local_points: (N, 3), float32, in lidar frame.
        lidar_H: pano height.
        lidar_W: pano width.
        lidar_K: lidar intrinsics.
        max_depth: max depth in meters.

    Return:
        pano: (H, W), float32.
    """

    # (N, 3) -> (N, 4), filled with zeros.
    local_points_with_intensities = np.concatenate(
        [local_points, np.zeros((local_points.shape[0], 1))], axis=1
    )
    pano, _ = lidar_to_pano_with_intensities(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        lidar_K=lidar_K,
        max_dpeth=max_dpeth,
    )
    return pano

#!
def kitti_points_to_range_image(xyzs,
                          intensities,
                          H=66,
                          W=1030,
                          inc_bottom=math.radians(-24.9),
                          inc_top=math.radians(2.0),
                          azimuth_left=np.pi,
                          azimuth_right=-np.pi,
                          max_depth=80.0):
    """
    将 LiDAR 点云投影为 range image + intensity image

    Args:
        xyzs:        (N, 3) 点坐标（在 LiDAR 坐标系）
        intensities: (N,)   对应每个点的强度
        H, W:        range image 的高度和宽度
        inc_bottom:  垂直下边界（弧度）
        inc_top:     垂直上边界（弧度）
        azimuth_left, azimuth_right: 水平角范围（左/右，弧度）
        max_depth:   最大深度，超过则丢弃

    Returns:
        range_map:      (H, W) 距离图
        intensity_map:  (H, W) 强度图
    """

    # 水平/垂直角分辨率
    h_res = (azimuth_right - azimuth_left) / W
    v_res = (inc_bottom - inc_top) / H

    # 初始化为 -1，表示空
    C = intensities.shape[1]
    range_map = np.ones((H, W), dtype=np.float32) * -1
    intensity_map = np.ones((H, W, C), dtype=np.float32) * -1

    # 预计算距离
    dists = np.linalg.norm(xyzs, axis=1)

    for xyz, intensity, dist in zip(xyzs, intensities, dists):
        x, y, z = xyz

        # 计算水平角 azimuth 和俯仰角 inclination
        azimuth = np.arctan2(y, x)
        inclination = np.arctan2(z, np.sqrt(x**2 + y**2))

        # 深度过滤
        if dist > max_depth:
            continue

        # 投影到像素坐标
        w_idx = np.round((azimuth - azimuth_left) / h_res).astype(int)
        h_idx = np.round((inclination - inc_top) / v_res).astype(int)

        # 越界丢弃
        if (w_idx < 0) or (w_idx >= W) or (h_idx < 0) or (h_idx >= H):
            continue

        # 只保留最近的点
        if range_map[h_idx, w_idx] == -1 or range_map[h_idx, w_idx] > dist:
            range_map[h_idx, w_idx] = dist
            intensity_map[h_idx, w_idx] = intensity

    range_map_3d = range_map[..., None]
    range_image_r1 = np.concatenate([range_map_3d, intensity_map], axis=-1)  
    range_image_r1[range_map == -1] = 0
    
    return range_image_r1






def pano_to_lidar_with_intensities(pano: np.ndarray, intensities, lidar_K):
    """
    Args:
        pano: (H, W), float32.
        intensities: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points_with_intensities: (N, 4), float32, in lidar frame.
    """
    fov_up, fov = lidar_K

    H, W = pano.shape
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    beta = -(i - W / 2) / W * 2 * np.pi
    alpha = (fov_up - j / H * fov) / 180 * np.pi
    dirs = np.stack(
        [
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ],
        -1,
    )
    local_points = dirs * pano.reshape(H, W, 1)

    # local_points: (H, W, 3)
    # intensities : (H, W)
    # local_points_with_intensities: (H, W, 4)
    local_points_with_intensities = np.concatenate(
        [local_points, intensities.reshape(H, W, 1)], axis=2
    )

    # Filter empty points.
    idx = np.where(pano != 0.0)
    local_points_with_intensities = local_points_with_intensities[idx]

    return local_points_with_intensities


def pano_to_lidar(pano, lidar_K):
    """
    Args:
        pano: (H, W), float32.
        lidar_K: lidar intrinsics (fov_up, fov)

    Return:
        local_points: (N, 3), float32, in lidar frame.
    """
    local_points_with_intensities = pano_to_lidar_with_intensities(
        pano=pano,
        intensities=np.zeros_like(pano),
        lidar_K=lidar_K,
    )
    return local_points_with_intensities[:, :3]
