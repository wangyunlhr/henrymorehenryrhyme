import numpy as np
import math
import torch
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
    # range_image_r1 = np.concatenate([range_map_3d, intensity_map], axis=-1)  #! 修改成3通道
    range_image_r1 = np.concatenate([range_map_3d, intensity_map, range_map[:, :, None]], axis=-1)  
    range_image_r1[range_map == -1] = 0
    
    return range_image_r1


def downsample_range_image_nearest(img4, scale=4):
    """
    img4: (H*scale, W*scale, 3)
      - img4[...,0] 是 range，无效为 0
      - img4[...,1:], 无效为 0
    return: (H, W, 3)
    """
    H4, W4, C = img4.shape
    assert C == 3
    assert H4 % scale == 0 and W4 % scale == 0
    H, W = H4 // scale, W4 // scale

    x = img4.reshape(H, scale, W, scale, 3)          # (H,s,W,s,3)
    r = x[..., 0]                                    # (H,s,W,s)
    valid = r > 0

    r_inf = np.where(valid, r, np.inf)               # 无效设 inf
    r_min = r_inf.min(axis=(1, 3))                   # (H,W)
    has = np.isfinite(r_min)

    # 找到最小值所在的位置（在 s*s 展平后的索引）
    idx = r_inf.reshape(H, W, scale*scale).argmin(axis=2)  # (H,W)

    x_flat = x.reshape(H, W, scale*scale, 3)         # (H,W,s*s,3)
    out = x_flat[np.arange(H)[:, None], np.arange(W)[None, :], idx]  # (H,W,3)

    # 没有有效点的格子置 0
    out[~has] = 0
    # range 通道写回 r_min（数值更稳）
    out[..., 0] = np.where(has, r_min, 0)

    return out



#! scale并且修改了round 为 floor
def kitti_points_to_range_image_floor(
    xyzs,
    intensities,
    H=256,
    W=4096,
    inc_bottom=math.radians(-24.9),
    inc_top=math.radians(2.0),
    max_depth=80.0,
):
    """
    将 LiDAR 点云投影为 range image（z-buffer: 每个像素取最小range）
    角度定义：
      - azimuth: [-pi, pi] -> wrap到 [0, 2pi)
      - inclination: [inc_bottom, inc_top]
    binning：
      - w_idx = floor(az / (2pi/W))
      - h_idx = floor((inc - inc_bottom) / ((inc_top-inc_bottom)/H))
    无效像素：range=0，其它通道也为0

    Args:
        xyzs:        (N, 3) float
        intensities: (N,) or (N, C) float
        H, W:        输出range image分辨率（高分辨率用256×4096）
        inc_bottom/inc_top: 垂直视场
        max_depth:   最大深度

    Returns:
        img: (H, W, 1+C)  第0通道是range，其余为intensity(或特征)
             无效处全0
    """
    xyzs = np.asarray(xyzs)
    intensities = np.asarray(intensities)

    if intensities.ndim == 1:
        intensities = intensities[:, None]  # (N,1)
    C = intensities.shape[1]

    x = xyzs[:, 0]
    y = xyzs[:, 1]
    z = xyzs[:, 2]

    dists = np.linalg.norm(xyzs, axis=1)

    # 角度
    azimuth = np.arctan2(y, x)                              # [-pi, pi]
    inclination = np.arctan2(z, np.sqrt(x * x + y * y))     # [-pi/2, pi/2]

    # 有效点筛选：深度 + 垂直FOV
    valid = (dists > 0) & (dists <= max_depth) & (inclination >= inc_bottom) & (inclination <= inc_top)
    if not np.any(valid):
        return np.zeros((H, W, 1 + C), dtype=np.float32)

    d = dists[valid]
    az = azimuth[valid]
    inc = inclination[valid]
    inten = intensities[valid]  # (Nv, C)

    # wrap到[0, 2pi)
    az = (az + np.pi) % (2.0 * np.pi)                       # [0, 2pi)

    # 正步长
    h_res = 2.0 * np.pi / W                                 # >0
    v_res = (inc_top - inc_bottom) / H                      # >0

    # floor binning（左闭右开）
    w_idx = np.floor(az / h_res).astype(np.int32)
    h_idx = np.floor((inc - inc_bottom) / v_res).astype(np.int32)
    h_idx = (H - 1) - h_idx  # 上下翻转，使得inc_top在上方 #!
    w_idx = (W - 1) - w_idx  #!
    # clip 防止浮点边界导致偶发越界
    w_idx = np.clip(w_idx, 0, W - 1)
    h_idx = np.clip(h_idx, 0, H - 1)

    # z-buffer：每个像素取最小range
    range_map = np.zeros((H, W), dtype=np.float32)          # 0=无效
    feat_map = np.zeros((H, W, C), dtype=np.float32)        # 0=无效

    # 为了正确做min，需要一个inf初始化的缓冲
    best = np.full((H, W), np.inf, dtype=np.float32)

    # 逐点更新（保持与你原代码一致的语义）
    for hi, wi, di, fi in zip(h_idx, w_idx, d, inten):
        if di < best[hi, wi]:
            best[hi, wi] = di
            range_map[hi, wi] = di
            feat_map[hi, wi] = fi

    img = np.concatenate([range_map[..., None], feat_map], axis=-1)  # (H,W,1+C)
    return img



def downsample_range_image_minrange_floor(img_hi, scale=4):
    """
    物理一致降采样（z-buffer pooling）：
    对每个 scale×scale block：
      - 选range(第0通道)最小的像素作为代表
      - 其它通道取该像素的值
    若block全无效(range==0)：输出全0

    Args:
        img_hi: (H*scale, W*scale, C)  第0通道是range，0=无效
        scale:  降采样倍率（256->64 用4）

    Returns:
        img_lo: (H, W, C)
    """
    img_hi = np.asarray(img_hi)
    Hs, Ws, C = img_hi.shape
    assert Hs % scale == 0 and Ws % scale == 0

    H, W = Hs // scale, Ws // scale

    x = img_hi.reshape(H, scale, W, scale, C)      # (H,s,W,s,C)
    r = x[..., 0]                                  # (H,s,W,s)

    valid = r > 0
    r_inf = np.where(valid, r, np.inf)             # 无效设inf

    r_min = r_inf.min(axis=(1, 3))                 # (H,W)
    has = np.isfinite(r_min)                       # 该block是否有有效点

    # 找最小range所在位置（展平 s*s）
    idx = r_inf.reshape(H, W, scale * scale).argmin(axis=2)  # (H,W)

    x_flat = x.reshape(H, W, scale * scale, C)     # (H,W,s*s,C)
    out = x_flat[np.arange(H)[:, None], np.arange(W)[None, :], idx]  # (H,W,C)

    # 全无效block输出全0
    out[~has] = 0
    # range通道用更稳的 r_min
    out[..., 0] = np.where(has, r_min, 0)

    return out



def range2point_consistent2(
    range_map,                  # (H,W) or (H,W,1) or (1,H,W)
    inc_bottom=math.radians(-24.9),
    inc_top=math.radians(2.0),
    angle_offset=0.0,           # forward 如果用了 az' = az + offset，这里就用 -offset；反之亦然
    offset_sign=-1.0,           # 关键：默认假设 forward 做了 az_used = az + angle_offset，则 inverse 做 az = az_used - angle_offset
    sensor2world=None,          # (4,4) torch/np, optional
    device="cuda",
    use_bin_center=True,        # 强烈建议 True：用 (idx + 0.5) 更接近“落入该 bin 的点”的平均方向
):
    """
    与 kitti_points_to_range_image_floor 对齐的 range->points（几何一致）。
    forward:
      az_wrapped = (atan2(y,x)+pi)%(2pi) in [0,2pi)
      w_idx_raw = floor(az_wrapped / h_res)
      h_idx_raw = floor((inc-inc_bottom)/v_res)
      h = (H-1) - h_idx_raw
      w = (W-1) - w_idx_raw
    inverse:
      h_idx_raw = (H-1) - h
      w_idx_raw = (W-1) - w
      inc = inc_bottom + (h_idx_raw + 0.5)*v_res
      az_wrapped = (w_idx_raw + 0.5)*h_res
      az = az_wrapped - pi   (-> [-pi,pi))
      再处理 angle_offset（取决于 forward 定义）
    """
    # --- to (H,W) float tensor ---
    if not torch.is_tensor(range_map):
        range_map = torch.tensor(range_map, dtype=torch.float32, device=device)
    else:
        range_map = range_map.to(device=device, dtype=torch.float32)

    if range_map.dim() == 3:
        if range_map.shape[0] == 1:         # (1,H,W)
            range_map = range_map[0]
        elif range_map.shape[2] == 1:       # (H,W,1)
            range_map = range_map[..., 0]
        else:
            raise ValueError("range_map must be (H,W), (1,H,W) or (H,W,1)")
    elif range_map.dim() != 2:
        raise ValueError("range_map dim must be 2 or 3")

    H, W = range_map.shape
    h_res = 2.0 * math.pi / W
    v_res = (inc_top - inc_bottom) / H

    # --- pixel indices (integer grid) ---
    # 注意：这里用“像素索引”h,w（0..H-1 / 0..W-1），再手动加 0.5 得到 bin center
    h = torch.arange(H, device=device, dtype=torch.float32)
    w = torch.arange(W, device=device, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(h, w, indexing="ij")  # (H,W)

    # invert flips
    h_idx_raw = (H - 1) - grid_h
    w_idx_raw = (W - 1) - grid_w

    # choose bin center
    if use_bin_center:
        h_pos = h_idx_raw + 0.5
        w_pos = w_idx_raw + 0.5
    else:
        h_pos = h_idx_raw
        w_pos = w_idx_raw

    # recover angles
    inc = inc_bottom + h_pos * v_res                          # (H,W)
    az_wrapped = w_pos * h_res                                # (H,W) in [0,2pi) approximately
    az = az_wrapped - math.pi                                 # -> [-pi, pi)

    # handle offset (make it explicit & controllable)
    if angle_offset != 0.0:
        az = az + offset_sign * angle_offset

    # rays in sensor frame
    cos_inc = torch.cos(inc)
    rays = torch.stack([
        cos_inc * torch.cos(az),
        cos_inc * torch.sin(az),
        torch.sin(inc),
    ], dim=-1)                                                 # (H,W,3)

    # (可选) normalize：理论上已是单位向量，归一化可抑制数值误差
    rays = rays / (torch.norm(rays, dim=-1, keepdim=True) + 1e-12)

    points = rays * range_map[..., None]                       # (H,W,3)

    # optional: to world
    if sensor2world is not None:
        if not torch.is_tensor(sensor2world):
            sensor2world = torch.tensor(sensor2world, dtype=torch.float32, device=device)
        else:
            sensor2world = sensor2world.to(device=device, dtype=torch.float32)
        R = sensor2world[:3, :3]
        t = sensor2world[:3, 3]
        points = points @ R.T + t

    return points



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
