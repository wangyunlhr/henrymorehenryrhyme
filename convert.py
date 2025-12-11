import numpy as np
import numpy as np
import matplotlib.pyplot as plt


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
    local_points: np.ndarray, lidar_H: int, lidar_W: int, lidar_K: int, max_depth=80
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
        max_depth=max_depth,
    )
    return pano








#! 根据beam将pointcloud转为pano
def lidar_to_pano_with_intensities_withbeams(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    beam_inclinations: np.ndarray, 
    angle_offset: np.float32, 
    pixel_offset: np.float32,
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
    local_point_intensities = local_points_with_intensities[:, 3:] #intensities, normals, labels


    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.
    pano = np.zeros((lidar_H, lidar_W))
    intensities = np.zeros((lidar_H, lidar_W, 5))
    frac_hist = np.zeros(10, dtype=np.int64)  # 统计行小数部分的直方图
    # 准备一个 beam index 数组，用于插值
    beam_idx = np.arange(lidar_H, dtype=np.float32)
    for local_points, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = local_points
        # beta = np.pi - np.arctan2(y, x)
        # alpha = np.arctan2(z, np.sqrt(x**2 + y**2)) + fov_down / 180 * np.pi
        # c = int(round(beta / (2 * np.pi / lidar_W)))
        # r = int(round(lidar_H - alpha / (fov / 180 * np.pi / lidar_H)))
        phi = np.arctan2(y, x)
        # 映回与 range2point 一致的连续横坐标
        grid_x = ((phi + np.pi + angle_offset.numpy()) / (2.0 * np.pi)) % 1.0  # [0,1)
        c_float = lidar_W - (lidar_W * grid_x + pixel_offset)
        c = np.rint(c_float).astype(np.int64) % lidar_W

        #! 行：替换为“最近束角”
        incl = np.arctan2(z, np.sqrt(x**2 + y**2))

        ### ---- 1. 计算连续行坐标 r_float，并统计小数部分 ----

        # if frac_hist is not None:
        #     # 连续 index（float），在 [0, lidar_H-1] 之间
        #     r_float = np.interp(incl, beam_inclinations, beam_idx)

        #     # 取小数部分 frac ∈ [0,1)
        #     frac = r_float - np.floor(r_float)

        #     # 映射到 10 个 bin：0.0~0.1, 0.1~0.2, ..., 0.9~1.0
        #     bin_idx = int(frac * 10)
        #     if bin_idx == 10:
        #         bin_idx = 9  # 防止 frac 恰好是 1.0 时越界

        #     frac_hist[bin_idx] += 1
        ### ---- 统计到此结束 ----


        r = int(np.argmin(np.abs(beam_inclinations - incl)))
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
    # show_frac_hist(frac_hist)
    return pano, intensities

def show_frac_hist(frac_hist):

    bins = np.arange(10)  # 0..9
    centers = (bins + 0.5) / 10.0  # 0.05, 0.15, ..., 0.95 作为横轴刻度位置

    plt.figure()
    plt.bar(centers, frac_hist, width=0.09)
    plt.xlabel("fractional part of r_float (0~1)")
    plt.ylabel("count")
    plt.title("Distribution of fractional part of r_float")
    plt.xticks(np.linspace(0.05, 0.95, 10), [f"{i/10:.1f}" for i in range(1, 11)])
    plt.grid(True, axis='y')
    plt.tight_layout()
    # plt.show()
    plt.savefig("r_float_fraction_hist.png", dpi=300, bbox_inches="tight")



def lidar_to_pano_with_beams(
    local_points: np.ndarray, local_intensities: np.ndarray, lidar_H: int, lidar_W: int, beam_inclinations: list, angle_offset: np.float32, pixel_offset: np.float32, max_depth=80, 
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
        [local_points, local_intensities], axis=1
    )
    beam_inclinations = np.array(beam_inclinations).astype(np.float64).copy()
    beam_inclinations = beam_inclinations[::-1]
    pano, pano_intensity = lidar_to_pano_with_intensities_withbeams(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        beam_inclinations = beam_inclinations,
        angle_offset = angle_offset,
        pixel_offset = pixel_offset,
        max_depth=max_depth,
    )
    return pano, pano_intensity


def expand_beam_inclinations(beam_inclinations: np.ndarray, factor: int) -> np.ndarray:
    """
    将束角数组按给定 factor 扩展：
    - 原有 H 束 -> H * factor 束
    - 偶/整倍( i*factor )位置为原束
    - 中间插值为相邻两束的线性插值
    - 最后一束后面的插值全部复制最后一束
    """
    assert factor >= 1 and isinstance(factor, int)

    H = beam_inclinations.shape[0]
    if factor == 1:
        return beam_inclinations.copy()

    beam_ktimes = np.zeros(H * factor, dtype=beam_inclinations.dtype)

    for i in range(H):
        base_idx = i * factor
        # 原束角放在 base_idx 位置
        beam_ktimes[base_idx] = beam_inclinations[i]

        if i < H - 1:
            # 与下一束之间插 factor-1 个点
            for m in range(1, factor):
                alpha = m / factor  # 0~1 之间
                beam_ktimes[base_idx + m] = (
                    (1.0 - alpha) * beam_inclinations[i]
                    + alpha * beam_inclinations[i + 1]
                )
        else:
            # 最后一束：后面 factor-1 个位置用自己填充
            for m in range(1, factor):
                beam_ktimes[base_idx + m] = beam_inclinations[i]

    return beam_ktimes


#! 扩展beams
def lidar_to_pano_with_intensities_withbeams_scale(
    local_points_with_intensities: np.ndarray,
    lidar_H: int,
    lidar_W: int,
    beam_inclinations: np.ndarray, 
    angle_offset: np.float32, 
    pixel_offset: np.float32,
    scale_h: int = 1,
    scale_w: int = 1,
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
    #! 
    assert isinstance(scale_h, int) and scale_h >= 1
    assert isinstance(scale_w, int) and scale_w >= 1

    H = lidar_H
    W = lidar_W

    # === 垂直方向: 扩展束角 ===
    beam_scaled = expand_beam_inclinations(beam_inclinations, scale_h)
    H_scaled = beam_scaled.shape[0] 
    # === 水平方向: 扩展分辨率 ===
    W_scaled = W * scale_w
    pixel_offset_scaled = pixel_offset * scale_w  # 像素偏移按分辨率放大

    # Un pack.
    local_points = local_points_with_intensities[:, :3]
    local_point_intensities = local_points_with_intensities[:, 3:] #intensities, normals, labels


    # Compute dists to lidar center.
    dists = np.linalg.norm(local_points, axis=1)

    # Fill pano and intensities.#! 初始化 (H_scaled, W_scaled) 图
    pano = np.zeros((H_scaled, W_scaled), dtype=np.float32)
    intensities = np.zeros((H_scaled, W_scaled, 5), dtype=np.float32)

    for pt, dist, local_point_intensity in zip(
        local_points,
        dists,
        local_point_intensities,
    ):
        # Check max depth.
        if dist >= max_depth:
            continue

        x, y, z = pt

        phi = np.arctan2(y, x)
        # 映回与 range2point 一致的连续横坐标
        grid_x = ((phi + np.pi + angle_offset.numpy()) / (2.0 * np.pi)) % 1.0  # [0,1)
        c_float = W_scaled - (W_scaled * grid_x + pixel_offset_scaled)
        c = np.rint(c_float).astype(np.int64) % W_scaled

        #! 行：替换为“最近束角”
        incl = np.arctan2(z, np.sqrt(x**2 + y**2))
        r = int(np.argmin(np.abs(beam_scaled - incl)))
        # Check out-of-bounds.
        if r >= H_scaled or r < 0 or c >= W_scaled or c < 0:
            continue

        # Set to min dist if not set.
        if pano[r, c] == 0.0:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity
        elif pano[r, c] > dist:
            pano[r, c] = dist
            intensities[r, c] = local_point_intensity

    return pano, intensities




def lidar_to_pano_with_beams_expand_scale(
    local_points: np.ndarray, local_intensities: np.ndarray, lidar_H: int, lidar_W: int, beam_inclinations: list, angle_offset: np.float32, pixel_offset: np.float32, scale_h: int, scale_w: int, max_depth=80, 
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
        [local_points, local_intensities], axis=1
    )
    beam_inclinations = np.array(beam_inclinations).astype(np.float64).copy()
    beam_inclinations = beam_inclinations[::-1]

    pano, pano_intensity = lidar_to_pano_with_intensities_withbeams_scale(
        local_points_with_intensities=local_points_with_intensities,
        lidar_H=lidar_H,
        lidar_W=lidar_W,
        beam_inclinations = beam_inclinations,
        angle_offset = angle_offset,
        pixel_offset = pixel_offset,
        scale_h = scale_h,
        scale_w = scale_w,
        max_depth=max_depth,
    )
    return pano, pano_intensity




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

def downsample_rangeimage_2x2_min(
    pano_2: np.ndarray,      # (2H, 2W)
    intens_2: np.ndarray,    # (2H, 2W, C)
):
    H2, W2 = pano_2.shape
    assert H2 % 2 == 0 and W2 % 2 == 0
    H = H2 // 2
    W = W2 // 2
    C = intens_2.shape[2]

    pano = np.zeros((H, W), dtype=pano_2.dtype)
    intens = np.zeros((H, W, C), dtype=intens_2.dtype)

    for r in range(H):
        for c in range(W):
            # 当前 coarse 像素对应的 2x2 子块
            block = pano_2[2*r:2*r+2, 2*c:2*c+2]  # (2,2)

            # 如果整个 block 没有点（全 0），保持为 0
            if np.all(block == 0):
                continue

            # 只在非零里找最小距离
            # 把 0 替换成 +inf，这样不会被选中
            block_masked = np.where(block > 0, block, np.inf)
            br, bc = np.unravel_index(
                np.argmin(block_masked), block_masked.shape
            )

            pano[r, c] = block[br, bc]
            intens[r, c] = intens_2[2*r + br, 2*c + bc]

    return pano, intens


