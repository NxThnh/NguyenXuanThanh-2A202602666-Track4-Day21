"""Pipeline phát hiện vật cản 3D cho Robot/Drone không dùng Deep Learning.

Pipeline chuẩn:
1. ROI Cropping (lọc không gian quan sát, loại bỏ điểm xa/bụi/ngoài vùng di chuyển)
2. Voxel Downsample (giảm mật độ điểm, tăng tốc độ xử lý và chuẩn hóa mật độ)
3. RANSAC Ground Removal (tách mặt phẳng mặt đất với ràng buộc vector pháp tuyến đứng)
4. DBSCAN Clustering (gom cụm vật thể không dựa vào hình dạng tiên nghiệm)
5. 3D Bounding Box Extraction & Distance Calculation (tính khoảng cách tới vật cản gần nhất)
6. 2D/BEV Occupancy Grid & Safety Inflation Costmap (phục vụ Robot Navigation)
7. Ground Truth Evaluation (so khớp với nhãn 3D KITTI/nuScenes đo Precision, Recall, F1)
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import box3d_corners_cam


@dataclass
class ClusterInfo:
    cluster_id: int
    points: np.ndarray  # (P, 3)
    num_points: int
    center: np.ndarray  # (3,)
    min_bound: np.ndarray  # (3,)
    max_bound: np.ndarray  # (3,)
    extent: np.ndarray  # (3,) [dx, dy, dz]
    min_distance: float  # Khoảng cách 3D nhỏ nhất tới cảm biến
    bev_distance: float  # Khoảng cách 2D nhỏ nhất trên mặt phẳng (x, y)
    obb_center: Optional[np.ndarray] = None
    obb_extent: Optional[np.ndarray] = None
    obb_rotation: Optional[np.ndarray] = None


@dataclass
class PipelineResult:
    raw_points_count: int
    roi_points_count: int
    downsampled_points_count: int
    ground_points_count: int
    obstacle_points_count: int
    num_clusters: int
    nearest_obstacle_dist: float
    nearest_cluster_id: int
    clusters: List[ClusterInfo] = field(default_factory=list)
    ground_plane: Optional[np.ndarray] = None  # [a, b, c, d]
    downsampled_points: Optional[np.ndarray] = None
    ground_points: Optional[np.ndarray] = None
    obstacle_points: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    timings_ms: Dict[str, float] = field(default_factory=dict)


def clean_points(points: np.ndarray) -> np.ndarray:
    """Lọc bỏ điểm NaN, Inf và điểm lỗi."""
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts3 = points[:, :3]
    valid = np.isfinite(pts3).all(axis=1)
    return pts3[valid]


def filter_roi(
    points: np.ndarray,
    x_range: Tuple[float, float] = (0.0, 50.0),
    y_range: Tuple[float, float] = (-15.0, 15.0),
    z_range: Tuple[float, float] = (-2.5, 1.5),
) -> np.ndarray:
    """Lọc vùng quan tâm (ROI) của robot/xe trong không gian sensor."""
    pts = clean_points(points)
    if len(pts) == 0:
        return pts

    mask = (
        (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
        (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) &
        (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])
    )
    return pts[mask]


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
    """Downsample point cloud bằng Open3D Voxel Grid."""
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if voxel_size <= 0:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down.points, dtype=np.float32)


def segment_ground_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.15,
    max_iterations: int = 1000,
    normal_z_threshold: float = 0.80,
    z_ground_hint: float = -1.6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tách mặt đất bằng RANSAC plane segmentation có ràng buộc pháp tuyến thẳng đứng.

    Args:
        points: (N, 3) point cloud đã lọc ROI và downsample
        distance_threshold: Ngưỡng khoảng cách RANSAC (m)
        max_iterations: Số vòng lặp RANSAC tối đa
        normal_z_threshold: Ràng buộc cos góc pháp tuyến với trục Z (|n_z| >= ngưỡng)
        z_ground_hint: Gợi ý chiều cao mặt đất ước lượng để lọc ứng viên

    Returns:
        obstacle_points: (M, 3) điểm vật cản (non-ground)
        ground_points: (G, 3) điểm mặt đất
        plane_model: (4,) tham số [a, b, c, d] của mp ax + by + cz + d = 0
        inlier_indices: mảng chỉ số điểm thuộc mặt đất
    """
    if len(points) < 10:
        empty = np.empty((0, 3), dtype=np.float32)
        return points, empty, np.zeros(4, dtype=np.float32), np.array([], dtype=int)

    # Ưu tiên tập điểm nằm ở nửa dưới để fit mặt phẳng đất nhanh và chính xác hơn
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # Chạy RANSAC mặt phẳng
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=max_iterations,
    )
    plane_model = np.array(plane_model, dtype=np.float32)

    # Kiểm tra vector pháp tuyến: [a, b, c] phải hướng gần thẳng đứng (trục Z up)
    norm = np.linalg.norm(plane_model[:3])
    if norm > 1e-6:
        n_z = abs(plane_model[2]) / norm
    else:
        n_z = 0.0

    # Nếu pháp tuyến không đứng (ví dụ RANSAC bị hút vào tường dọc hoặc thân xe)
    if n_z < normal_z_threshold:
        # Lọc thô điểm theo z thấp trước rồi fit RANSAC trên tập con thấp
        low_mask = points[:, 2] < (z_ground_hint + 0.8)
        if low_mask.sum() > 20:
            low_pcd = o3d.geometry.PointCloud()
            low_pcd.points = o3d.utility.Vector3dVector(points[low_mask].astype(np.float64))
            sub_plane, sub_inliers = low_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=max_iterations,
            )
            sub_norm = np.linalg.norm(sub_plane[:3])
            if sub_norm > 1e-6 and (abs(sub_plane[2]) / sub_norm) >= normal_z_threshold:
                plane_model = np.array(sub_plane, dtype=np.float32)
                # Tính lại inliers trên toàn bộ điểm
                pts_dist = np.abs(points @ plane_model[:3] + plane_model[3]) / sub_norm
                inliers = np.where(pts_dist <= distance_threshold)[0]

    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[inliers] = True

    ground_points = points[inlier_mask]
    obstacle_points = points[~inlier_mask]
    return obstacle_points, ground_points, plane_model, np.array(inliers, dtype=int)


def cluster_dbscan(
    obstacle_points: np.ndarray,
    eps: float = 0.6,
    min_points: int = 10,
    max_cluster_points: int = 50000,
) -> Tuple[List[ClusterInfo], np.ndarray]:
    """Phân cụm vật cản bằng thuật toán DBSCAN của Open3D."""
    if len(obstacle_points) < min_points:
        return [], np.full(len(obstacle_points), -1, dtype=int)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obstacle_points.astype(np.float64))

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False),
        dtype=int,
    )

    clusters: List[ClusterInfo] = []
    unique_labels = np.unique(labels)

    for cid in unique_labels:
        if cid == -1:
            continue
        c_pts = obstacle_points[labels == cid]
        if len(c_pts) > max_cluster_points:
            continue

        c_min = c_pts.min(axis=0)
        c_max = c_pts.max(axis=0)
        extent = c_max - c_min
        center = c_pts.mean(axis=0)

        # Khoảng cách 3D và BEV nhỏ nhất tới tâm sensor (0,0,0)
        dists_3d = np.linalg.norm(c_pts, axis=1)
        dists_bev = np.linalg.norm(c_pts[:, :2], axis=1)
        min_dist_3d = float(dists_3d.min())
        min_dist_bev = float(dists_bev.min())

        obb_center, obb_extent, obb_rot = None, None, None
        if len(c_pts) >= 4:
            try:
                sub_pcd = o3d.geometry.PointCloud()
                sub_pcd.points = o3d.utility.Vector3dVector(c_pts.astype(np.float64))
                obb = sub_pcd.get_oriented_bounding_box()
                obb_center = np.asarray(obb.center, dtype=np.float32)
                obb_extent = np.asarray(obb.extent, dtype=np.float32)
                obb_rot = np.asarray(obb.R, dtype=np.float32)
            except Exception:
                pass

        clusters.append(
            ClusterInfo(
                cluster_id=int(cid),
                points=c_pts,
                num_points=len(c_pts),
                center=center,
                min_bound=c_min,
                max_bound=c_max,
                extent=extent,
                min_distance=min_dist_3d,
                bev_distance=min_dist_bev,
                obb_center=obb_center,
                obb_extent=obb_extent,
                obb_rotation=obb_rot,
            )
        )

    # Sắp xếp clusters theo khoảng cách gần nhất tới ego
    clusters.sort(key=lambda c: c.min_distance)
    return clusters, labels


def run_obstacle_pipeline(
    points: np.ndarray,
    voxel_size: float = 0.1,
    ground_distance_thresh: float = 0.15,
    dbscan_eps: float = 0.6,
    dbscan_min_points: int = 10,
    x_range: Tuple[float, float] = (0.0, 50.0),
    y_range: Tuple[float, float] = (-15.0, 15.0),
    z_range: Tuple[float, float] = (-2.5, 1.5),
) -> PipelineResult:
    """Chạy toàn bộ pipeline phát hiện vật cản và ghi nhận latency từng bước."""
    timings = {}
    t0 = time.perf_counter()

    # Bước 1: ROI Cropping
    t_start = time.perf_counter()
    roi_pts = filter_roi(points, x_range=x_range, y_range=y_range, z_range=z_range)
    timings["roi_crop"] = (time.perf_counter() - t_start) * 1000.0

    # Bước 2: Voxel Downsample
    t_start = time.perf_counter()
    down_pts = voxel_downsample(roi_pts, voxel_size=voxel_size)
    timings["voxel_downsample"] = (time.perf_counter() - t_start) * 1000.0

    # Bước 3: RANSAC Ground Removal
    t_start = time.perf_counter()
    obs_pts, ground_pts, plane_model, _ = segment_ground_ransac(
        down_pts, distance_threshold=ground_distance_thresh
    )
    timings["ransac_ground"] = (time.perf_counter() - t_start) * 1000.0

    # Bước 4: DBSCAN Clustering
    t_start = time.perf_counter()
    clusters, cluster_labels = cluster_dbscan(
        obs_pts, eps=dbscan_eps, min_points=dbscan_min_points
    )
    timings["dbscan_clustering"] = (time.perf_counter() - t_start) * 1000.0
    timings["total_pipeline"] = (time.perf_counter() - t0) * 1000.0

    nearest_dist = float(clusters[0].min_distance) if clusters else float("inf")
    nearest_id = clusters[0].cluster_id if clusters else -1

    return PipelineResult(
        raw_points_count=len(points),
        roi_points_count=len(roi_pts),
        downsampled_points_count=len(down_pts),
        ground_points_count=len(ground_pts),
        obstacle_points_count=len(obs_pts),
        num_clusters=len(clusters),
        nearest_obstacle_dist=nearest_dist,
        nearest_cluster_id=nearest_id,
        clusters=clusters,
        ground_plane=plane_model,
        downsampled_points=down_pts,
        ground_points=ground_pts,
        obstacle_points=obs_pts,
        cluster_labels=cluster_labels,
        timings_ms=timings,
    )


def generate_occupancy_grid_bev(
    obstacle_points: np.ndarray,
    x_range: Tuple[float, float] = (0.0, 50.0),
    y_range: Tuple[float, float] = (-15.0, 15.0),
    resolution: float = 0.1,
    inflation_radius_m: float = 0.3,
) -> Dict[str, Any]:
    """Tạo bản đồ Occupancy Grid 2D/BEV kèm bản đồ chi phí an toàn (Safety Costmap).

    Args:
        obstacle_points: (M, 3) điểm vật thể trong LiDAR frame
        x_range, y_range: phạm vi tọa độ BEV (mét)
        resolution: kích thước mỗi cell (mét/pixel)
        inflation_radius_m: bán kính mở rộng vùng nguy hiểm xung quanh vật cản (m)

    Returns:
        dict chứa: binary_grid, inflated_costmap, x_bins, y_bins, shape, resolution
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    nx = int(np.ceil((x_max - x_min) / resolution))
    ny = int(np.ceil((y_max - y_min) / resolution))

    binary_grid = np.zeros((nx, ny), dtype=np.uint8)

    if len(obstacle_points) > 0:
        pts = obstacle_points
        valid_x = (pts[:, 0] >= x_min) & (pts[:, 0] < x_max)
        valid_y = (pts[:, 1] >= y_min) & (pts[:, 1] < y_max)
        valid = valid_x & valid_y
        pts_valid = pts[valid]

        if len(pts_valid) > 0:
            ix = np.floor((pts_valid[:, 0] - x_min) / resolution).astype(int)
            iy = np.floor((pts_valid[:, 1] - y_min) / resolution).astype(int)
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)
            binary_grid[ix, iy] = 255

    # Tính toán Safety Inflation Costmap (dilated / Gaussian blurred distance field)
    inflation_kernel_size = int(np.ceil(inflation_radius_m / resolution)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflation_kernel_size, inflation_kernel_size))
    inflated_mask = cv2.dilate(binary_grid, kernel)

    # Khoảng cách Euclidean tới vật cản gần nhất (Distance Transform)
    inv_binary = cv2.bitwise_not(binary_grid)
    dist_map = cv2.distanceTransform(inv_binary, cv2.DIST_L2, 5) * resolution

    # Costmap: 100 tại vật cản, giảm dần ra ngoài tới 0 tại inflation_radius_m
    costmap = np.zeros_like(dist_map, dtype=np.float32)
    costmap[binary_grid > 0] = 100.0
    safe_zone = (dist_map > 0) & (dist_map <= inflation_radius_m)
    costmap[safe_zone] = 100.0 * (1.0 - (dist_map[safe_zone] / inflation_radius_m))

    return {
        "binary_grid": binary_grid,
        "inflated_mask": inflated_mask,
        "costmap": costmap,
        "dist_map": dist_map,
        "nx": nx,
        "ny": ny,
        "resolution": resolution,
        "x_range": x_range,
        "y_range": y_range,
    }


def gt_boxes_to_velo(labels: List[KittiObject], calib: KittiCalib) -> List[Dict[str, Any]]:
    """Chuyển đổi nhãn Ground Truth 3D từ rectified camera frame sang velodyne frame."""
    gt_boxes_velo = []
    inv_T = np.linalg.inv(calib.T_cam_velo)

    for obj in labels:
        if obj.type in ["DontCare"]:
            continue
        corners_cam = box3d_corners_cam(obj)  # (8, 3)
        corners_hom = np.hstack([corners_cam, np.ones((8, 1))])
        corners_velo = (corners_hom @ inv_T.T)[:, :3]

        c_min = corners_velo.min(axis=0)
        c_max = corners_velo.max(axis=0)
        center = corners_velo.mean(axis=0)
        extent = c_max - c_min
        dist = float(np.linalg.norm(center))

        gt_boxes_velo.append({
            "type": obj.type,
            "corners_velo": corners_velo,
            "center_velo": center,
            "min_bound": c_min,
            "max_bound": c_max,
            "extent": extent,
            "dist": dist,
            "kitti_obj": obj,
        })
    return gt_boxes_velo


def compute_bev_iou(
    box_a_min: np.ndarray,
    box_a_max: np.ndarray,
    box_b_min: np.ndarray,
    box_b_max: np.ndarray,
) -> float:
    """Tính 2D BEV Axis-Aligned IoU giữa 2 bounding boxes."""
    inter_x_min = max(box_a_min[0], box_b_min[0])
    inter_x_max = min(box_a_max[0], box_b_max[0])
    inter_y_min = max(box_a_min[1], box_b_min[1])
    inter_y_max = min(box_a_max[1], box_b_max[1])

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area_a = (box_a_max[0] - box_a_min[0]) * (box_a_max[1] - box_a_min[1])
    area_b = (box_b_max[0] - box_b_min[0]) * (box_b_max[1] - box_b_min[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def match_clusters_with_gt(
    clusters: List[ClusterInfo],
    gt_boxes_velo: List[Dict[str, Any]],
    iou_thresh: float = 0.20,
    center_dist_thresh: float = 2.0,
) -> Dict[str, Any]:
    """Đánh giá chất lượng phân cụm so với nhãn Ground Truth (Precision, Recall, F1-score)."""
    if len(gt_boxes_velo) == 0:
        return {
            "num_gt": 0,
            "num_clusters": len(clusters),
            "tp": 0,
            "fp": len(clusters),
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matches": [],
        }

    # Ma trận chi phí dựa trên BEV IoU và khoảng cách tâm
    matched_gt = set()
    matched_clusters = set()
    matches = []

    # Ưu tiên khớp theo IoU cao nhất
    candidates = []
    for c_idx, cluster in enumerate(clusters):
        for g_idx, gt in enumerate(gt_boxes_velo):
            iou = compute_bev_iou(cluster.min_bound, cluster.max_bound, gt["min_bound"], gt["max_bound"])
            dist = float(np.linalg.norm(cluster.center - gt["center_velo"]))
            if iou >= iou_thresh or dist <= center_dist_thresh:
                # Điểm số kết hợp (IoU càng cao, dist càng nhỏ)
                score = iou + max(0.0, 1.0 - dist / center_dist_thresh)
                candidates.append((score, iou, dist, c_idx, g_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    for score, iou, dist, c_idx, g_idx in candidates:
        if c_idx not in matched_clusters and g_idx not in matched_gt:
            matched_clusters.add(c_idx)
            matched_gt.add(g_idx)
            matches.append({
                "cluster_id": clusters[c_idx].cluster_id,
                "gt_type": gt_boxes_velo[g_idx]["type"],
                "iou": iou,
                "center_dist": dist,
                "gt_dist": gt_boxes_velo[g_idx]["dist"],
            })

    tp = len(matched_clusters)
    fp = len(clusters) - tp
    fn = len(gt_boxes_velo) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "num_gt": len(gt_boxes_velo),
        "num_clusters": len(clusters),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }
