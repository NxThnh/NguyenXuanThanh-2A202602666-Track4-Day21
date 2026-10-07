"""Các hàm trực quan hóa 2D BEV và 3D cho Obstacle Detection Pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Thiết lập style đồ họa đẹp, hiện đại, rõ ràng
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_pipeline_stages(
    points_raw: np.ndarray,
    points_down: np.ndarray,
    ground_points: np.ndarray,
    obstacle_points: np.ndarray,
    clusters: list,
    nearest_dist: float,
    output_path: str | Path,
    title_suffix: str = "",
) -> None:
    """Vẽ 4 bước của pipeline phát hiện vật cản (BEV perspective)."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor="#ffffff")
    fig.suptitle(f"3D LiDAR Obstacle Detection Pipeline {title_suffix}", fontsize=16, fontweight="bold", y=0.98)

    # Khung nhìn BEV chuẩn (X: 0->45m trước, Y: -15->15m hai bên)
    x_lim = (-2, 45)
    y_lim = (-15, 15)

    def style_ax(ax, title):
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel("X Forward (m)", fontsize=11, fontweight="medium")
        ax.set_ylabel("Y Left/Right (m)", fontsize=11, fontweight="medium")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_aspect("equal")
        # Vẽ các vòng tròn cự ly (Range rings: 10m, 20m, 30m, 40m)
        for r in [10, 20, 30, 40]:
            circle = plt.Circle((0, 0), r, color="#aaaaaa", fill=False, linestyle=":", alpha=0.6, linewidth=1.0)
            ax.add_patch(circle)
        # Ego sensor marker
        ax.plot(0, 0, marker="^", color="crimson", markersize=9, label="Ego Sensor")

    # Bước 1: Raw Point Cloud
    style_ax(axes[0], f"1. Raw LiDAR Cloud\n({len(points_raw):,} points)")
    if len(points_raw) > 0:
        # Sample nhẹ nếu raw quá dày để vẽ nhanh
        sample = points_raw[np.random.choice(len(points_raw), min(len(points_raw), 25000), replace=False)] if len(points_raw) > 25000 else points_raw
        scatter0 = axes[0].scatter(sample[:, 0], sample[:, 1], c=sample[:, 2], cmap="viridis", s=1.0, alpha=0.5)
        plt.colorbar(scatter0, ax=axes[0], orientation="horizontal", pad=0.15, label="Z Height (m)", shrink=0.7)

    # Bước 2: Voxel Downsample
    style_ax(axes[1], f"2. Voxel Downsample (0.1m)\n({len(points_down):,} points)")
    if len(points_down) > 0:
        axes[1].scatter(points_down[:, 0], points_down[:, 1], c=points_down[:, 2], cmap="plasma", s=2.5, alpha=0.65)

    # Bước 3: RANSAC Ground Removal
    style_ax(axes[2], f"3. RANSAC Ground Removal\n(Ground: {len(ground_points):,} | Obs: {len(obstacle_points):,})")
    if len(ground_points) > 0:
        axes[2].scatter(ground_points[:, 0], ground_points[:, 1], c="#b0b0b0", s=1.2, alpha=0.3, label="Ground")
    if len(obstacle_points) > 0:
        axes[2].scatter(obstacle_points[:, 0], obstacle_points[:, 1], c="#d9381e", s=3.5, alpha=0.8, label="Obstacle")
    axes[2].legend(loc="upper right", frameon=True, fontsize=9)

    # Bước 4: DBSCAN Clustering & Bounding Boxes
    style_ax(axes[3], f"4. DBSCAN Clustering & 3D Boxes\n({len(clusters)} clusters | Nearest: {nearest_dist:.2f}m)")
    cmap = plt.get_cmap("tab20")
    for idx, c in enumerate(clusters):
        color = cmap(idx % 20)
        axes[3].scatter(c.points[:, 0], c.points[:, 1], color=color, s=4.0, alpha=0.85)

        # Vẽ 2D Bounding Box (BEV)
        dx = c.extent[0]
        dy = c.extent[1]
        rect = patches.Rectangle(
            (c.min_bound[0], c.min_bound[1]), dx, dy,
            linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.18
        )
        axes[3].add_patch(rect)

        # Highlight vật cản gần nhất
        if idx == 0 and nearest_dist < float("inf"):
            axes[3].plot([0, c.center[0]], [0, c.center[1]], color="red", linestyle="--", linewidth=1.8, label="Nearest Vector")
            axes[3].annotate(
                f"ID:{c.cluster_id}\n{c.min_distance:.1f}m",
                (c.center[0], c.center[1]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, fontweight="bold", color="darkred",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8)
            )

    axes[3].legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bev_occupancy_grid(
    grid_dict: Dict[str, Any],
    clusters: list,
    output_path: str | Path,
    title: str = "2D Bird's-Eye View (BEV) Occupancy Grid & Costmap",
) -> None:
    """Vẽ bản đồ Occupancy Grid và Costmap an toàn dành cho robot/drone navigation."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="#ffffff")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    x_range = grid_dict["x_range"]
    y_range = grid_dict["y_range"]
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

    # 1. Binary Occupancy Grid
    ax1 = axes[0]
    ax1.imshow(grid_dict["binary_grid"].T, origin="lower", extent=extent, cmap="Greys", alpha=0.9)
    ax1.set_title(f"Binary Occupancy Grid ({grid_dict['resolution']}m/cell)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X Forward (m)")
    ax1.set_ylabel("Y Left/Right (m)")
    ax1.plot(0, 0, marker="^", color="crimson", markersize=10, label="Robot Base")

    # Range rings
    for r in [10, 20, 30, 40]:
        c = plt.Circle((0, 0), r, color="#777777", fill=False, linestyle=":", alpha=0.7)
        ax1.add_patch(c)

    # 2. Safety Inflation Costmap
    ax2 = axes[1]
    im2 = ax2.imshow(grid_dict["costmap"].T, origin="lower", extent=extent, cmap="turbo", alpha=0.85)
    ax2.set_title("Safety Inflation Costmap (Danger Zone 0-100)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X Forward (m)")
    ax2.set_ylabel("Y Left/Right (m)")
    ax2.plot(0, 0, marker="^", color="white", markersize=10, label="Robot Base")

    for r in [10, 20, 30, 40]:
        c = plt.Circle((0, 0), r, color="#ffffff", fill=False, linestyle=":", alpha=0.5)
        ax2.add_patch(c)

    # Vẽ cluster boxes lên costmap
    for c in clusters[:15]:
        rect = patches.Rectangle(
            (c.min_bound[0], c.min_bound[1]), c.extent[0], c.extent[1],
            linewidth=1.2, edgecolor="white", facecolor="none", linestyle="--"
        )
        ax2.add_patch(rect)

    plt.colorbar(im2, ax=ax2, orientation="vertical", shrink=0.75, label="Collision Cost")
    ax1.legend(loc="upper right", frameon=True)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_param_sweep(
    sweep_df,
    output_path: str | Path,
    title: str = "Parameter Sweep Analysis: Voxel Size vs Ground Distance vs DBSCAN eps",
) -> None:
    """Vẽ đồ thị phân tích sweep tham số đa biến."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#ffffff")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)

    # Subplot 1: Voxel Size -> Latency & Cluster Count
    ax1 = axes[0, 0]
    sub1 = sweep_df.groupby("voxel_size")[["latency_ms", "num_clusters"]].mean().reset_index()
    line1 = ax1.plot(sub1["voxel_size"], sub1["latency_ms"], marker="o", color="#d9381e", linewidth=2, label="Latency (ms)")
    ax1.set_xlabel("Voxel Size (m)", fontweight="medium")
    ax1.set_ylabel("Latency (ms)", color="#d9381e", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#d9381e")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(sub1["voxel_size"], sub1["num_clusters"], marker="s", color="#1f77b4", linewidth=2, linestyle="--", label="Num Clusters")
    ax1_twin.set_ylabel("Num Clusters", color="#1f77b4", fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("Voxel Size vs Latency & Cluster Count", fontsize=12, fontweight="bold")

    # Subplot 2: RANSAC ground_thresh -> Obstacle count & Nearest Distance
    ax2 = axes[0, 1]
    sub2 = sweep_df.groupby("ground_thresh")[["obstacle_points", "nearest_dist"]].mean().reset_index()
    ax2.plot(sub2["ground_thresh"], sub2["obstacle_points"], marker="o", color="#2ca02c", linewidth=2, label="Obstacle Points")
    ax2.set_xlabel("Ground Distance Threshold (m)", fontweight="medium")
    ax2.set_ylabel("Obstacle Points Count", color="#2ca02c", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(sub2["ground_thresh"], sub2["nearest_dist"], marker="^", color="#9467bd", linewidth=2, linestyle="--", label="Nearest Dist (m)")
    ax2_twin.set_ylabel("Nearest Obstacle Dist (m)", color="#9467bd", fontweight="bold")
    ax2.set_title("Ground Threshold vs Obstacle Retention", fontsize=12, fontweight="bold")

    # Subplot 3: DBSCAN eps -> Num Clusters & Mean Extent
    ax3 = axes[1, 0]
    sub3 = sweep_df.groupby("dbscan_eps")[["num_clusters", "mean_extent_dx"]].mean().reset_index()
    ax3.plot(sub3["dbscan_eps"], sub3["num_clusters"], marker="s", color="#ff7f0e", linewidth=2, label="Num Clusters")
    ax3.set_xlabel("DBSCAN Epsilon (m)", fontweight="medium")
    ax3.set_ylabel("Num Clusters", color="#ff7f0e", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(sub3["dbscan_eps"], sub3["mean_extent_dx"], marker="d", color="#8c564b", linewidth=2, linestyle="--", label="Cluster Length dx (m)")
    ax3_twin.set_ylabel("Mean Cluster Size dx (m)", color="#8c564b", fontweight="bold")
    ax3.set_title("DBSCAN Epsilon vs Chaining (Cluster Growth)", fontsize=12, fontweight="bold")

    # Subplot 4: Latency Breakdown Bar Chart (ROI, Voxel, RANSAC, DBSCAN)
    ax4 = axes[1, 1]
    timing_cols = [c for c in ["time_roi", "time_voxel", "time_ransac", "time_dbscan"] if c in sweep_df.columns]
    if timing_cols:
        means = [sweep_df[c].mean() for c in timing_cols]
        labels = ["ROI Crop", "Voxel Grid", "RANSAC Plane", "DBSCAN"]
        bars = ax4.bar(labels, means, color=["#17becf", "#bcbd22", "#e377c2", "#7f7f7f"], edgecolor="black", alpha=0.85)
        ax4.set_ylabel("Average Latency (ms)", fontweight="bold")
        ax4.set_title("Pipeline Latency Breakdown (Mean ms)", fontsize=12, fontweight="bold")
        ax4.grid(True, axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.1f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_gt_matching(
    clusters: list,
    gt_boxes: list,
    eval_dict: Dict[str, Any],
    output_path: str | Path,
    title: str = "Obstacle Detection vs 3D Ground Truth Bounding Boxes",
) -> None:
    """Vẽ kết quả so khớp giữa cụm vật cản phát hiện và nhãn Ground Truth."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#ffffff")
    ax.set_title(
        f"{title}\n[Precision: {eval_dict['precision']*100:.1f}% | Recall: {eval_dict['recall']*100:.1f}% | F1: {eval_dict['f1']*100:.1f}%]",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlim(-2, 45)
    ax.set_ylim(-15, 15)
    ax.set_xlabel("X Forward (m)")
    ax.set_ylabel("Y Left/Right (m)")
    ax.set_aspect("equal")

    # Range rings
    for r in [10, 20, 30, 40]:
        c = plt.Circle((0, 0), r, color="#cccccc", fill=False, linestyle=":", alpha=0.6)
        ax.add_patch(c)
    ax.plot(0, 0, marker="^", color="crimson", markersize=10, label="Ego Sensor")

    # Vẽ Ground Truth boxes (xanh lá nét đứt)
    for gt in gt_boxes:
        dx = gt["extent"][0]
        dy = gt["extent"][1]
        rect = patches.Rectangle(
            (gt["min_bound"][0], gt["min_bound"][1]), dx, dy,
            linewidth=2.0, edgecolor="#2ca02c", facecolor="#2ca02c", alpha=0.15, linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(
            gt["center_velo"][0], gt["center_velo"][1], f"GT:{gt['type']}",
            fontsize=8, fontweight="bold", color="#1b661b", ha="center", va="center"
        )

    # Vẽ Detected Clusters (xanh dương nét liền)
    for c in clusters:
        rect = patches.Rectangle(
            (c.min_bound[0], c.min_bound[1]), c.extent[0], c.extent[1],
            linewidth=1.5, edgecolor="#1f77b4", facecolor="#1f77b4", alpha=0.25
        )
        ax.add_patch(rect)
        ax.scatter(c.points[:, 0], c.points[:, 1], color="#1f77b4", s=3.0, alpha=0.7)

    # Custom legend
    p_gt = patches.Patch(edgecolor="#2ca02c", facecolor="none", linestyle="--", linewidth=2.0, label="Ground Truth Box")
    p_det = patches.Patch(edgecolor="#1f77b4", facecolor="none", linewidth=1.5, label="Detected Cluster Box")
    ax.legend(handles=[p_gt, p_det], loc="upper right", frameon=True)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_failure_case_absorption(
    low_obs_pts: np.ndarray,
    ground_pts_good: np.ndarray,
    ground_pts_fail: np.ndarray,
    output_path: str | Path,
) -> None:
    """Minh họa Failure Case 1: Lọc mặt đất quá tay làm mất vật cản thấp (Ground Over-segmentation)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="#ffffff")
    fig.suptitle(
        "Failure Case 1: Low Obstacle Absorption via Large RANSAC Distance Threshold (Debug Layer: Preprocess)",
        fontsize=13, fontweight="bold", y=0.98
    )

    # Case A: Correct threshold (0.15m)
    ax1 = axes[0]
    ax1.set_title("Standard Ground Threshold (0.15m)\n-> Low Obstacle (Pallet/Curb) DETECTED", fontsize=11, fontweight="bold", color="darkgreen")
    ax1.scatter(ground_pts_good[:, 0], ground_pts_good[:, 2], c="#999999", s=2, alpha=0.4, label="Ground")
    ax1.scatter(low_obs_pts[:, 0], low_obs_pts[:, 2], c="red", s=18, alpha=0.9, label="Low Obstacle (<25cm)")
    ax1.set_xlabel("X Forward (m)")
    ax1.set_ylabel("Z Height (m)")
    ax1.set_ylim(-2.2, 0.5)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Case B: Excessive threshold (0.35m)
    ax2 = axes[1]
    ax2.set_title("Excessive Ground Threshold (0.35m)\n-> Low Obstacle Absorbed into Ground (MISSED!)", fontsize=11, fontweight="bold", color="crimson")
    ax2.scatter(ground_pts_fail[:, 0], ground_pts_fail[:, 2], c="#999999", s=2, alpha=0.4, label="Ground (Absorbed)")
    ax2.set_xlabel("X Forward (m)")
    ax2.set_ylabel("Z Height (m)")
    ax2.set_ylim(-2.2, 0.5)
    ax2.annotate(
        "FAIL: Low obstacle absorbed into plane inliers!\nZero obstacle points detected.",
        (low_obs_pts[:, 0].mean() if len(low_obs_pts) else 10.0, -1.5),
        xytext=(15, 0.0), arrowprops=dict(facecolor="crimson", shrink=0.05, width=1.5),
        fontsize=10, fontweight="bold", color="crimson", bbox=dict(boxstyle="round", facecolor="mistyrose")
    )
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_failure_case_chaining(
    pts_obj_a: np.ndarray,
    pts_obj_b: np.ndarray,
    output_path: str | Path,
) -> None:
    """Minh họa Failure Case 2: DBSCAN Chaining Effect / Under-segmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#ffffff")
    fig.suptitle(
        "Failure Case 2: DBSCAN Under-segmentation / Chaining Effect (Debug Layer: Preprocess)",
        fontsize=13, fontweight="bold", y=0.98
    )

    # Proper eps (0.5m)
    ax1 = axes[0]
    ax1.set_title("Proper eps = 0.5m -> 2 Separate Clusters (Pedestrian & Car)", fontsize=11, fontweight="bold", color="darkgreen")
    ax1.scatter(pts_obj_a[:, 0], pts_obj_a[:, 1], c="#1f77b4", s=15, label="Cluster 1 (Pedestrian)")
    ax1.scatter(pts_obj_b[:, 0], pts_obj_b[:, 1], c="#ff7f0e", s=15, label="Cluster 2 (Car)")
    # Draw boxes
    for pts, col in [(pts_obj_a, "#1f77b4"), (pts_obj_b, "#ff7f0e")]:
        mi, ma = pts.min(axis=0), pts.max(axis=0)
        rect = patches.Rectangle((mi[0], mi[1]), ma[0]-mi[0], ma[1]-mi[1], linewidth=1.5, edgecolor=col, facecolor="none")
        ax1.add_patch(rect)
    ax1.set_xlabel("X Forward (m)")
    ax1.set_ylabel("Y Left/Right (m)")
    ax1.legend(loc="upper left")
    ax1.set_aspect("equal")

    # Excessive eps (1.2m)
    ax2 = axes[1]
    ax2.set_title("Excessive eps = 1.2m -> Merged into 1 Giant Cluster (Chaining Fail)", fontsize=11, fontweight="bold", color="crimson")
    all_pts = np.vstack([pts_obj_a, pts_obj_b])
    ax2.scatter(all_pts[:, 0], all_pts[:, 1], c="crimson", s=15, label="Single Merged Cluster")
    mi, ma = all_pts.min(axis=0), all_pts.max(axis=0)
    rect = patches.Rectangle((mi[0], mi[1]), ma[0]-mi[0], ma[1]-mi[1], linewidth=2.0, edgecolor="crimson", facecolor="crimson", alpha=0.15, linestyle="--")
    ax2.add_patch(rect)
    ax2.plot(all_pts.mean(axis=0)[0], all_pts.mean(axis=0)[1], marker="x", markersize=12, color="black", label="False Box Center (Empty Space!)")
    ax2.set_xlabel("X Forward (m)")
    ax2.set_ylabel("Y Left/Right (m)")
    ax2.legend(loc="upper left")
    ax2.set_aspect("equal")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_failure_case_fragmentation(
    near_cluster_pts: np.ndarray,
    far_cluster_pts: np.ndarray,
    output_path: str | Path,
) -> None:
    """Minh họa Failure Case 3: Far-range Sparsity & Cluster Fragmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#ffffff")
    fig.suptitle(
        "Failure Case 3: Far-Range Beam Sparsity & Fragmentation (Debug Layer: Geometry / Sensor)",
        fontsize=13, fontweight="bold", y=0.98
    )

    ax1 = axes[0]
    ax1.set_title("Near Obstacle (12m): Dense Returns -> Solid Single Cluster", fontsize=11, fontweight="bold", color="darkgreen")
    ax1.scatter(near_cluster_pts[:, 0], near_cluster_pts[:, 1], c="#2ca02c", s=15)
    mi, ma = near_cluster_pts.min(axis=0), near_cluster_pts.max(axis=0)
    ax1.add_patch(patches.Rectangle((mi[0], mi[1]), ma[0]-mi[0], ma[1]-mi[1], linewidth=1.5, edgecolor="#2ca02c", facecolor="none"))
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal")

    ax2 = axes[1]
    ax2.set_title("Far Obstacle (42m): Sparse Beams -> Fragmented or Missed", fontsize=11, fontweight="bold", color="crimson")
    ax2.scatter(far_cluster_pts[:, 0], far_cluster_pts[:, 1], c="crimson", s=15)
    ax2.annotate(
        "Beam spacing > eps (0.6m)!\nPoints separated beyond eps,\nbroken into fragments or filtered out as noise.",
        (far_cluster_pts[:, 0].mean(), far_cluster_pts[:, 1].mean()),
        xytext=(far_cluster_pts[:, 0].mean() - 4, far_cluster_pts[:, 1].mean() + 2),
        arrowprops=dict(facecolor="crimson", shrink=0.05, width=1.2),
        fontsize=9, fontweight="bold", color="crimson", bbox=dict(boxstyle="round", facecolor="mistyrose")
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_aspect("equal")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
