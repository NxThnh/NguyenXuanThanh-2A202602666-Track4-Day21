# Báo cáo Day 6: [ĐIỀN tên đề tài ngắn]

- **Họ tên:** Nguyễn Xuân Thành
- **MSSV:** 2A202602666
- **Lớp:** AI20K-T4
- **Link repo:** https://github.com/NxThnh/NguyenXuanThanh-2A202602666-Track4-Day21
- **Topic:** D — Robot/drone obstacle: phát hiện vật cản bằng hình học không gian (Voxel Downsample + RANSAC Ground Removal + DBSCAN Clustering + 2D BEV Occupancy Grid)
- **Dataset:** data/kitti_mini, data/nuscenes_mini_subset, data/synthetic
- **Các frame đã dùng:** 000011, 000001, 000021, 000049, scene-0103_010

## 1. Claim

Pipeline phát hiện vật cản hình học (Voxel Grid 0.10 m + RANSAC Ground distance_threshold = 0.15 m + DBSCAN eps = 0.60 m) đạt Recall > 85% trên KITTI-mini với latency CPU < 60 ms; tuy nhiên khi nới lỏng distance_threshold lên 0.35 m, 100% vật cản thấp (< 20 cm) bị hấp thụ vào mặt phẳng đất.

## 2. Evidence

| Cấu hình / mức perturb | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| [ĐIỀN] | | | |

![demo](../results/figures/[ĐIỀN].png)

## 3. Failure case

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

[ĐIỀN]

## 5. Cách chạy lại

```bash
[ĐIỀN]
```

## 6. Khai báo sử dụng AI

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
