"""GPU 粒子系统：负责粒子初始化、随机机动、扫描剔除与密度聚合。"""

import math

import numpy as np
import torch


class ParticleSystem:
    """封装与目标粒子相关的 CUDA 张量与计算流程。"""

    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.device_runtime is None:
            raise RuntimeError("Device is not configured. Build config before ParticleSystem.")
        self.device = self.cfg.device_runtime.device

        # 初始化粒子位置/朝向/活跃掩码。
        self.p_locs, self.p_angles, self.p_active_mask = self._init_particles_on_device(
            self.cfg.simulation.n_particles
        )
        # 全活跃起步，首次无需 nonzero 重建。
        self.active_idx_cache = torch.arange(
            self.cfg.simulation.n_particles,
            device=self.device,
            dtype=torch.int64,
        )

    @property
    def active_count(self) -> int:
        """返回当前活跃粒子数量。"""
        return int(self.active_idx_cache.numel()) if self.cfg.debug.use_active_index_cache else int(
            torch.count_nonzero(self.p_active_mask).item()
        )

    def _init_particles_on_device(self, num_particles: int):
        """
        在设备上构造初始粒子分布。

        方法：网格分层 + 单元内随机抖动（jitter），再映射到整数坐标系。
        """
        d = self.cfg.derived

        grid_nx = max(1, int(math.sqrt(num_particles * d.area_width_u / d.area_height_u)))
        grid_ny = int(math.ceil(num_particles / grid_nx))

        # 每个粒子先绑定网格单元，再做单元内抖动。
        cell_indices = torch.arange(num_particles, device=self.device, dtype=torch.int64)
        cell_x = cell_indices.remainder(grid_nx).to(torch.float32)
        cell_y = torch.div(cell_indices, grid_nx, rounding_mode="floor").to(torch.float32)

        jitter_x = torch.rand(num_particles, device=self.device)
        jitter_y = torch.rand(num_particles, device=self.device)

        # 统一使用整数网格坐标，降低长期迭代中的浮点漂移风险。
        x = torch.round(((cell_x + jitter_x) / float(grid_nx)) * d.area_width_u)
        y = torch.round(((cell_y + jitter_y) / float(grid_ny)) * d.area_height_u)

        x = torch.clamp(x, 0, d.area_width_u).to(torch.int32)
        y = torch.clamp(y, 0, d.area_height_u).to(torch.int32)

        p_locs = torch.stack((x, y), dim=1)
        p_angles = torch.rand(num_particles, device=self.device, dtype=torch.float32) * d.two_pi
        p_active = torch.ones(num_particles, device=self.device, dtype=torch.bool)
        return p_locs, p_angles, p_active

    def _get_active_idx(self):
        """获取活跃粒子索引；默认优先复用缓存，避免每步 nonzero。"""
        if self.cfg.debug.use_active_index_cache:
            return self.active_idx_cache
        return torch.nonzero(self.p_active_mask, as_tuple=True)[0]

    def update_particles(self) -> None:
        """推进活跃粒子一步：随机方向位移 + 边界反射。"""
        d = self.cfg.derived
        active_idx = self._get_active_idx()
        if active_idx.numel() == 0:
            return

        new_angles = torch.rand(active_idx.numel(), device=self.device, dtype=torch.float32) * d.two_pi
        dx = torch.round(d.particle_step_u * torch.cos(new_angles)).to(torch.int32)
        dy = torch.round(d.particle_step_u * torch.sin(new_angles)).to(torch.int32)

        p_active = self.p_locs[active_idx]
        p_active[:, 0] += dx
        p_active[:, 1] += dy

        mask_x = (p_active[:, 0] < 0) | (p_active[:, 0] > d.area_width_u)
        mask_y = (p_active[:, 1] < 0) | (p_active[:, 1] > d.area_height_u)

        # 向量化反射：触碰左右边界做 x 法线反射；上下边界做 y 法线反射。
        reflected_x = (d.pi - new_angles).remainder(d.two_pi)
        reflected_xy = (-reflected_x).remainder(d.two_pi)
        new_angles = torch.where(mask_x, reflected_x, new_angles)
        new_angles = torch.where(mask_y, reflected_xy, new_angles)

        p_active[:, 0] = torch.clamp(p_active[:, 0], 0, d.area_width_u)
        p_active[:, 1] = torch.clamp(p_active[:, 1], 0, d.area_height_u)

        self.p_locs[active_idx] = p_active
        self.p_angles[active_idx] = new_angles

    def remove_scanned_particles(self, uav_pos_u: tuple[int, int], detection_probability: float) -> int:
        """
        按 UAV 扫描圆 + 命中概率剔除粒子。

        注意：不做物理删除，仅更新 active_mask 并维护 active_idx_cache。
        """
        d = self.cfg.derived
        active_idx = self._get_active_idx()
        if active_idx.numel() == 0:
            return 0

        uav_x, uav_y = int(uav_pos_u[0]), int(uav_pos_u[1])

        # 坐标转 int64，避免平方后 int32 溢出。
        x = self.p_locs[active_idx, 0].to(torch.int64)
        y = self.p_locs[active_idx, 1].to(torch.int64)
        dx = x - uav_x
        dy = y - uav_y
        dist2 = dx * dx + dy * dy
        hit_mask = dist2 <= d.uav_scan_radius_u2

        p = float(detection_probability)
        if torch.any(hit_mask):
            if p >= 1.0:
                remove_mask = hit_mask
            elif p <= 0.0:
                remove_mask = torch.zeros_like(hit_mask, dtype=torch.bool)
            else:
                hit_rand = torch.rand(active_idx.numel(), device=self.device, dtype=torch.float32)
                remove_mask = hit_mask & (hit_rand < p)
            self.p_active_mask[active_idx[remove_mask]] = False
        else:
            remove_mask = torch.zeros_like(hit_mask, dtype=torch.bool)

        # 命中剔除后直接收缩索引缓存，维持增量更新路径。
        next_active_idx = active_idx[~remove_mask]
        if self.cfg.debug.use_active_index_cache:
            self.active_idx_cache = next_active_idx
            return int(self.active_idx_cache.numel())

        return int(torch.count_nonzero(self.p_active_mask).item())

    def get_counts_in_grids(self) -> np.ndarray:
        """
        统计网格密度并返回 CPU numpy 矩阵（供可视化使用）。

        这是允许 CUDA 同步的路径；其余计算路径尽量保持异步。
        """
        d = self.cfg.derived
        active_idx = self._get_active_idx()
        if active_idx.numel() == 0:
            return np.zeros((d.n_y_bins, d.n_x_bins), dtype=np.float32)

        x = self.p_locs[active_idx, 0].to(torch.int64)
        y = self.p_locs[active_idx, 1].to(torch.int64)

        x_bin = torch.div(x, d.grid_size_u, rounding_mode="floor")
        y_bin = torch.div(y, d.grid_size_u, rounding_mode="floor")
        x_bin = torch.clamp(x_bin, 0, d.n_x_bins - 1)
        y_bin = torch.clamp(y_bin, 0, d.n_y_bins - 1)

        linear = y_bin * d.n_x_bins + x_bin
        counts_1d = torch.bincount(linear, minlength=d.n_x_bins * d.n_y_bins)
        counts_2d = counts_1d.reshape(d.n_y_bins, d.n_x_bins).to(torch.float32)

        # 仅在回传绘图数据时同步，避免无谓阻塞。
        torch.cuda.synchronize()
        return counts_2d.cpu().numpy()
