"""MiniMax H3 ProSampler —— 噪点定位。

数学原理
--------
噪点 = 高频能量异常集中 + 时间维度不稳定的残差。对视频 latent（[B,C,T,H,W]）
做三维分解：

1. **空间高频能量比**：对每帧做 2D FFT（rfft2，正交归一），定义径向频率
   高于 Nyquist 一半的高频带能量占比:

       E_high(frame) = Σ_{k: |k| > 0.5*|k_max|} |X̂(k)|² / Σ_k |X̂(k)|²

   远景人像（小目标、高频细节）与高动态纹理处的 E_high 显著偏高——这正是
   扩散模型采样残留噪声最容易聚集的地方。

2. **时间运动显著性**：帧间时间梯度均值:

       M = mean_t ||x_{t+1} - x_t||²

   高动态画面的 M 高，也是噪点重灾区。

3. 综合为 per-tile 噪点权重:

       w = clip( 0.5 * (E_high - threshold)/(1 - threshold), 0, 1 )
           + 0.5 * motion_sensitivity * M_norm

   w ∈ [0,1]，越接近 1 表示该 tile 越需要精修。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def tile_ranges(size: int, tile_size: int, overlap: int) -> list[tuple[int, int]]:
    """一维 tile 划分：[(start, end), ...]，相邻 tile 共享 overlap。"""
    if tile_size >= size:
        return [(0, size)]
    step = max(1, tile_size - overlap)
    ranges: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + tile_size, size)
        ranges.append((start, end))
        if end >= size:
            break
        start = end - overlap
        if start >= size - 1:
            ranges.append((size - tile_size, size))
            break
    return ranges


def _high_freq_energy_ratio(x: torch.Tensor) -> torch.Tensor:
    """x: [B, C, H, W] -> 每帧高频能量占比 [B]（径向频率 > 0.5 Nyquist）。"""
    b = x.shape[0]
    h, w = x.shape[-2], x.shape[-1]
    X = torch.fft.rfft2(x, norm="ortho")  # [B, C, H, W//2+1]
    ky = torch.fft.fftfreq(h, device=x.device).abs().unsqueeze(-1)  # [H, 1]
    kx = torch.fft.rfftfreq(w, device=x.device).abs().unsqueeze(0)  # [1, W//2+1]
    radial = (ky * ky + kx * kx).sqrt()
    max_freq = radial.max()
    high_mask = radial > (0.5 * max_freq)
    energy = (X.abs() ** 2).sum(dim=1)  # [B, H, W//2+1]
    e_high = energy[:, high_mask].sum(dim=1)
    e_total = energy.sum(dim=(1, 2)).clamp_min(1e-8)
    return e_high / e_total


def _motion_saliency(video: torch.Tensor) -> torch.Tensor:
    """video: [B, C, T, H, W] -> 时间运动显著性 [B]，batch 内归一化到 [0,1]。"""
    if video.shape[2] < 2:
        return torch.zeros(video.shape[0], device=video.device, dtype=video.dtype)
    diff = (video[:, :, 1:] - video[:, :, :-1]).abs().mean(dim=(1, 2, 3, 4))  # [B]
    denom = diff.max().clamp_min(1e-8)
    return diff / denom


def compute_noise_weight_map(
    video: torch.Tensor,
    tile_size: int = 96,
    overlap: int = 16,
    motion_sensitivity: float = 0.5,
    noise_threshold: float = 0.45,
) -> tuple[torch.Tensor, list[tuple[int, int]], list[tuple[int, int]], torch.Tensor]:
    """计算视频 latent 的 per-tile 噪点权重图。

    Args:
        video: [B, C, T, H, W]（视频 latent，float）。
        tile_size / overlap: 空间 tile 尺寸与重叠。
        motion_sensitivity: 运动项权重系数。
        noise_threshold: 高频能量占比阈值（0~1）。

    Returns:
        (weight_map, range_h, range_w, motion_map)
        weight_map: [B, H_tiles, W_tiles]，w ∈ [0,1]。
        range_h / range_w: 各 tile 的 (start, end)。
        motion_map: [B, H_tiles, W_tiles]，运动显著性分量（调试用）。
    """
    if video.ndim != 5:
        raise ValueError(f"视频 latent 需要 [B,C,T,H,W]，得到 {tuple(video.shape)}")
    b, c, t, h, w = video.shape
    rh = tile_ranges(h, tile_size, overlap)
    rw = tile_ranges(w, tile_size, overlap)
    device = video.device
    dtype = video.dtype

    weight = torch.zeros(b, len(rh), len(rw), device=device, dtype=dtype)
    motion_map = torch.zeros_like(weight)
    # 时间梯度预计算一次（全 latent，避免每个 tile 重复算）
    if t >= 2:
        tgrad = (video[:, :, 1:] - video[:, :, :-1]).abs().mean(dim=(1, 2))  # [B, H, W]
        tgrad_norm = tgrad / tgrad.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    else:
        tgrad_norm = None

    for i, (y0, y1) in enumerate(rh):
        for j, (x0, x1) in enumerate(rw):
            tile = video[:, :, :, y0:y1, x0:x1]  # [B, C, T, Ht, Wt]
            frames = tile.permute(0, 2, 1, 3, 4).reshape(b * t, c, y1 - y0, x1 - x0)
            e_high = _high_freq_energy_ratio(frames)  # [B*T]
            e_high = e_high.reshape(b, t).mean(dim=1)  # [B]
            denom = max(1e-6, 1.0 - noise_threshold)
            hf = ((e_high - noise_threshold) / denom).clamp(0.0, 1.0)
            if tgrad_norm is not None:
                mt = tgrad_norm[:, y0:y1, x0:x1].mean(dim=(1, 2))  # [B]
            else:
                mt = torch.zeros(b, device=device, dtype=dtype)
            motion_map[:, i, j] = mt
            weight[:, i, j] = 0.5 * hf + 0.5 * motion_sensitivity * mt

    return weight.clamp(0.0, 1.0), rh, rw, motion_map
