"""MiniMax H3 ProSampler —— 噪点敏感 sigma 预算重分配（自适应步长预计算）。

数学原理
--------
视频扩散中，远景人像的高频细节与高动态区域的运动纹理主要在**中低 sigma 段**
（归一化进度 s ~ 0.15~0.55）成型；此段噪声残留也最严重。我们把 sigma 归一化
轴看作概率空间，构造噪点敏感度密度:

    w(s) = 1 + (boost - 1) * exp( -(s - center)^2 / (2 * width^2) )

然后对离散 sigma 序列做**概率测度变换 + 逆变换采样**（等价于 ODE 积分中按局部
Lipschitz 常数自适应选步的离散化）：在累计权重轴上均匀取目标步数节点，反查
sigma 值。效果是——噪点敏感段步长加密、平坦段步长放宽，总步数预算不变，
算力被重分配到真正影响噪点的区段，从而实现"同样步数更干净 / 同样质量更快"。
"""
from __future__ import annotations

import torch


def _gaussian_weight(s: torch.Tensor, center: float, width: float, boost: float) -> torch.Tensor:
    """噪点敏感度密度（归一化 sigma 轴）。"""
    return 1.0 + (boost - 1.0) * torch.exp(-0.5 * ((s - center) / width) ** 2)


def _inverse_cdf_resample(s: torch.Tensor, w: torch.Tensor, n: int) -> torch.Tensor:
    """在累计权重轴上均匀取 n 个节点，线性插值反查 sigma（保端点、严格单调）。"""
    cdf = torch.cumsum(w, dim=0)
    cdf = cdf / cdf[-1]
    targets = torch.linspace(0.0, 1.0, n, device=s.device, dtype=s.dtype)
    idx = torch.searchsorted(cdf, targets)
    idx = idx.clamp(0, len(s) - 2)
    lo = idx
    hi = (idx + 1).clamp(max=len(s) - 1)
    w_lo = cdf[lo]
    w_hi = cdf[hi]
    denom = (w_hi - w_lo).clamp_min(1e-12)
    alpha = ((targets - w_lo) / denom).clamp(0.0, 1.0)
    s_new = s[lo] * (1.0 - alpha) + s[hi] * alpha
    # 严格单调化：逆变换采样在高权重区会聚集出重复节点，0 步长对采样器致命，
    # 尾部回溯保证相邻节点最小间距 eps，再锁定端点。
    eps = 1e-6
    s_new = s_new.clone()
    for i in range(len(s_new) - 2, -1, -1):
        if s_new[i] <= s_new[i + 1]:
            s_new[i] = s_new[i + 1] + eps
    s_new[0] = s[0]
    s_new[-1] = s[-1]
    return s_new.clamp(s[-1], s[0])


def adaptive_sigmas(
    sigmas: torch.Tensor,
    target_steps: int,
    center: float = 0.32,
    width: float = 0.18,
    boost: float = 2.5,
) -> torch.Tensor:
    """把基线 sigmas 重分配为目标步数，噪点敏感段加密。

    Args:
        sigmas: 递减序列 [N+1]，首项 sigma_max，末项 ≈ 0。
        target_steps: 目标步数（返回序列长度为 target_steps + 1）。
        center / width / boost: 噪点敏感度高斯核参数。

    Returns:
        新 sigmas [target_steps + 1]，单调递减，首尾与输入一致。
    """
    if target_steps < 2 or len(sigmas) < 3:
        return sigmas
    s = sigmas / sigmas[0].clamp_min(1e-8)
    s = s.clamp(0.0, 1.0)
    w = _gaussian_weight(s, center, width, boost)
    s_new = _inverse_cdf_resample(s, w, target_steps + 1)
    return s_new * sigmas[0]


def refine_sigmas_plan(
    sigma_noise: float,
    refine_steps: int,
    device: torch.device | str = "cpu",
    power: float = 1.5,
) -> torch.Tensor:
    """生成单 tile 精修的 sigma 序列：sigma_noise 指数递减到 0。

    power > 1 使尾部（贴近干净的细节成型段）步长更密，符合噪点精修需求。
    """
    steps = max(1, refine_steps)
    t = torch.linspace(1.0, 0.0, steps + 1, device=device)
    sigmas = sigma_noise * (t ** power)
    sigmas[-1] = 0.0
    return sigmas
