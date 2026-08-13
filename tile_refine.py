"""MiniMax H3 ProSampler —— 分块精修 + 多尺度融合。

数学原理
--------
高噪点 tile 从当前近似干净的 latent 回退到小 sigma（img2img 式局部重采样），
只 forward 该 tile 的空间子区域（时间窗保持全帧，维持时间一致性），因此：

    * 显存占用 ≈ 全帧采样的 1/(H_tiles * W_tiles)，8GB 显存友好；
    * 局部精修步数与噪点权重联动（RK45 风格：误差大 -> 步数密），
      refine_steps_i = clamp( round(refine_steps * w_i / (w_i + rk_tol)), 1, 2*refine_steps )

融合采用两种可选策略：
    * feather：overlap 带 cosine 羽化，简单快速；
    * pyramid：空间拉普拉斯金字塔逐层加权重建，边界过渡更自然，
      低频层混合更平滑、高频层保留精修细节。
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

import comfy.samplers
from comfy.nested_tensor import NestedTensor

logger = logging.getLogger("H3ProSampler")


def _feather_weight(h: int, w: int, overlap: int, device, dtype) -> torch.Tensor:
    """[H, W] 羽化权重：中心 1，边缘 overlap 带内线性降到 0。"""
    y = torch.ones(h, device=device, dtype=dtype)
    x = torch.ones(w, device=device, dtype=dtype)
    if overlap > 0:
        ramp = torch.linspace(0.0, 1.0, min(overlap, h), device=device, dtype=dtype)
        y[: ramp.shape[0]] = ramp
        y[-ramp.shape[0]:] = ramp.flip(0)
        ramp = torch.linspace(0.0, 1.0, min(overlap, w), device=device, dtype=dtype)
        x[: ramp.shape[0]] = ramp
        x[-ramp.shape[0]:] = ramp.flip(0)
    return y.unsqueeze(1) * x.unsqueeze(0)  # [H, W]


def _pyramid_blend(
    refined: torch.Tensor,
    original: torch.Tensor,
    weight: torch.Tensor,
    levels: int = 3,
) -> torch.Tensor:
    """空间拉普拉斯金字塔融合。refined/original: [B,C,T,H,W]，weight: [H,W]。"""
    if levels <= 1:
        return refined * weight.unsqueeze(0).unsqueeze(0) + original * (1.0 - weight.unsqueeze(0).unsqueeze(0))
    b, c, t, hh, ww = refined.shape
    a = refined.reshape(b * t, c, hh, ww)
    b0 = original.reshape(b * t, c, hh, ww)
    w5 = weight.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]，随金字塔逐层下采样
    la_list: list[torch.Tensor] = []
    lb_list: list[torch.Tensor] = []
    w_list: list[torch.Tensor] = [w5]
    cur_a, cur_b, cur_w = a, b0, w5
    for _ in range(levels):
        ch, cw = cur_a.shape[-2], cur_a.shape[-1]
        if ch < 4 or cw < 4:
            break
        g_a = F.avg_pool2d(cur_a, 2)
        g_b = F.avg_pool2d(cur_b, 2)
        up_a = F.interpolate(g_a, size=(ch, cw), mode="bilinear", align_corners=False)
        up_b = F.interpolate(g_b, size=(ch, cw), mode="bilinear", align_corners=False)
        la_list.append(cur_a - up_a)
        lb_list.append(cur_b - up_b)
        cur_a, cur_b = g_a, g_b
        cur_w = F.avg_pool2d(cur_w, 2)
        w_list.append(cur_w)
    # 最低频按权重混合，再逐层上采样 + 高频残差按对应层权重混合
    out = cur_a * w_list[-1] + cur_b * (1.0 - w_list[-1])
    for k, (la, lb) in enumerate(zip(reversed(la_list), reversed(lb_list))):
        out = F.interpolate(out, size=(la.shape[-2], la.shape[-1]), mode="bilinear", align_corners=False)
        wk = w_list[len(w_list) - 1 - k]
        if wk.shape[-2:] != (la.shape[-2], la.shape[-1]):
            wk = F.interpolate(wk, size=(la.shape[-2], la.shape[-1]), mode="bilinear", align_corners=False)
        out = out + la * wk + lb * (1.0 - wk)
    return out.reshape(b, c, t, hh, ww)


def _tile_refine_once(
    model,
    tile_video: torch.Tensor,
    audio: torch.Tensor | None,
    positive,
    sigma_noise: float,
    steps_i: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """对单个 tile 做 img2img 式局部精修，返回精修后 [B,C,T,Ht,Wt]。

    无 negative（节点已移除该端口）：纯正向引导，cfg 固定 1.0。
    negative 传 []（非 None）：dynamic 模型下 CFGGuider.inner_set_conds
    会对 negative 迭代，None 不可迭代；[] 安全且 cfg=1.0 时不参与计算。
    """
    tile_video = tile_video.to(device)
    if audio is not None:
        audio = audio.to(device)
    gen = torch.Generator(device=device).manual_seed((seed + 0x9E3779B9) & 0xFFFFFFFF)
    noise_t = torch.randn(tile_video.shape, device=device, dtype=tile_video.dtype, generator=gen)
    if audio is not None:
        noise = NestedTensor([noise_t, audio.clone()])
        latent_image = NestedTensor([tile_video.clone(), audio.clone()])
    else:
        noise = noise_t
        latent_image = tile_video.clone()

    from .sigma_plan import refine_sigmas_plan
    sigmas = refine_sigmas_plan(sigma_noise, steps_i, device=device).to(device)

    sampler_obj = comfy.samplers.sampler_object("euler")
    out = comfy.samplers.sample(
        model, noise, positive, [], 1.0, device,
        sampler_obj, sigmas,
        model_options=model.model_options,
        latent_image=latent_image,
        disable_pbar=True,
        seed=seed,
    )
    if isinstance(out, NestedTensor):
        out = out.tensors[0] if hasattr(out, "tensors") else out.unbind()[0]
    return out.to(device)


def refine_video_tiles(
    model,
    video: torch.Tensor,
    audio: torch.Tensor | None,
    positive,
    weight_map: torch.Tensor,
    range_h: list[tuple[int, int]],
    range_w: list[tuple[int, int]],
    noise_threshold: float,
    refine_steps: int,
    refine_sigma_start: float,
    blend_mode: str,
    seed: int,
    device: torch.device,
    k_cfg: float = 0.5,
    rk_tol: float = 0.05,
    tile_overlap: int = 16,
) -> torch.Tensor:
    """对高噪点 tile 局部精修并融合，返回精修后的完整 video latent。

    Args:
        video: [B, C, T, H, W]（主采样输出，intermediate device）。
        weight_map: [B, H_tiles, W_tiles] 噪点权重。
        ...
    Returns:
        refined video [B, C, T, H, W]。
    """
    if refine_steps <= 0:
        return video
    model_sampling = model.model.model_sampling
    sigma_max = float(model_sampling.sigma_max)
    b, c, t, h, w = video.shape
    out = video.clone().to(device)
    refined_count = 0

    for i, (y0, y1) in enumerate(range_h):
        for j, (x0, x1) in enumerate(range_w):
            w_tile = float(weight_map[0, i, j].item()) if b == 1 else float(weight_map[:, i, j].mean().item())
            if w_tile <= noise_threshold:
                continue
            # RK45 风格：误差越大步数越密；容差越小整体越严格
            steps_i = int(round(refine_steps * w_tile / (w_tile + rk_tol)))
            steps_i = max(1, min(steps_i, refine_steps * 2))
            sigma_noise = refine_sigma_start * sigma_max
            # 无 negative：纯正向引导，精修 cfg 固定 1.0（k_cfg 不再放大 CFG）
            cfg_refine = 1.0

            tile_v = video[:, :, :, y0:y1, x0:x1].to(device)
            refined_tile = _tile_refine_once(
                model, tile_v, audio, positive,
                sigma_noise, steps_i, seed + i * 131 + j, device,
            )
            # 融合
            hh, ww = refined_tile.shape[-2], refined_tile.shape[-1]
            feather = _feather_weight(hh, ww, tile_overlap, device, video.dtype)
            if blend_mode == "pyramid":
                blended = _pyramid_blend(refined_tile, tile_v, feather, levels=3)
            else:
                blended = refined_tile * feather.unsqueeze(0).unsqueeze(0) + tile_v * (1.0 - feather.unsqueeze(0).unsqueeze(0))
            out[:, :, :, y0:y1, x0:x1] = blended.to(out.dtype)
            refined_count += 1
            del tile_v, refined_tile, blended
            torch.cuda.empty_cache()

    if refined_count:
        logger.info("H3ProSampler: refined %d high-noise tile(s)", refined_count)
    else:
        logger.info("H3ProSampler: no high-noise tile detected, skip refine")
    return out
