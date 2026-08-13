"""MiniMax H3 ProSampler —— RTX 高清化集成。

封装 Nvidia_RTX_Nodes_ComfyUI 的 nvvfx VideoSuperRes 接口（与 RTXVideoSuperResolution
节点同源），自动按 MAX_PIXELS 切分 batch，规避 8GB 显存压力。
输入/输出均为 ComfyUI IMAGE 格式 [N, H, W, C]（float32，0~1）。
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger("H3ProSampler")

_QUALITY_MAP = {
    "LOW": None,
    "MEDIUM": None,
    "HIGH": None,
    "ULTRA": None,
}
MAX_PIXELS = 1024 * 1024 * 16


def _load_nvvfx():
    try:
        import nvvfx
        _QUALITY_MAP["LOW"] = nvvfx.effects.QualityLevel.LOW
        _QUALITY_MAP["MEDIUM"] = nvvfx.effects.QualityLevel.MEDIUM
        _QUALITY_MAP["HIGH"] = nvvfx.effects.QualityLevel.HIGH
        _QUALITY_MAP["ULTRA"] = nvvfx.effects.QualityLevel.ULTRA
        return nvvfx
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "RTX 高清化需要 nvvfx 库（Nvidia_RTX_Nodes_ComfyUI），且需 NVIDIA 显卡驱动支持。"
        ) from e


def rtx_upscale_frames(images: torch.Tensor, scale: float, quality: str) -> torch.Tensor:
    """对 [N, H, W, C] 帧序列做 RTX 视频超分。"""
    nvvfx = _load_nvvfx()
    if images.ndim != 4:
        raise ValueError(f"RTX 超分需要 [N,H,W,C]，得到 {tuple(images.shape)}")
    b, h, w, c = images.shape
    out_w = max(8, round(w * scale / 8) * 8)
    out_h = max(8, round(h * scale / 8) * 8)
    out_pixels = out_w * out_h
    batch_size = max(1, MAX_PIXELS // out_pixels)
    q = _QUALITY_MAP.get(quality, nvvfx.effects.QualityLevel.HIGH)

    with nvvfx.VideoSuperRes(q) as sr:
        sr.output_width = out_w
        sr.output_height = out_h
        sr.load()
        out_tensor = torch.empty((b, out_h, out_w, c), device=images.device, dtype=images.dtype)
        for i in range(0, b, batch_size):
            batch = images[i:i + batch_size].cuda().permute(0, 3, 1, 2).float().contiguous()
            for j in range(batch.shape[0]):
                dlpack_out = sr.run(batch[j]).image
                out_tensor[i + j: i + j + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)
            del batch
            torch.cuda.empty_cache()
    return out_tensor
