"""MiniMax H3 ProSampler —— H3 专用噪点感知采样节点。

主流程
------
1. 噪点敏感 sigma 预算重分配（自适应步长预计算）：总步数不变，把算力集中到
   远景/高动态噪点最易成型的 sigma 段，提速且更干净。
2. 主采样：完全兼容原版 KSampler 语义（sampler/scheduler/cfg/denoise）。
3. 分块精修：FFT 高频能量 + 时间运动显著性定位高噪点 tile，局部小 sigma
   重采样（img2img 式），拉普拉斯金字塔/羽化融合回全局，8GB 显存友好。
4. RTX 高清化：可选，采样后解码 -> nvvfx 视频超分（自动 batch 切分）。
"""
from __future__ import annotations

import logging
import math

import torch

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import comfy.model_sampling
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io, ComfyExtension

from .noise_localization import compute_noise_weight_map
from .sigma_plan import adaptive_sigmas
from .tile_refine import refine_video_tiles
from .rtx_upscale import rtx_upscale_frames

logger = logging.getLogger("H3ProSampler")


def _to_frames(x: torch.Tensor) -> torch.Tensor:
    """VAE decode 输出统一为 ComfyUI IMAGE 格式 [N, H, W, C]。"""
    if isinstance(x, NestedTensor):
        x = x.tensors[0]
    if x.ndim == 5:  # [B, T, H, W, C] -> [B*T, H, W, C]
        b, t, h, w, c = x.shape
        return x.reshape(b * t, h, w, c)
    if x.ndim == 4:
        return x
    raise ValueError(f"无法识别的 VAE 解码输出形状 {tuple(x.shape)}")


def _finite(value, default):
    """数值防御：None / NaN / inf / 非数值字符串一律回退默认值。

    旧工作流在节点 schema 变更后可能出现 widget 值错位（如 scheduler 的
    'simple' 被塞进 noise_threshold、值变成 NaN），此处兜底防止执行崩溃。
    """
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    return fv if math.isfinite(fv) else default


def _sanitize_params(kwargs):
    """将 execute 数值入参做合法性收敛，非法值回退默认并记录日志。"""
    sanitized = {}
    defaults = {
        "seed": 0, "steps": 20, "noise_threshold": 0.45, "tile_size": 96,
        "tile_overlap": 16, "motion_sensitivity": 0.5, "refine_steps": 3,
        "refine_sigma_start": 0.6, "rk_tol": 0.05, "rtx_scale": 2.0,
        "sampler_name": "dpmpp_2m", "scheduler": "karras",
        "blend_mode": "pyramid", "rtx_quality": "HIGH",
        "adaptive_step": True, "rtx_enable": False,
        "ref2va_mode": False, "shift_video": 12.0, "shift_audio": 3.0,
    }
    for key, default in defaults.items():
        val = kwargs.get(key, default)
        if key in ("sampler_name", "scheduler"):
            if not isinstance(val, str) or not val:
                val = default
        elif key == "blend_mode":
            if val not in ("pyramid", "feather"):
                val = default
        elif key == "rtx_quality":
            if val not in ("LOW", "MEDIUM", "HIGH", "ULTRA"):
                val = default
        elif key in ("adaptive_step", "rtx_enable", "ref2va_mode"):
            if not isinstance(val, bool):
                val = default if not isinstance(val, (int, float)) or not math.isfinite(float(val)) else bool(val)
        else:
            fv = _finite(val, default)
            val = int(fv) if key in ("seed", "steps", "tile_size", "tile_overlap", "refine_steps") else fv
        if val != kwargs.get(key):
            sanitized[key] = (kwargs.get(key), val)
            kwargs[key] = val
    if sanitized:
        logger.warning("非法参数已回退默认值: %s", {k: (str(v0), v1) for k, (v0, v1) in sanitized.items()})
    return kwargs


def _apply_h3_shift(model, shift_video, shift_audio):
    """克隆模型并应用 H3 sigma shift（与内置 MiniMaxH3SigmaShift 同机制）。

    仅在用户显式调整 shift 时调用；默认 12.0/3.0 与模型 config 一致时不 patch，
    避免覆盖工作流中外部 SigmaShift 节点设置的调度。
    """
    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    ms = ModelSamplingAdvanced(m.model.model_config)
    ms.set_parameters(shift=shift_video, audio_shift=shift_audio)
    if hasattr(original, "noise_scale"):
        ms.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", ms)
    to = m.model_options.get("transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = shift_video
    to["minimax_h3_sigma_shift_audio"] = shift_audio
    m.model_options["transformer_options"] = to
    return m


def _is_ref2va_conditioning(positive):
    """检测 conditioning 是否携带 ref2va 参考 token（MiniMaxH3ReferenceToVideo 输出）。"""
    if not isinstance(positive, list):
        return False
    return any(isinstance(c, dict) and c.get("minimax_refs") for c in positive)


class MiniMaxH3ProSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ProSampler",
            display_name="MiniMax H3 Pro Sampler (Noise-Aware + RTX)",
            description=(
                "MiniMax H3 专用采样节点：FFT+运动显著性噪点定位、分块精修、"
                "拉普拉斯金字塔融合、RTX 高清化。远景人像/高动态画面去噪增强。"
            ),
            category="sampling",
            search_aliases=["minimax", "h3", "sampler", "noise", "rtx", "tile", "pro"],
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 扩散模型（可串联 Block Cache T8）。"),
                io.Conditioning.Input("positive", tooltip="正向提示词条件。"),
                io.Latent.Input("latent_image", tooltip="H3 视频 latent（含音频流）。"),
                io.Sampler.Input("sampler", optional=True, advanced=True, tooltip="H3 专用采样器：接入 MiniMax-H3 Turbo Sampler (4-step) 的 SAMPLER 输出后优先使用；未接入时回退 sampler_name。"),
                io.Vae.Input("vae", optional=True, advanced=True, tooltip="接入 VAE 后启用 RTX 高清化输出。"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=20, min=1, max=10000, tooltip="总采样步数。"),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="dpmpp_2m", tooltip="未接入 sampler 端口时的兜底采样器。"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="karras"),
                # ---- 噪点定位 ----
                io.Float.Input("noise_threshold", default=0.45, min=0.0, max=1.0, step=0.01, tooltip="高频能量占比阈值，w 超过才触发精修。越低精修越多。", advanced=True),
                io.Int.Input("tile_size", default=96, min=32, max=512, step=8, tooltip="精修 tile 空间尺寸（latent 像素）。", advanced=True),
                io.Int.Input("tile_overlap", default=16, min=0, max=128, step=8, tooltip="相邻 tile 重叠，避免接缝。", advanced=True),
                io.Float.Input("motion_sensitivity", default=0.5, min=0.0, max=2.0, step=0.05, tooltip="运动显著性权重，高动态画面调大。", advanced=True),
                # ---- 分块精修 ----
                io.Int.Input("refine_steps", default=3, min=1, max=10, tooltip="高噪点 tile 的局部精修步数（按权重自动分配 1~2x）。", advanced=True),
                io.Float.Input("refine_sigma_start", default=0.6, min=0.05, max=2.0, step=0.05, tooltip="精修回退噪声强度（相对 sigma_max）。越大去噪越强。", advanced=True),
                io.Combo.Input("blend_mode", options=["pyramid", "feather"], default="pyramid", tooltip="融合方式：pyramid=拉普拉斯金字塔，feather=cosine 羽化。", advanced=True),
                # ---- 数学采样 ----
                io.Boolean.Input("adaptive_step", default=True, tooltip="噪点敏感 sigma 预算重分配（总步数不变，算力集中到噪点段）。", advanced=True),
                io.Float.Input("rk_tol", default=0.05, min=0.005, max=0.5, step=0.005, tooltip="局部误差容差（RK 风格）：越小精修步数越密。", advanced=True),
                # ---- ref2va 参考生视频（位置与旧工作流保存顺序一致：rk_tol 之后、RTX 之前；改动此顺序会致旧工作流 widget 值错位）----
                io.Boolean.Input("ref2va_mode", default=False, tooltip="ref2va 参考生视频模式：配合 MiniMaxH3ReferenceToVideo 的 conditioning/latent 使用。开启后关闭自适应 sigma 重分配，稳定参考 token 注入（官方 ref2va 链路用普通 simple 调度）。也可由节点自动识别参考条件。", advanced=True),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01, tooltip="H3 视频流 sigma shift。默认 12.0 与模型一致（不重复 patch）；调小更锐利、调大更平滑。", advanced=True),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01, tooltip="H3 音频流 sigma shift。默认 3.0 与模型一致；仅当需覆盖外部 SigmaShift 节点时调整。", advanced=True),
                # ---- RTX ----
                io.Boolean.Input("rtx_enable", default=False, tooltip="启用 RTX 视频超分高清化（需连接 VAE）。"),
                io.Float.Input("rtx_scale", default=2.0, min=1.0, max=4.0, step=0.01, advanced=True),
                io.Combo.Input("rtx_quality", options=["LOW", "MEDIUM", "HIGH", "ULTRA"], default="HIGH", advanced=True),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="精修后的 H3 latent（含音频流），可继续接 VAE Decode。"),
                io.Image.Output("images", tooltip="RTX 超分后的视频帧（需 rtx_enable）。"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        positive,
        latent_image,
        sampler=None,
        vae=None,
        seed=0,
        steps=20,
        cfg=6.0,
        sampler_name="dpmpp_2m",
        scheduler="karras",
        denoise=1.0,
        noise_threshold=0.45,
        tile_size=96,
        tile_overlap=16,
        motion_sensitivity=0.5,
        refine_steps=3,
        refine_sigma_start=0.6,
        blend_mode="pyramid",
        adaptive_step=True,
        k_cfg=0.5,
        rk_tol=0.05,
        ref2va_mode=False,
        shift_video=12.0,
        shift_audio=3.0,
        rtx_enable=False,
        rtx_scale=2.0,
        rtx_quality="HIGH",
        **kwargs,  # 兼容旧工作流残留输入（已移除的 negative 等），静默忽略
    ) -> io.NodeOutput:
        # 数值防御：旧工作流 widget 错位产生的 NaN/字符串值一律回退默认值
        p = _sanitize_params(dict(
            seed=seed, steps=steps, sampler_name=sampler_name, scheduler=scheduler,
            noise_threshold=noise_threshold, tile_size=tile_size, tile_overlap=tile_overlap,
            motion_sensitivity=motion_sensitivity, refine_steps=refine_steps,
            refine_sigma_start=refine_sigma_start, blend_mode=blend_mode,
            adaptive_step=adaptive_step, rk_tol=rk_tol, rtx_enable=rtx_enable,
            rtx_scale=rtx_scale, rtx_quality=rtx_quality,
            ref2va_mode=ref2va_mode, shift_video=shift_video, shift_audio=shift_audio,
        ))
        seed, steps, sampler_name, scheduler = p["seed"], p["steps"], p["sampler_name"], p["scheduler"]
        noise_threshold, tile_size, tile_overlap = p["noise_threshold"], p["tile_size"], p["tile_overlap"]
        motion_sensitivity, refine_steps, refine_sigma_start = p["motion_sensitivity"], p["refine_steps"], p["refine_sigma_start"]
        blend_mode, adaptive_step, rk_tol = p["blend_mode"], p["adaptive_step"], p["rk_tol"]
        rtx_enable, rtx_scale, rtx_quality = p["rtx_enable"], p["rtx_scale"], p["rtx_quality"]
        ref2va_mode, shift_video, shift_audio = p["ref2va_mode"], p["shift_video"], p["shift_audio"]
        if kwargs.get("negative") is not None:
            logger.info("忽略旧工作流残留的 negative 输入（端口已移除，CFG 固定 1.0）")
        lat = latent_image["samples"]
        is_nested = isinstance(lat, NestedTensor)
        load_dev = model.load_device

        # ---- 0. ref2va 参考生视频适配 ----
        # ref2va checkpoint 必须携带参考 token 条件（MiniMaxH3ReferenceToVideo 输出）。
        # 检测到参考条件或手动开启 ref2va_mode 时关闭自适应 sigma 重分配，
        # 与官方 ref2va 链路（simple 调度）保持一致，稳定参考 token 注入时机。
        is_ref2va = ref2va_mode or _is_ref2va_conditioning(positive)
        if ref2va_mode and not _is_ref2va_conditioning(positive):
            logger.warning(
                "ref2va_mode=True 但 positive 中未检测到参考 token（minimax_refs）。"
                "请确认参考图/视频已通过 MiniMaxH3ReferenceToVideo 的 ref_images/ref_videos "
                "或 MiniMaxH3ImageToVideo 的 first_frame 注入；否则 ref2va 模型在无参考时画面会糊。"
            )
        use_adaptive = adaptive_step and not is_ref2va
        if is_ref2va and adaptive_step:
            logger.info("ref2va 模式：关闭自适应 sigma 重分配，使用模型原生调度")
        # 用户显式调整 shift 时总是 patch 模型；ref2va 模式下若模型尚未被外部
        # MiniMaxH3SigmaShift patch（transformer_options 无 minimax_h3_sigma_shift_video），
        # 也用默认 12.0/3.0 自动补上，避免漏接 SigmaShift 节点导致 sigma 调度不匹配。
        to_opts = model.model_options.get("transformer_options", {})
        already_shifted = "minimax_h3_sigma_shift_video" in to_opts
        need_default_shift = is_ref2va and not already_shifted
        if abs(shift_video - 12.0) > 1e-9 or abs(shift_audio - 3.0) > 1e-9 or need_default_shift:
            try:
                model = _apply_h3_shift(model, shift_video, shift_audio)
                logger.info("已应用 H3 sigma shift: video=%.2f audio=%.2f", shift_video, shift_audio)
            except Exception:  # noqa: BLE001
                logger.exception("H3 sigma shift patch 失败，使用模型默认调度")

        # ---- 1. 噪点敏感 sigma 预算重分配 + 主采样 ----
        # 已移除 negative 端口：无 CFG 引导，cfg 固定按 1.0（纯正向引导）。
        # negative 传空列表 [] 而非 None：MiniMaxH3 属 dynamic 模型，
        # CFGGuider.inner_set_conds 会对 negative 做 cond_has_hooks/convert_cond
        # 迭代，None 不可迭代会抛 TypeError；[] 安全，且 cfg=1.0 时
        # sampling_function 将 uncond 置 None 不参与计算。
        effective_cfg = 1.0
        noise = comfy.sample.prepare_noise(lat, seed, latent_image.get("batch_index"))
        if sampler is not None:
            # H3 专用采样器（SAMPLER 对象，如 MiniMax-H3 Turbo Sampler 4-step）：
            # KSampler.__init__ 会对非字符串 sampler 做成员校验并覆盖，因此这里
            # 只用 euler 生成调度 sigmas，实际步进交给传入的 SAMPLER 对象。
            ks = comfy.samplers.KSampler(
                model,
                steps=steps,
                device=load_dev,
                sampler="euler",
                scheduler=scheduler,
                denoise=denoise,
                model_options=model.model_options,
            )
            sigmas = ks.sigmas
            if use_adaptive and len(sigmas) >= 4:
                try:
                    sigmas = adaptive_sigmas(sigmas, max(2, len(sigmas) - 1)).to(load_dev)
                except Exception:  # noqa: BLE001
                    logger.exception("adaptive sigmas 失败，回退基线调度")
            samples = comfy.samplers.sample(
                model, noise, positive, [], effective_cfg, load_dev,
                sampler, sigmas,
                model_options=model.model_options,
                latent_image=lat,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=seed,
            )
        else:
            ks = comfy.samplers.KSampler(
                model,
                steps=steps,
                device=load_dev,
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                model_options=model.model_options,
            )
            sigmas = ks.sigmas
            if use_adaptive and len(sigmas) >= 4:
                try:
                    sigmas = adaptive_sigmas(sigmas, max(2, len(sigmas) - 1)).to(load_dev)
                except Exception:  # noqa: BLE001
                    logger.exception("adaptive sigmas 失败，回退基线调度")
            samples = ks.sample(
                noise, positive, [], effective_cfg,
                latent_image=lat, sigmas=sigmas,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=seed,
            )

        # ---- 2. 噪点定位 + 分块精修 + 融合 ----
        if refine_steps > 0:
            video = samples.tensors[0] if is_nested else samples
            audio = samples.tensors[1] if is_nested and len(samples.tensors) > 1 else None
            video = video.to(load_dev)
            wmap, rh, rw, _ = compute_noise_weight_map(
                video, tile_size, tile_overlap, motion_sensitivity, noise_threshold
            )
            video_refined = refine_video_tiles(
                model, video, audio, positive,
                wmap, rh, rw,
                noise_threshold=noise_threshold,
                refine_steps=refine_steps,
                refine_sigma_start=refine_sigma_start,
                blend_mode=blend_mode,
                seed=seed,
                device=load_dev,
                k_cfg=k_cfg,
                rk_tol=rk_tol,
                tile_overlap=tile_overlap,
            )
            if is_nested:
                refined = NestedTensor([video_refined] + list(samples.tensors[1:]))
            else:
                refined = video_refined
            del video, wmap
            torch.cuda.empty_cache()
        else:
            refined = samples

        # ---- 3. 移回 intermediate device，组装 latent ----
        if isinstance(refined, NestedTensor):
            refined = refined.to(comfy.model_management.intermediate_device())
        else:
            refined = refined.to(comfy.model_management.intermediate_device())
        out = dict(latent_image)
        out["samples"] = refined

        # ---- 4. 视频帧输出（VAE 已连接时始终可用）----
        # rtx_enable=True：输出 RTX 超分帧；rtx_enable=False：输出原始解码帧。
        # 若 VAE 已连接却不输出，下游 CreateVideo/SaveVideo 会收到空视频，
        # 报 'NoneType' object has no attribute 'shape'。
        images = None
        if vae is not None:
            # NestedTensor（视频+音频 latent 对）需先取视频分量再交给 VAE，
            # 否则 VAE 内部 .to(z) 收到 NestedTensor 会抛
            # TypeError: to() received an invalid combination of arguments。
            decode_samples = out["samples"]
            if isinstance(decode_samples, NestedTensor):
                decode_samples = decode_samples.unbind()[0]
            decoded = vae.decode(decode_samples)
            frames = _to_frames(decoded)
            images = rtx_upscale_frames(frames, rtx_scale, rtx_quality) if rtx_enable else frames
        elif rtx_enable:
            raise ValueError("RTX 高清化需要连接 VAE 输入")

        return io.NodeOutput(out, images)


class H3ProSamplerExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ProSampler]


async def comfy_entrypoint():
    return H3ProSamplerExtension()
