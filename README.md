
# ComfyUI-MiniMaxH3-ProSampler

MiniMax H3 专用噪点感知采样节点（Noise-Aware Pro Sampler），在保持总采样步数不变的前提下，将算力集中到噪点最易成型的 sigma 段，并对高噪点区域做分块精修，可选接入 **NVIDIA RTX 视频超分（VSR）** 实现高清化输出。

## 特性

- **噪点敏感 sigma 预算重分配**：总步数不变，自适应步长预计算，把采样算力集中到远景/高动态噪点最易成型的 sigma 段，提速且更干净。
- **分块精修（Tile Refine）**：基于 FFT 高频能量 + 时间运动显著性定位高噪点 tile，对局部做小 sigma 重采样（img2img 式），再用拉普拉斯金字塔 / 羽化融合回全局，**8GB 显存友好**。
- **完全兼容原版 KSampler 语义**：支持 sampler / scheduler / steps / denoise，可串联 Block Cache T8 等 H3 生态节点。
- **ref2va 参考生视频适配**：自动识别 `MiniMaxH3ReferenceToVideo` / `MiniMaxH3ImageToVideo` 注入的参考 token，关闭自适应 sigma 重分配并自动补齐 12.0/3.0 的 video/audio shift，稳定参考 token 注入时机。
- **RTX 高清化（可选）**：采样后 VAE 解码，调用 NVIDIA `nvvfx` 视频超分接口，自动按 MAX_PIXELS 切分 batch，规避低显存压力。接入 VAE 后 `images` 端口始终输出解码帧（`rtx_enable=False` 时输出原始帧）。

## 依赖：必须搭配 NVIDIA-RTX-NODES

本节点的 **RTX 高清化功能**依赖 `nvvfx` 库，该库由 NVIDIA 官方节点提供：

- 仓库地址：**[Comfy-Org/Nvidia_RTX_Nodes_ComfyUI](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI)**
- 安装方式见下方「安装教程」。

> 提示：若仅使用噪点感知采样 + 分块精修（`rtx_enable = False`），**不强制**安装 RTX Nodes；只有启用 RTX 超分输出时才需要。

## 节点参数

### 输入端口

| 端口 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | MODEL | 是 | MiniMax H3 扩散模型（可串联 Block Cache T8） |
| positive | CONDITIONING | 是 | 正向提示词条件 |
| latent_image | LATENT | 是 | H3 视频 latent（含音频流） |
| sampler | SAMPLER | 否 | H3 专用采样器（如 MiniMax-H3 Turbo Sampler 4-step）的 SAMPLER 输出，接入后优先使用 |
| vae | VAE | 否 | 接入后启用 RTX 高清化输出 |

### 基础参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| seed | INT | 0 | 0 ~ 2^64-1 | 随机种子，支持 Control After Generate |
| steps | INT | 20 | 1 ~ 10000 | 总采样步数 |
| sampler_name | COMBO | dpmpp_2m | ComfyUI 内置 | 未接入 sampler 端口时的兜底采样器 |
| scheduler | COMBO | karras | ComfyUI 内置 | 调度器 |
| rtx_enable | BOOLEAN | False | - | 启用 RTX 视频超分高清化（需连接 VAE） |

> **参数顺序说明**：`ref2va_mode / shift_video / shift_audio` 固定位于 `rk_tol` 之后、`rtx_enable` 之前。旧工作流按此顺序保存 widget 值，请勿调整顺序，否则会导致旧工作流参数错位。

### 高级参数（Advanced）

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| noise_threshold | FLOAT | 0.45 | 0.0 ~ 1.0 | 高频能量占比阈值，超过才触发精修；越低精修越多 |
| tile_size | INT | 96 | 32 ~ 512 | 精修 tile 空间尺寸（latent 像素） |
| tile_overlap | INT | 16 | 0 ~ 128 | 相邻 tile 重叠，避免接缝 |
| motion_sensitivity | FLOAT | 0.5 | 0.0 ~ 2.0 | 运动显著性权重，高动态画面调大 |
| refine_steps | INT | 3 | 1 ~ 10 | 高噪点 tile 的局部精修步数（按权重自动分配 1~2 倍） |
| refine_sigma_start | FLOAT | 0.6 | 0.05 ~ 2.0 | 精修回退噪声强度（相对 sigma_max），越大去噪越强 |
| blend_mode | COMBO | pyramid | pyramid / feather | 融合方式：pyramid=拉普拉斯金字塔，feather=cosine 羽化 |
| adaptive_step | BOOLEAN | True | - | 噪点敏感 sigma 预算重分配（总步数不变） |
| rk_tol | FLOAT | 0.05 | 0.005 ~ 0.5 | 局部误差容差（RK 风格），越小精修步数越密 |
| ref2va_mode | BOOLEAN | False | - | ref2va 参考生视频模式：配合 `MiniMaxH3ReferenceToVideo` 的 conditioning/latent 使用；开启后关闭自适应 sigma 重分配。也可由节点自动识别参考条件 |
| shift_video | FLOAT | 12.0 | 0.01 ~ 100.0 | H3 视频流 sigma shift，默认与模型一致（不重复 patch）；调小更锐利、调大更平滑 |
| shift_audio | FLOAT | 3.0 | 0.01 ~ 100.0 | H3 音频流 sigma shift，默认与模型一致；仅当需覆盖外部 SigmaShift 节点时调整 |
| rtx_scale | FLOAT | 2.0 | 1.0 ~ 4.0 | RTX 超分倍率 |
| rtx_quality | COMBO | HIGH | LOW / MEDIUM / HIGH / ULTRA | RTX 超分质量档位 |

### 输出

| 端口 | 类型 | 说明 |
|------|------|------|
| latent | LATENT | 精修后的 H3 latent（含音频流），可继续接 VAE Decode |
| images | IMAGE | VAE 已连接时始终输出的视频帧：`rtx_enable=True` 为 RTX 超分帧，`rtx_enable=False` 为原始解码帧（避免下游 SaveVideo 收到空视频报错） |

## 环境要求

| 组件 | 要求 |
|------|------|
| ComfyUI | 支持 `comfy_api.latest` 的新版本（如 ComfyUI-aki 整合包） |
| Python | 3.10+（使用 ComfyUI 自带 Python 环境） |
| PyTorch | 随 ComfyUI 安装的 CUDA 版本即可 |
| 显卡 | NVIDIA RTX 系列（RTX 30/40/50 系），**RTX 高清化功能**需要显卡驱动支持视频超分（VSR） |
| 显存 | 8GB 可流畅运行分块精修（设计目标）；RTX 超分按 MAX_PIXELS 自动切分 batch |
| 额外库 | `nvvfx`（由 Nvidia_RTX_Nodes_ComfyUI 提供，仅启用 RTX 功能时需要） |

## 安装教程

### 方式一：ComfyUI Manager（推荐）

1. 安装 [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. 打开 ComfyUI → Manager → Custom Nodes Manager
3. 搜索 `MiniMax H3 Pro Sampler` 或 `MiniMaxH3ProSampler`
4. 点击 Install，安装完成后重启 ComfyUI

### 方式二：Git Clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YHZR9493/ComfyUI-MiniMaxH3-ProSampler.git
```

重启 ComfyUI 即可。

### 安装依赖节点 NVIDIA-RTX-NODES（可选，仅 RTX 高清化需要）

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git
cd Nvidia_RTX_Nodes_ComfyUI
# 使用 ComfyUI 自带 Python 安装依赖
<ComfyUI_Python>/python.exe -m pip install -r requirements.txt
```

> Windows 下 ComfyUI 整合包的 Python 通常位于 `ComfyUI/python/python.exe`；安装 `nvidia-vfx` 需 NVIDIA 官方驱动支持，且建议关闭代理或配置镜像源后安装。

## 使用说明

典型工作流接线：

```
MiniMax H3 Model ──► model
Prompt ───────────► positive     ┌────────────────────────────┐
H3 Latent ────────► latent_image │  MiniMax H3 Pro Sampler    │
(可选) H3 Turbo ──► sampler      │  (Noise-Aware + RTX)       │
(可选) VAE ───────► vae          └──────┬──────────────┬──────┘
                                        │              │
                                     latent          images
                                        │              │
                                   VAE Decode   (RTX 超分帧)
```

- **纯精修模式**：只连 `model / positive / latent_image`，`rtx_enable = False`。
- **RTX 高清化模式**：连接 `vae` 并打开 `rtx_enable`，从 `images` 端口取超分后的视频帧。

### ref2va 参考生视频（最佳实践）

参考生视频（参考图/视频驱动生成）**必须使用 `res_multistep` 采样方式才是最佳状态**，不要使用默认的 `euler`：

- **采样器**：在 `KSamplerSelect` 中选择 **`res_multistep`**，并将该 SAMPLER 输出接入本节点的 `sampler` 端口（或把 `sampler_name` 设为 `res_multistep`）。
- **步数**：建议约 **20 步**（`steps = 20`），`res_multistep` + 20 步可充分发挥 ref2va 参考 token 的引导效果，画面更清晰、参考一致性更好。
- **参考注入**：参考图/视频必须通过 `MiniMaxH3ReferenceToVideo`（`ref_images` / `ref_videos`）或 `MiniMaxH3ImageToVideo`（`first_frame`）注入 positive 条件；节点会自动检测参考 token 并关闭自适应 sigma 重分配、自动补齐默认 sigma shift（video 12.0 / audio 3.0）。
- **未接参考却开 ref2va_mode**：节点会输出警告日志（`ref2va_mode=True 但 positive 中未检测到参考 token`），此时画面容易发糊——请接上参考图/视频，或关闭 `ref2va_mode`。

## 常见问题

- **启用 RTX 报错 `RTX 高清化需要 nvvfx 库`**：请先安装 Nvidia_RTX_Nodes_ComfyUI 及其 requirements（`nvidia-vfx`），并确认显卡驱动支持 VSR。
- **ref2va 参考生视频画面糊 / 参考不生效**：① 确认参考图/视频已通过 `MiniMaxH3ReferenceToVideo` / `MiniMaxH3ImageToVideo` 注入；② 采样器换成 `res_multistep`（非 euler），步数约 20 步；③ 确认 `positive` 中带 `minimax_refs` 参考 token（节点日志会给出警告）。
- **显存不足**：调小 `tile_size`（如 64）、调小 `tile_overlap`（如 8），或关闭 `rtx_enable`。

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。
*（内容由AI生成，仅供参考）*
