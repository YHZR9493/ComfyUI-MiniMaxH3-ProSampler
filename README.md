# ComfyUI-MiniMaxH3-ProSampler

MiniMax H3 专用噪点感知采样节点（Noise-Aware Pro Sampler），在保持总采样步数不变的前提下，将算力集中到噪点最易成型的 sigma 段，并对高噪点区域做分块精修，可选接入 **NVIDIA RTX 视频超分（VSR）** 实现高清化输出。

## 特性

- **噪点敏感 sigma 预算重分配**：总步数不变，自适应步长预计算，把采样算力集中到远景/高动态噪点最易成型的 sigma 段，提速且更干净。
- **分块精修（Tile Refine）**：基于 FFT 高频能量 + 时间运动显著性定位高噪点 tile，对局部做小 sigma 重采样（img2img 式），再用拉普拉斯金字塔 / 羽化融合回全局，**8GB 显存友好**。
- **完全兼容原版 KSampler 语义**：支持 sampler / scheduler / steps / denoise，可串联 Block Cache T8 等 H3 生态节点。
- **RTX 高清化（可选）**：采样后 VAE 解码，调用 NVIDIA `nvvfx` 视频超分接口，自动按 MAX_PIXELS 切分 batch，规避低显存压力。

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
| rtx_scale | FLOAT | 2.0 | 1.0 ~ 4.0 | RTX 超分倍率 |
| rtx_quality | COMBO | HIGH | LOW / MEDIUM / HIGH / ULTRA | RTX 超分质量档位 |

### 输出

| 端口 | 类型 | 说明 |
|------|------|------|
| latent | LATENT | 精修后的 H3 latent（含音频流），可继续接 VAE Decode |
| images | IMAGE | RTX 超分后的视频帧（需 rtx_enable） |

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

## 常见问题

- **启用 RTX 报错 `RTX 高清化需要 nvvfx 库`**：请先安装 Nvidia_RTX_Nodes_ComfyUI 及其 requirements（`nvidia-vfx`），并确认显卡驱动支持 VSR。
- **显存不足**：调小 `tile_size`（如 64）、调小 `tile_overlap`（如 8），或关闭 `rtx_enable`。

## 许可证

本项目基于 MIT 许可证开源，详见 [LICENSE](LICENSE)。
