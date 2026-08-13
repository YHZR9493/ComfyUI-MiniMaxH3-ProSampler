from comfy_api.latest import ComfyExtension

from .nodes import MiniMaxH3ProSampler, H3ProSamplerExtension

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ProSampler": MiniMaxH3ProSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ProSampler": "MiniMax H3 Pro Sampler (Noise-Aware + RTX)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "H3ProSamplerExtension", "comfy_entrypoint"]


async def comfy_entrypoint():
    return H3ProSamplerExtension()
