"""Streaming / VRAM-constrained quantization for TensorRT-Edge-LLM.

Adds accelerate-based CPU/disk offload model loading and ModelOpt's
``layerwise`` calibration to ``tensorrt_edgellm``'s quantization pipeline, so
models much larger than available GPU VRAM can be quantized. See README.md
for background and usage.
"""

from .modelopt_patches import register_vlm_layerwise_support
from .quantize import quantize_and_export_streaming

register_vlm_layerwise_support()

__all__ = ["quantize_and_export_streaming"]
