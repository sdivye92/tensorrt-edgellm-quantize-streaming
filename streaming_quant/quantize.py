"""Streaming-quantization orchestrator.

``tensorrt_edgellm.quantization.quantize.quantize_and_export`` already
contains all the model-specific dispatch logic Edge-LLM supports for the
``llm`` path -- visual/audio/CP calibration dataloaders, MTP draft
quantization, per-architecture patches, checkpoint export. Only two of its
assumptions are incompatible with a VRAM-constrained GPU:

1. ``_load_model`` always loads the whole model onto one device
   (``.to(device)``).
2. ``build_quant_config`` never sets ModelOpt's ``algorithm.layerwise`` flag,
   so ``mtq.quantize`` calibrates the whole resident model at once instead of
   one decoder layer at a time.

Rather than reimplementing ``quantize_and_export``'s branching, this module
monkeypatches those names inside ``tensorrt_edgellm.quantization.quantize``'s
module namespace for the duration of one call, then invokes the real
``quantize_and_export`` unchanged. Every other branch (visual quantization,
MTP, CP, ...) keeps working exactly as it does in stock Edge-LLM, so any
model shape the generic ``llm`` path already supports is supported here too.

A third name, ``_is_image_blind_calibration``, also needs a thin wrapper:
turning ``algorithm`` from a bare string (``"max"``) into a dict (to carry
``layerwise``) breaks its ``quant_cfg.get("algorithm") not in (None, "max")``
check (`quantize.py:752`), which would otherwise silently force every
non-visual quantization run onto the image-calibration path (confirmed by
running this against Cosmos-Reason2-2B with ``--quantization fp8`` and no
``--visual_quantization``: it selected the image dataset and tried to
download MMMU instead of running the plain text-calibration path).

Qwen3-Omni and Qwen3-Omni-Next are dispatched by ``quantize_and_export``
to dedicated drivers (``qwen3_omni.py``) that build their own model via
``_load_omni_model``/``load_qwen3_asr_joint_for_calibration``, not
``_load_model`` -- those loaders are untouched by this patch and still
require full-GPU residency. See README.md 'Out of scope'.
"""

import functools
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Union

import tensorrt_edgellm.quantization.quantize as _edgellm_quantize

from .loader import load_streaming_model
from .modelopt_patches import calib_activation_offload


def _inject_layerwise(
    algorithm: Union[str, dict],
    layerwise: bool,
    layerwise_checkpoint_dir: Optional[str],
) -> dict:
    """Normalize ``algorithm`` to a dict and merge in the layerwise flags.

    ModelOpt's stock named configs (``mtq.FP8_DEFAULT_CFG`` etc.) carry
    ``algorithm`` as either a bare string (``"max"``) or already a dict (some
    AWQ variants); ``algorithm.layerwise``/``layerwise_checkpoint_dir`` are
    real fields on ModelOpt 0.45.0's ``QuantizeAlgorithmConfig``
    (``modelopt/torch/quantization/config.py``), supported by every
    calibration algorithm Edge-LLM uses (max, AWQ, SmoothQuant) except
    SVDQuant, which Edge-LLM doesn't use.
    """
    algorithm = {"method": algorithm} if isinstance(algorithm,
                                                     str) else dict(algorithm)
    if layerwise:
        algorithm["layerwise"] = True
        if layerwise_checkpoint_dir:
            algorithm["layerwise_checkpoint_dir"] = layerwise_checkpoint_dir
    return algorithm


@contextmanager
def _patched_pipeline(
    offload_folder: Optional[str],
    max_gpu_memory_gb: Optional[float],
    max_cpu_memory_gb: Optional[float],
    use_seq_device_map: bool,
    layerwise: bool,
    layerwise_checkpoint_dir: Optional[str],
    offload_calib_activations: bool,
):
    """Temporarily replace three names inside ``tensorrt_edgellm.quantization.quantize``.

    ``quantize_and_export`` calls all three unqualified, so Python resolves
    them from that module's globals at call time -- patching those globals
    (not the original definitions in ``quantization_configs.py``) is what
    makes the substitution visible to it, and restoring them in ``finally``
    keeps the patch scoped to one call even if it raises.
    """
    original_load_model = _edgellm_quantize._load_model
    original_build_quant_config = _edgellm_quantize.build_quant_config
    original_is_image_blind = _edgellm_quantize._is_image_blind_calibration

    def streaming_load_model(model_dir, dtype="fp16", device="cuda"):
        return load_streaming_model(
            model_dir,
            dtype=dtype,
            device=device,
            offload_folder=offload_folder,
            max_gpu_memory_gb=max_gpu_memory_gb,
            max_cpu_memory_gb=max_cpu_memory_gb,
            use_seq_device_map=use_seq_device_map,
        )

    @functools.wraps(original_build_quant_config)
    def layerwise_build_quant_config(*args: Any, **kwargs: Any):
        cfg = original_build_quant_config(*args, **kwargs)
        cfg["algorithm"] = _inject_layerwise(cfg["algorithm"], layerwise,
                                             layerwise_checkpoint_dir)
        return cfg

    @functools.wraps(original_is_image_blind)
    def string_algorithm_aware_is_image_blind(model, quant_cfg):
        # Undo the str->dict promotion above just for this check, which
        # compares quant_cfg["algorithm"] against the literal string "max".
        algorithm = quant_cfg.get("algorithm")
        if isinstance(algorithm, dict):
            quant_cfg = {**quant_cfg, "algorithm": algorithm.get("method")}
        return original_is_image_blind(model, quant_cfg)

    _edgellm_quantize._load_model = streaming_load_model
    _edgellm_quantize.build_quant_config = layerwise_build_quant_config
    _edgellm_quantize._is_image_blind_calibration = string_algorithm_aware_is_image_blind
    # nullcontext when disabled, so the stock ModelOpt code path is untouched
    # unless the flag is passed.
    activation_offload = (calib_activation_offload()
                          if offload_calib_activations else nullcontext())
    try:
        with activation_offload:
            yield
    finally:
        _edgellm_quantize._load_model = original_load_model
        _edgellm_quantize.build_quant_config = original_build_quant_config
        _edgellm_quantize._is_image_blind_calibration = original_is_image_blind


def quantize_and_export_streaming(
    *,
    offload_folder: Optional[str] = None,
    max_gpu_memory_gb: Optional[float] = None,
    max_cpu_memory_gb: Optional[float] = None,
    use_seq_device_map: bool = False,
    layerwise: bool = False,
    layerwise_checkpoint_dir: Optional[str] = None,
    offload_calib_activations: bool = False,
    **quantize_and_export_kwargs: Any,
) -> str:
    """Streaming/offload-aware wrapper around Edge-LLM's ``quantize_and_export``.

    Every keyword accepted by
    ``tensorrt_edgellm.quantization.quantize.quantize_and_export`` (model_dir,
    output_dir, quantization, visual_quantization, kv_cache_quantization,
    num_samples, ...) is forwarded unchanged via ``quantize_and_export_kwargs``.
    The arguments listed above control only how the model is loaded
    (device_map/offload) and whether ModelOpt calibrates layer-by-layer.
    """
    if offload_calib_activations and not layerwise:
        raise ValueError(
            "offload_calib_activations requires layerwise=True: it offloads "
            "the per-layer calibration activations that only ModelOpt's "
            "layerwise algorithm collects.")
    with _patched_pipeline(offload_folder, max_gpu_memory_gb,
                           max_cpu_memory_gb, use_seq_device_map, layerwise,
                           layerwise_checkpoint_dir,
                           offload_calib_activations):
        return _edgellm_quantize.quantize_and_export(
            **quantize_and_export_kwargs)
