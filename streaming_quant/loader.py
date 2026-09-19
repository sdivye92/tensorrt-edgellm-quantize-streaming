"""Accelerate offload-aware model loading for streaming quantization.

Mirrors ``tensorrt_edgellm.quantization.quantize._load_model``'s dispatch
(Phi-4MM / NemotronH / Qwen3-ASR special cases, then a generic Auto* factory
fallback) exactly, except the generic branch's ``.from_pretrained(...).to(device)``
is replaced with a ``device_map`` / ``max_memory`` / ``offload_folder`` load.
This is the same pattern NVIDIA's own ModelOpt ships in
``examples/hf_ptq/example_utils.py::get_model()`` (nvidia-modelopt==0.45.0,
the exact version tensorrt_edgellm pins) for disk-offload PTQ loading.

Weights that don't fit in the GPU budget land on CPU; weights that don't fit
GPU+CPU spill to ``offload_folder`` on disk. ModelOpt's ``layerwise``
calibration (see quantize.py) then pages each decoder layer's real weights
in from wherever accelerate parked them, right before calibrating it, via
its own accelerate-hook-aware materialization -- this loader only needs to
get the model into a dispatched/offloaded state; it does not need to know
anything about layer-by-layer paging itself.
"""

import gc
from typing import Optional

import torch
from accelerate.utils import get_max_memory
from transformers import (AutoModel, AutoModelForCausalLM,
                           AutoModelForImageTextToText,
                           AutoModelForTextToWaveform, AutoProcessor,
                           AutoTokenizer)

from tensorrt_edgellm.quantization.quantize import (_is_nemotron_h_model,
                                                     _is_phi4mm_model)
from tensorrt_edgellm.quantization.qwen3_asr_loader import is_qwen3_asr_model

_BYTES_PER_GB = 1024**3


def _resolve_device_map_and_memory(
    device: str,
    max_gpu_memory_gb: Optional[float],
    max_cpu_memory_gb: Optional[float],
    use_seq_device_map: bool,
):
    """Compute the accelerate ``device_map``/``max_memory`` to load under.

    Mirrors ModelOpt's own disk-offload branch (``example_utils.py``): the
    memory budgets are handed straight to ``from_pretrained`` and accelerate
    does its own placement planning internally -- no separate
    ``infer_auto_device_map`` pre-pass is needed.
    """
    if device == "cpu":
        return "cpu", None

    max_memory = get_max_memory()
    for key in list(max_memory):
        if isinstance(key, int):
            if max_gpu_memory_gb is not None:
                max_memory[key] = int(max_gpu_memory_gb * _BYTES_PER_GB)
        elif key == "cpu" and max_cpu_memory_gb is not None:
            max_memory[key] = int(max_cpu_memory_gb * _BYTES_PER_GB)

    device_map = "sequential" if use_seq_device_map else "auto"
    return device_map, max_memory


def load_streaming_model(
    model_dir: str,
    dtype: str = "fp16",
    device: str = "cuda",
    offload_folder: Optional[str] = None,
    max_gpu_memory_gb: Optional[float] = None,
    max_cpu_memory_gb: Optional[float] = None,
    use_seq_device_map: bool = False,
):
    """Load model + tokenizer + optional processor with accelerate offload.

    Drop-in replacement for
    ``tensorrt_edgellm.quantization.quantize._load_model`` with the same
    return shape ``(model, tokenizer, processor)``. Phi-4MM and Qwen3-ASR use
    dedicated loaders upstream that this project does not yet stream (see
    README) and raise ``NotImplementedError`` here rather than silently
    loading fully onto one GPU.
    """
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_dir,
                                              trust_remote_code=True)
    try:
        processor = AutoProcessor.from_pretrained(model_dir,
                                                  trust_remote_code=True,
                                                  min_pixels=128 * 28 * 28,
                                                  max_pixels=2048 * 32 * 32)
    except Exception:
        processor = None

    if _is_phi4mm_model(model_dir):
        raise NotImplementedError(
            "Phi-4MM loads via tensorrt_edgellm.lora.load_phi4mm_model, a "
            "dedicated loader this project does not yet stream. See "
            "README.md 'Out of scope'.")
    if is_qwen3_asr_model(model_dir):
        raise NotImplementedError(
            "Qwen3-ASR loads via a dedicated joint audio+text model loader "
            "this project does not yet stream. See README.md 'Out of scope'.")

    if _is_nemotron_h_model(model_dir):
        from tensorrt_edgellm.quantization.nemotron_h_patch import \
            apply as _apply_nemotron_h_patch
        _apply_nemotron_h_patch()

    device_map, max_memory = _resolve_device_map_and_memory(
        device, max_gpu_memory_gb, max_cpu_memory_gb, use_seq_device_map)

    # Same most-specific-first factory order as the original _load_model:
    # ImageTextToText before CausalLM, because Qwen3.5/Qwen3-VL register
    # both architectures for the same checkpoint and CausalLM would silently
    # drop the visual tower.
    factories = [
        f for f in (AutoModelForTextToWaveform, AutoModelForImageTextToText,
                    AutoModelForCausalLM, AutoModel) if f is not None
    ]
    last_err: Optional[Exception] = None
    model = None
    for factory in factories:
        try:
            model = factory.from_pretrained(
                model_dir,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
            )
            gc.collect()
            break
        except (ValueError, KeyError) as e:
            last_err = e
    if model is None:
        raise RuntimeError(
            f"Could not load {model_dir} via any AutoModel factory"
        ) from last_err

    # The original _load_model does a blanket model.to(torch_dtype) here as
    # a belt-and-suspenders cast for factories that don't fully honor
    # torch_dtype. Skipped here: calling .to() on a model with accelerate
    # offload hooks attached bypasses their weights_map bookkeeping and can
    # desync offloaded parameters. torch_dtype is passed to from_pretrained
    # above instead, which loads weights in the target dtype directly.
    # VERIFY on real hardware: if some architecture still ends up with a
    # stray fp32 submodule under streaming load, that's the tradeoff to
    # revisit.

    # modelopt export_hf_checkpoint crashes when architectures is None
    # (e.g. Qwen3.5 resolves to text_config with architectures=None).
    if getattr(model.config, "architectures", None) is None:
        model.config.architectures = [type(model).__name__]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, processor
