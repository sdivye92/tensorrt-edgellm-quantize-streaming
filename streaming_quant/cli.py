"""CLI for streaming/offload-aware quantization.

Mirrors ``tensorrt-edgellm-quantize llm``'s flag surface exactly (see
``tensorrt_edgellm/scripts/quantize.py``) plus streaming-specific flags. Only
the ``llm`` path is supported -- ``draft`` (Eagle3/DFlash speculative
draft) models are out of scope for this project; use the stock
``tensorrt-edgellm-quantize draft`` for those.

Usage::

    tensorrt-edgellm-quantize-streaming \\
        --model_dir /path/to/vlm \\
        --output_dir /path/to/output \\
        --quantization nvfp4 \\
        --visual_quantization fp8 \\
        --offload_folder /path/to/scratch/offload \\
        --max_gpu_memory_gb 6 \\
        --layerwise
"""

import argparse

from tensorrt_edgellm.quantization.datasets import (DEFAULT_AUDIO_DATASET,
                                                     DEFAULT_IMAGE_DATASET,
                                                     DEFAULT_TEXT_DATASET,
                                                     available_datasets)

from .quantize import quantize_and_export_streaming


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Streaming/offload-aware TensorRT-Edge-LLM quantization: "
            "quantizes models larger than available GPU VRAM by loading "
            "with accelerate CPU/disk offload and calibrating one decoder "
            "layer at a time (ModelOpt's layerwise calibration)."))
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"],
        help="Backbone quantization method.")
    parser.add_argument("--lm_head_quantization",
                        default=None,
                        choices=["fp8", "int4_awq", "nvfp4", "mxfp8"],
                        help="LM-head quantization method.")
    parser.add_argument(
        "--visual_quantization",
        default=None,
        choices=["fp8"],
        help=("Quantize the visual tower (visual / vision_tower / "
              "multi_modal_projector). Only fp8 is exposed today. When "
              "unset the visual tower stays at fp16."))
    parser.add_argument(
        "--visual_mha_quantization",
        default=None,
        choices=["fp8"],
        help="Run the visual attention (Q*K^T and P*V matmuls) in FP8.")
    parser.add_argument(
        "--audio_quantization",
        default=None,
        choices=["fp8"],
        help=("Quantize the audio tower (audio_tower / audio_embed). When "
              "unset the audio tower stays at fp16. Not supported for "
              "Qwen3-ASR under this project -- see README.md."))
    parser.add_argument(
        "--cp_quantization",
        default=None,
        choices=["fp8"],
        help=("Quantize the Talker CodePredictor of Qwen3-Omni/Qwen3-TTS. "
              "Not supported for Qwen3-Omni under this project -- see "
              "README.md."))
    parser.add_argument("--kv_cache_quantization",
                        default=None,
                        choices=["fp8"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--text_dataset",
        default=DEFAULT_TEXT_DATASET,
        help=("Registered text calibration dataset name. "
              f"Default: {DEFAULT_TEXT_DATASET}. Available: "
              f"{', '.join(available_datasets('text'))}."))
    parser.add_argument(
        "--image_dataset",
        default=DEFAULT_IMAGE_DATASET,
        help=("Registered image calibration dataset name (used with "
              "--visual_quantization). "
              f"Default: {DEFAULT_IMAGE_DATASET}. Available: "
              f"{', '.join(available_datasets('image'))}."))
    parser.add_argument(
        "--audio_dataset",
        default=DEFAULT_AUDIO_DATASET,
        help=("Registered audio calibration dataset name. "
              f"Default: {DEFAULT_AUDIO_DATASET}. Available: "
              f"{', '.join(available_datasets('audio'))}."))
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument(
        "--mtp_draft_dir",
        default=None,
        help=("Optional separate checkpoint providing MTP draft weights. "
              "Defaults to --model_dir."))
    parser.add_argument(
        "--fuse_gdn_qkvzba_scales",
        action="store_true",
        help=("NVFP4 hybrid-GDN models only -- see "
              "tensorrt-edgellm-quantize's flag of the same name."))

    streaming = parser.add_argument_group("streaming/offload")
    streaming.add_argument(
        "--offload_folder",
        default=None,
        help=("Disk-offload directory. Weights that don't fit the GPU+CPU "
              "memory budgets below are streamed from here during "
              "calibration. Required for models that don't fit in system "
              "RAM at all; optional (CPU-only offload) otherwise."))
    streaming.add_argument(
        "--max_gpu_memory_gb",
        type=float,
        default=None,
        help="Per-GPU memory budget for model weights during quantization.")
    streaming.add_argument(
        "--max_cpu_memory_gb",
        type=float,
        default=None,
        help="CPU RAM budget for offloaded model weights.")
    streaming.add_argument(
        "--use_seq_device_map",
        action="store_true",
        help=("Use device_map='sequential' instead of 'auto'. Helpful when "
              "'auto' splits the model unevenly across multiple GPUs; "
              "irrelevant on a single-GPU machine."))
    streaming.add_argument(
        "--layerwise",
        action="store_true",
        help=("Calibrate one decoder layer at a time (ModelOpt's "
              "algorithm.layerwise) instead of the whole resident model at "
              "once. This is what keeps peak GPU memory near one layer's "
              "footprint instead of the whole model's."))
    streaming.add_argument(
        "--offload_calib_activations",
        action="store_true",
        help=("Keep per-layer calibration activations in CPU RAM instead of "
              "on the GPU, paging one batch back per forward. Requires "
              "--layerwise. ModelOpt retains all --num_samples layer inputs "
              "on the GPU during calibration, which --max_gpu_memory_gb does "
              "not bound (that budget covers weights only); this trades host "
              "PCIe traffic for that headroom. Use it when --layerwise still "
              "OOMs and you don't want to lower --num_samples."))
    streaming.add_argument(
        "--layerwise_checkpoint_dir",
        default=None,
        help=("Per-layer calibration checkpoint directory. Requires "
              "--layerwise. A run interrupted partway through resumes from "
              "the last completed layer instead of restarting."))
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.layerwise_checkpoint_dir and not args.layerwise:
        parser.error("--layerwise_checkpoint_dir requires --layerwise")
    if args.offload_calib_activations and not args.layerwise:
        parser.error("--offload_calib_activations requires --layerwise")

    quantize_and_export_streaming(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        mtp_draft_dir=args.mtp_draft_dir,
        quantization=args.quantization,
        lm_head_quantization=args.lm_head_quantization,
        visual_quantization=args.visual_quantization,
        visual_mha_quantization=args.visual_mha_quantization,
        audio_quantization=args.audio_quantization,
        cp_quantization=args.cp_quantization,
        kv_cache_quantization=args.kv_cache_quantization,
        dtype=args.dtype,
        device=args.device,
        text_dataset=args.text_dataset,
        image_dataset=args.image_dataset,
        audio_dataset=args.audio_dataset,
        num_samples=args.num_samples,
        fuse_gdn_qkvzba_scales=args.fuse_gdn_qkvzba_scales,
        offload_folder=args.offload_folder,
        max_gpu_memory_gb=args.max_gpu_memory_gb,
        max_cpu_memory_gb=args.max_cpu_memory_gb,
        use_seq_device_map=args.use_seq_device_map,
        layerwise=args.layerwise,
        layerwise_checkpoint_dir=args.layerwise_checkpoint_dir,
        offload_calib_activations=args.offload_calib_activations,
    )


if __name__ == "__main__":
    main()
