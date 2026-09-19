# streaming-quant

Quantize TensorRT-Edge-LLM-supported VLMs/LLMs on a GPU with far less VRAM
than the checkpoint's FP16 size, by loading with HuggingFace `accelerate`
CPU/disk offload and calibrating one decoder layer at a time (NVIDIA
ModelOpt's `layerwise` calibration).

## Why this works

TensorRT-Edge-LLM's `tensorrt-edgellm-quantize` loads the full FP16 model
onto one GPU before calibrating (`tensorrt_edgellm/quantization/quantize.py`'s
`_load_model` does `.from_pretrained(...).to(device)`), so a 17 GB checkpoint
needs a GPU with roughly 17 GB free, even though the quantized result is much
smaller.

`nvidia-modelopt==0.45.0` — the exact version Edge-LLM pins — has a
first-class `algorithm.layerwise` calibration mode
(`modelopt/torch/quantization/config.py`) that calibrates one transformer
decoder layer at a time, and is explicitly built to page each layer's real
weights in from wherever HuggingFace `accelerate`'s offload hooks parked them
(CPU RAM or disk), right before calibrating it
(`modelopt/torch/quantization/utils/core_utils.py`'s
`persistent_materialization`/`enable_weight_access_and_writeback`). NVIDIA's
own ModelOpt examples (`examples/hf_ptq/example_utils.py::get_model()`) load
models this way for exactly this reason.

So the only two things Edge-LLM's pipeline needs that it doesn't do today
are: (1) load the model with `device_map`/`max_memory`/`offload_folder`
instead of `.to(device)`, and (2) set `algorithm.layerwise = True` in the
quant config passed to `mtq.quantize`. Everything else — calibration
dataloaders, checkpoint export, repacking — is reused unmodified from
`tensorrt_edgellm`.

## Setup

This project imports `tensorrt_edgellm` as a library; install both into the
same virtual environment:

```bash
pip install -e '/home/sdivye/Documents/Projects/TensorRT-Edge-LLM[tools]'
pip install -e .   # this project
```

(`tensorrt-edgellm[tools]` already pins `torch==2.13.0`,
`transformers==5.14.1`, `nvidia-modelopt==0.45.0`; this project adds
`accelerate>=1.10.1` on top.)

## Usage

```bash
tensorrt-edgellm-quantize-streaming \
    --model_dir /home/sdivye/Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561 \
    --output_dir /path/to/output \
    --quantization nvfp4 \
    --visual_quantization fp8 \
    --offload_folder /path/to/scratch/offload \
    --max_gpu_memory_gb 6 \
    --layerwise
```

Every flag from `tensorrt-edgellm-quantize llm` is supported (see
`streaming_quant/cli.py` or `--help`), plus:

- `--offload_folder PATH` — disk-offload directory for weights exceeding the
  GPU+CPU budgets.
- `--max_gpu_memory_gb FLOAT` / `--max_cpu_memory_gb FLOAT` — memory budgets.
- `--use_seq_device_map` — `device_map="sequential"` instead of `"auto"`
  (multi-GPU only).
- `--layerwise` — calibrate one decoder layer at a time. This is what keeps
  peak GPU memory near one layer's footprint instead of the whole model's;
  without it, offload alone still avoids holding everything on the GPU at
  once, but `mtq.quantize`'s forward pass will still try to run activations
  through the *whole* model per calibration batch, materializing every layer
  it touches.
- `--offload_calib_activations` — keep the per-layer calibration activations
  in CPU RAM instead of on the GPU (requires `--layerwise`). ModelOpt's
  layerwise collector retains all `--num_samples` decoder-layer inputs on the
  GPU for the duration of a layer's calibration, and holds two layers' worth
  during the hand-off forward — memory that `--max_gpu_memory_gb` does *not*
  bound, since that budget only governs accelerate's weight placement. This
  flag pages one batch back to the GPU per forward instead, trading host PCIe
  traffic for that headroom.

  Measured on Cosmos-Reason2-2B (FP8, `--max_gpu_memory_gb 4`, seq 512,
  `--num_samples 128`): peak `torch.cuda.max_memory_allocated` falls from
  5.682 GB to 5.123 GB, i.e. ~0.56 GB, and the exported checkpoint is
  bit-identical (all 1018 tensors, 392 of them scales, max abs diff 0.0 — the
  offload only relocates tensors, it does not change numerics). The saving is
  linear in `--num_samples`, so `--num_samples 512` should recover closer to
  ~2.2 GB; that extrapolation has not been measured directly.

  Note the remaining peak is dominated by weights and per-forward transients,
  not by captured activations, so this flag on its own will not bring peak
  usage down to `--max_gpu_memory_gb`. Lowering `--num_samples` is the cheaper
  fix when calibration quality allows it; reach for this flag when you need to
  keep the sample count and `--layerwise` still OOMs.
- `--layerwise_checkpoint_dir PATH` — resumable per-layer calibration
  checkpoints (requires `--layerwise`).

Output is a standard Edge-LLM unified checkpoint
(`model.safetensors[.index.json]` + `hf_quant_config.json`), loadable by
`tensorrt_edgellm.checkpoint.loader`/`repacking` and the normal
`tensorrt-edgellm-build`/inference path exactly like a checkpoint produced by
stock `tensorrt-edgellm-quantize`.

## How it works

`streaming_quant/loader.py` replaces `_load_model`'s generic Auto* factory
branch with an accelerate `device_map`/`offload_folder` load.
`streaming_quant/quantize.py` monkeypatches `_load_model` and
`build_quant_config` inside `tensorrt_edgellm.quantization.quantize`'s module
namespace for the duration of one call, then invokes Edge-LLM's own
`quantize_and_export` unchanged — so every model-specific branch already in
Edge-LLM (visual/audio/CP calibration, MTP draft quantization,
per-architecture patches) keeps working exactly as it does upstream.

`streaming_quant/modelopt_patches.py` holds the two patches that target
ModelOpt rather than Edge-LLM: decoder-layer discovery for VLM wrappers that
nest the text decoder under `language_model`, and (under
`--offload_calib_activations`) the CPU offload of captured calibration
activations. The latter overrides `LayerActivationCollector._set_layer_states`
to install a list that copies each captured `(args, kwargs)` to CPU on append,
and `_patch_all_layers` to wrap each layer's inner `_original_forward` so
CPU-resident inputs are paged back to the layer's device for the one forward
that consumes them. Both replay paths — `layerwise_calibrate`'s
`_layer_forward_loop` and the collector's own "run" mode, which pops from its
queue *inside* the patched forward — funnel through that inner attribute,
which is why hydration lives there rather than at the call sites.

## Further reading

[`docs/layerwise-calibration-notes.md`](docs/layerwise-calibration-notes.md) covers, in
detail: the two environment fixes needed to get `--layerwise` running on a consumer GPU
(CPython headers for Triton's JIT, and the `nvcc` toolchain ModelOpt's FP8 kernel needs —
whose absence is a *warning*, not an error, and costs ~22x calibration speed); how to
read the `skip / run / capture` calibration log; and the design, failure modes and
measured results of `--offload_calib_activations`.

## Out of scope / known limitations

- **Qwen3-Omni, Qwen3-Omni-Next, Qwen3-ASR, Phi-4MM, NemotronH*:** these use
  dedicated loaders (`qwen3_omni.py`, `qwen3_asr_loader.py`,
  `tensorrt_edgellm.lora.load_phi4mm_model`) that bypass or extend
  `_load_model` and are not streamed by this project. Phi-4MM and Qwen3-ASR
  raise `NotImplementedError`; Qwen3-Omni/Omni-Next are dispatched by
  `quantize_and_export` to their own drivers before this project's patch
  applies at all, so they'll silently attempt a full-GPU-residency load.
  (\*NemotronH's own architecture patch is applied, but it still goes through
  the generic streaming-capable Auto* branch — only the three above are
  actually blocked.)
- **GPTQ** is not wired into Edge-LLM's CLI today (only AWQ for INT4) and is
  out of scope here too. NVFP4, FP8, and INT4 (AWQ) are all supported, since
  they're Edge-LLM's existing three backbone formats and all use ModelOpt
  calibration algorithms (`max`, AWQ) that support `layerwise=True`.
- **Not yet validated on real hardware.** This was built and grounded against
  ModelOpt 0.45.0's actual source and NVIDIA's own reference examples (not
  guesswork), but there was no GPU/`pip`-capable environment available while
  writing it. Before trusting output checkpoints: run the staged
  verification below.
- The blanket `model.to(torch_dtype)` cast Edge-LLM's original `_load_model`
  does after loading is skipped for the streaming path (see the comment in
  `loader.py`) because it can desync accelerate's offload bookkeeping;
  `torch_dtype` is passed to `from_pretrained` directly instead. Watch for
  any architecture that ends up with a stray non-target-dtype submodule.
- Some HF architectures (Nemotron-VL, T5, BART, per ModelOpt's own loader)
  disable `device_map="auto"` entirely for compatibility reasons. Cosmos-
  Reason2-2B (`Qwen3VLForConditionalGeneration`) isn't one of them, but an
  architecture that is would need the same kind of carve-out added to
  `loader.py`.

## Verification plan

1. **Static sanity** — a tiny HF model (e.g. `sshleifer/tiny-gpt2`), CPU-only,
   confirms the `layerwise`/config plumbing runs end-to-end without touching
   GPU memory or offload at all.
2. **Small model, real offload** — a 1–3B model with `--max_gpu_memory_gb`
   capped well below its FP16 size, `--layerwise --offload_folder`. Watch
   `nvidia-smi` to confirm peak GPU memory stays near the budget, and load
   the exported checkpoint via `tensorrt_edgellm.checkpoint.loader.load_weights`.
3. **Target model** — Cosmos-Reason2-2B at
   `/home/sdivye/Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561`
   (already downloaded) on the RTX 5060 (8 GB VRAM), for each of
   `--quantization nvfp4 / fp8 / int4_awq` (optionally with
   `--visual_quantization fp8`).
4. **Engine build** — run Edge-LLM's normal engine-build/inference path
   against the resulting checkpoint to confirm it's a drop-in replacement.
