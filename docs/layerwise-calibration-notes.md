# Layerwise calibration: environment fixes, the log format, and activation offload

Working notes from bringing `tensorrt-edgellm-quantize-streaming --layerwise` up on a
single 8 GB consumer GPU. Three separate topics, in the order they were hit:

1. [Two environment fixes](#1-two-environment-fixes) — the Triton JIT failure and the
   silent 22x FP8 slowdown.
2. [Reading the calibration log](#2-reading-the-calibration-log) — what
   `skip / run / capture` actually means.
3. [GPU memory and `--offload_calib_activations`](#3-gpu-memory-and---offload_calib_activations)
   — why `--max_gpu_memory_gb` did not bound peak usage, and the patch that addresses it.

Reference environment for everything below:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop, 7.96 GB, compute capability 12.0 (`sm_120`) |
| torch | 2.13.0+cu130 (`torch.version.cuda == "13.0"`) |
| Python | 3.12.3, venv on `/usr/bin/python3.12` |
| transformers | 5.14.1 |
| Model | Cosmos-Reason2-2B (`Qwen3VLForConditionalGeneration`), 28 text decoder layers, hidden 2048, head_dim 128 |

---

## 1. Two environment fixes

### 1.1 `fatal error: Python.h: No such file or directory`

**Symptom.** Calibration died on the very first forward pass:

```
Calibrating:   0%|          | 0/512 [00:00<?, ?it/s]
/tmp/tmp5mabx13j/cuda_utils.c:9:10: fatal error: Python.h: No such file or directory
    9 | #include <Python.h>
subprocess.CalledProcessError: Command '['/usr/bin/gcc', '/tmp/.../cuda_utils.c', ...]'
    returned non-zero exit status 1.
```

**Cause.** Nothing to do with quantization. The traceback bottoms out in Triton's
first-use JIT of its own `cuda_utils` shim — `triton/backends/nvidia/driver.py`
compiles a small C extension with `gcc` the first time a Triton kernel runs, and that
needs the CPython development headers. `/usr/include/python3.12/` did not exist at all;
the only headers present belonged to an unrelated miniconda 3.14 install that `gcc` was
never pointed at.

The path into Triton is worth noting, because it makes the failure unavoidable rather
than `--layerwise`-specific: Qwen3-VL's RoPE

```python
# transformers/models/qwen3_vl/modeling_qwen3_vl.py:367
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
```

dispatches through `torch._native.ops.bmm_outer_product`, which is a Triton kernel. So
it fires on the first calibration sample of any run.

**Fix.**

```sh
sudo apt install -y python3.12-dev     # 3.12.3-1ubuntu0.17
```

`gcc` was already present, so that was the only missing piece.

**Verification.**

```
$ ./venv/bin/python -c "from triton.runtime import driver; \
    print(type(driver.active).__name__, driver.active.get_current_device())"
CudaDriver 0
```

---

### 1.2 The silent 22x FP8 slowdown (`CUDA_HOME` / no `nvcc`)

**Symptom.** Calibration ran, but layer 1 went at 86 it/s and layer 2 at 3.92 it/s,
with a warning in between:

```
Loading extension modelopt_cuda_ext_fp8...
modelopt/torch/utils/cpp_extension.py:97: UserWarning: CUDA_HOME environment variable
  is not set. Please set it to your CUDA install root.
CUDA extension for FP8 quantization could not be built and loaded, FP8 simulated
  quantization will not be available.
```

This one is dangerous precisely because it is only a warning: the run completes and
produces a correct checkpoint, just ~22x slower per layer. At 28 layers that is the
difference between minutes and an hour.

**Cause.** No CUDA toolkit on the machine at all — no `/usr/local/cuda`, no `nvcc`.
The venv had only the CUDA *runtime* wheels (`nvidia-cuda-runtime`, `nvidia-cublas`,
etc.), which ship `.so` libraries but no compiler. ModelOpt needs to compile
`tensor_quant_gpu_fp8.cu` at first use, so with no `nvcc` it falls back to a pure-PyTorch
FP8 simulation.

**Fix.** Four wheels plus a symlink. They had to be discovered one at a time, since each
fixed one error and exposed the next:

```sh
./venv/bin/pip install \
    "nvidia-cuda-nvcc==13.0.88" \
    "nvidia-nvvm==13.0.88" \
    "nvidia-cuda-crt==13.0.88" \
    "nvidia-cuda-cccl==13.0.85"

# the wheel ships only the versioned library, but -lcudart wants the bare name
cd venv/lib/python3.12/site-packages/nvidia/cu13/lib
ln -sfn libcudart.so.13 libcudart.so
```

The error chain, for anyone retracing it:

| Error | Cause | Fix |
|---|---|---|
| `CUDA_HOME ... is not set` | no toolkit at all | install `nvidia-cuda-nvcc` |
| `ptxas fatal: Unsupported .version 9.4; current version is '9.0'` | pip resolved `nvidia-nvvm` to 13.4.92, which emits PTX 9.4, while `ptxas` from nvcc 13.0.88 caps at 9.0 | pin `nvidia-nvvm` and `nvidia-cuda-crt` to 13.0.88 |
| `cuda_fp16.h:4492: fatal error: nv/target: No such file` | libcu++ / CCCL headers are a separate wheel | install `nvidia-cuda-cccl` |
| `Ninja is required to load C++ extensions` | `ninja` was installed but `venv/bin` was not on `PATH` | put `venv/bin` on `PATH` |
| `/usr/bin/ld: cannot find -lcudart` | wheel ships `libcudart.so.13` only | symlink `libcudart.so` |

**Version pinning is the subtle part.** `nvidia-cuda-nvcc==13.0.88` matches
`torch.version.cuda == "13.0"`, but pip pulls its `nvidia-nvvm` / `nvidia-cuda-crt`
dependencies at *latest*, which silently mixes a 13.4 front end with a 13.0 `ptxas`.
Pin all three together.

**Required at runtime.** The build is cached at
`~/.cache/torch_extensions/py312_cu130/modelopt_cuda_ext_fp8/`, so it compiles once —
but ModelOpt checks `CUDA_HOME` before loading, so the environment still has to be set
on every run:

```sh
export CUDA_HOME=$PWD/venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$PWD/venv/bin:$CUDA_HOME/bin:$PATH
```

**Verification.**

```
Loading extension modelopt_cuda_ext_fp8...
Loaded extension modelopt_cuda_ext_fp8 in 9.5 seconds
FP8 EXT: <module 'modelopt_cuda_ext_fp8' from
  '/home/sdivye/.cache/torch_extensions/py312_cu130/modelopt_cuda_ext_fp8/modelopt_cuda_ext_fp8.so'>
```

Compiled with `-gencode=arch=compute_120,code=sm_120`, matching the RTX 5060.

> If you ever see per-layer calibration inexplicably drop from ~90 it/s to ~4 it/s,
> check for this warning first — it is the same fallback re-appearing because
> `CUDA_HOME` was not exported.

---

## 2. Reading the calibration log

A line like:

```
Calibrating layer 27/28 | skip: 25 | run: [26] | capture: [27]
```

decodes as:

```
Calibrating layer 27/28 | skip: 25 | run: [26] | capture: [27]
                    ^^        ^^        ^^^^         ^^^^
             now on layer 27   |     layer 26      layer 27
                        25 layers (1-25)
```

Indices are **1-based** (`groups[mode].append(i + 1)`,
`layerwise_calib.py:333`). `skip` prints as a **count** while `run` and `capture`
print as explicit lists — `layerwise_calib.py:340`:

```python
parts.append(f"{mode}: {len(ids)}" if mode == "skip" else f"{mode}: {ids}")
```

A full model forward still runs for every calibration sample, but only **one layer in it
does real work**. The three modes come from the patched forward
(`layerwise_calib.py:213-237`):

### `skip` — layers 1-25: already calibrated, contribute nothing

Each has been *replaced in the `ModuleList`* by a `_SkipLayer`, whose forward returns
zeros of the recorded shape/dtype/device:

```python
def forward(self, *args, **kwargs):
    return LayerActivationCollector._zeros_from_meta(
        self._original._layerwise_calib.output_meta
    )
```

`_SkipLayer` holds **no parameters** — it stashes the real layer via
`object.__setattr__` specifically to avoid registering it as a submodule — so accelerate
has nothing left to stream for finished layers.

### `run` — layer 26: just calibrated, and now feeds layer 27

It **ignores** the (zero) tensors passed to it and pops a real activation from its own
cache instead (`layerwise_calib.py:229`):

```python
real_args, real_kwargs = info.cached_inputs.popleft()
output = self._original_forward(*real_args, **real_kwargs)
info.output_meta = LayerActivationCollector._extract_output_meta(output)
```

This is the only layer doing genuine math, now with its newly quantized weights.

### `capture` — layer 27: records its inputs and aborts the pass

```python
info.collected_inputs.append((args, kwargs))
raise _EarlyStopForwardError()
```

Layer 28 and the `lm_head` never execute for this sample.

### Why the zeros don't corrupt the result

The skipped layers really do emit garbage — but it is discarded, because layer 26 never
reads it. Real activations advance **one hop per step, through the cache**, rather than
being recomputed from the embeddings each time:

```
step 26:  layer 26 captures real inputs  ──┐
          calibrate layer 26               │ cached
step 27:  layer 26 replays those ──────────┘→ layer 27 captures its real inputs
          calibrate layer 27
```

Each layer is therefore "real" for exactly two steps — once as `capture`/calibrate, once
as `run` — then becomes a zero-returning dummy. That is why `skip` is 25 and not 26 at
this step: layer 26 is still needed to feed layer 27, and is swapped to a dummy
(`_swap_to_dummy`) only on the next step.

### What it buys, and what it costs

- Only one layer's weights need materializing per step, so `--max_gpu_memory_gb` can sit
  far below the model's full size.
- Finished layers drop out of accelerate's GPU/CPU/disk shuffle entirely, having no
  parameters.
- **Cost:** one full forward pass per layer — 28 passes over the calibration set instead
  of one. This is why fix 1.2 matters so much: the per-sample cost is paid 28 times.

**Healthy signature:** `skip` climbing steadily while `run` and `capture` each advance by
one per step. If `run` stalls at the same index across steps, the state machine is not
advancing.

---

## 3. GPU memory and `--offload_calib_activations`

### 3.1 Why `--max_gpu_memory_gb 4` still reached 7880 MiB

`--max_gpu_memory_gb` bounds **weight placement only**. It goes into accelerate's
`max_memory` (`streaming_quant/loader.py:56-57`), which governs `from_pretrained`
placement and nothing else. Calibration activations sit entirely outside that budget.

The specific consumer is in ModelOpt, not this project. `layerwise_calib.py:235`:

```python
if info.mode == "capture":
    info.collected_inputs.append((args, kwargs))   # tensors are on CUDA
    raise _EarlyStopForwardError()
```

Every sample's layer input is retained **on whichever device it was captured from** —
the GPU — for the whole of that layer's calibration.

Per sample, for this model at seq 512, fp16:

| Tensor | Size |
|---|---|
| hidden state `[1, 512, 2048]` | 2.0 MB |
| mRoPE `position_embeddings` (cos + sin, 3 sections x 128) | ~0.8 MB |

At `--num_samples 512` that is **~1.5 GB per layer**. And the peak is roughly
**2x** that, because during the hand-off forward both layers' sets are live at once:
`_set_layer_states` moves layer *i*'s set into the replay queue
(`layerwise_calib.py:318`) while layer *i+1* accumulates its own, and the
`layer_inputs` list in `layerwise_calibrate`'s loop keeps the former alive until
`del layer_inputs`.

So: ~4 GB of weights + ~3 GB of activations + allocator reserve ≈ the observed
7880 MiB. Note also that `nvidia-smi` reports *reserved* memory, not live — the caching
allocator holds freed blocks — so it overstates what is actually in use.

### 3.2 The idea, by example

Walk one layer with `--num_samples 512`:

**Phase 1, capture.** 512 forwards; each appends a live CUDA `(args, kwargs)` to
`collected_inputs`. ~1.5 GB now resident, and it stays resident for all of phase 2.

**Phase 2, calibrate.** `layerwise_calibrate` replays that list through the layer so the
quantizers observe real activations (`model_calib.py:1752`):

```python
def _layer_forward_loop(m, _inputs=layer_inputs):
    for args, kwargs_input in _inputs:
        m(*args, **kwargs_input)
```

**Phase 3, hand off.** The set is re-containered into a replay queue and the next layer
starts accumulating its own 512. This is the ~3 GB peak.

The patch changes one thing: **keep those 512 entries in CPU RAM, and move one back to
the GPU only for the single forward that consumes it.**

```python
# capture: store on CPU instead of GPU
info.collected_inputs.append(_move_to_device((args, kwargs), torch.device("cpu")))

# replay: hydrate one entry, use it, drop it
args, kwargs = _move_to_device((args, kwargs), layer_device)
return inner(*args, **kwargs)
```

GPU residency for captured activations goes from `num_samples` entries to **one**.

This is not a novel treatment of these objects: ModelOpt already round-trips the exact
same structures through CPU for layerwise checkpointing —
`_move_to_device(next_layer_inputs, _cpu)` at `layerwise_calib.py:680`, reloaded with
`map_location=layer_device`. CPU is a device they are already known to survive.

### 3.3 Implementation: three seams

Enabled by `--offload_calib_activations` (requires `--layerwise`). Lives in
`streaming_quant/modelopt_patches.py` as the `calib_activation_offload()` context
manager, scoped to one `quantize_and_export` call and restored in `finally`. When the
flag is off, nothing is patched — the stock path is untouched.

**Seam 1 — `_set_layer_states`: the capture-side container.**

```python
cur.collected_inputs = offload_list_cls(cur.collected_inputs)
```

where the container is a `list` subclass overriding only `append`:

```python
class _CpuOffloadedInputs(list):
    def append(self, item):
        super().append(move_to_device(item, _CPU))
```

Overriding the *container* rather than the forward that calls it is what avoids the GPU
accumulation at its source: the appending closure is defined inside
`_patch_all_layers` and is not separately patchable. `list.__init__`/`extend` are left
alone so downstream `list(...)` / `deque(...)` copies behave normally.

**Seam 2 — `_patch_all_layers`: hydrate on the way in.**

```python
layer._original_forward = _hydrating_forward(
    layer, layer._original_forward, move_to_device, get_module_device)
```

The inner `_original_forward` attribute is the single point **both** replay paths funnel
through, which is why hydration lives there rather than at the two call sites:

- calibration replay enters the patched forward in `"original"` mode from
  `_layer_forward_loop`, which then calls `self._original_forward(...)`;
- `"run"` mode pops from its queue **inside** the patched forward
  (`layerwise_calib.py:229`) and so is unreachable from outside it.

The device is resolved per call via `get_module_device(layer)`, not captured once,
because the layer may be on `meta` outside `persistent_materialization`. Moving an
already-on-device tensor is a no-op, so capture and plain `"original"` traffic are
unaffected beyond one cheap tree walk.

**Seam 3 — `_cleanup_layers`: strip the wrapper again.**

This one is not optional, and it was a real bug in the first draft. ModelOpt's teardown
(`modelopt/torch/utils/network.py`) restores by assignment:

```python
def unpatch_forward_method(module, orig_forward_cache_name):
    setattr(module, "forward", getattr(module, orig_forward_cache_name))
    delattr(module, orig_forward_cache_name)
```

It restores **whatever `_original_forward` currently holds**. Since seam 2 overwrote that
attribute, teardown would have *promoted* the hydrating wrapper to each layer's permanent
`forward` instead of removing it — leaving a device-move tree walk on every forward for
the rest of the process, export included. So the wrapper is tagged and stripped before
ModelOpt's restore runs:

```python
wrapper = getattr(layer, "_original_forward", None)
if getattr(wrapper, "_sq_hydrating", False):
    layer._original_forward = wrapper.__wrapped__
```

### 3.4 Measured results

Cosmos-Reason2-2B, FP8, `--max_gpu_memory_gb 4`, `--layerwise`, seq 512,
`--num_samples 128`. Both runs completed all 28 layers and exported.

| | peak `torch.cuda.max_memory_allocated` |
|---|---|
| offload off | **5.682 GB** |
| offload on | **5.123 GB** |

**Numerics are unchanged.** Comparing the two exported checkpoints:

```
same key set: True | n = 1018
scale tensors: 392
tensors differing: 0 | max abs diff: 0.0
hf_quant_config.json identical
```

Bit-identical, which is the expected result: the patch only relocates tensors.

Teardown and capture behaviour, unit-tested against a toy 2-layer model:

```
wrapper installed: True
offload container: _CpuOffloadedInputs
stored devices: cpu cpu | non-tensor preserved: True
wrapper stripped: True
no _original_forward left: True
model still runs: (2, 4)
```

### 3.5 Honest limits

- **The saving scales with `--num_samples`.** 0.56 GB at 128 samples; the mechanism is
  linear, so 512 should recover roughly 2.2 GB. **That figure is extrapolated, not
  measured.**
- **This flag alone will not bring peak usage down to `--max_gpu_memory_gb`.** 5.1 GB
  remained with activations offloaded, so the peak is dominated by weights plus
  per-forward transients — the untied 151k-vocab `embed_tokens` / `lm_head` pair is
  ~620 MB each, and logits add a few hundred MB.
- **Lowering `--num_samples` is the cheaper fix** when calibration quality allows it,
  since that cost is linear in sample count too. Reach for this flag when you need to
  keep the sample count and `--layerwise` still OOMs.
- Cost paid: host PCIe traffic, roughly `3 MB x num_samples x 2 forwards x 28 layers`.

### 3.6 Useful knobs, no code change

```sh
--num_samples 128                                   # linear cut in captured activations
--max_gpu_memory_gb 3                               # leave headroom above the weights
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # reduce fragmentation
--layerwise_checkpoint_dir DIR                      # resume instead of restarting
```
