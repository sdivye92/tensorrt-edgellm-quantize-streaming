"""Register VLM decoder-layer discovery for ModelOpt's layerwise calibration.

ModelOpt 0.45.0's ``LayerActivationCollector.get_decoder_layers`` (used by
``algorithm.layerwise``) ships two built-in discoverers
(`modelopt/torch/quantization/plugins/huggingface.py`):

- ``get_nemotron_h_decoder_layers`` -- ``model.backbone.layers``
- ``get_homogeneous_hf_decoder_layers`` -- ``model.model.layers``

Composite VLM wrappers produced by recent ``transformers`` refactors (Qwen2-VL,
Qwen2.5-VL, Qwen3-VL, and other ``*ForConditionalGeneration`` classes that
follow the same convention) nest the text decoder one level deeper, under a
``language_model`` submodule -- e.g. Qwen3-VL's structure is
``Qwen3VLForConditionalGeneration.model.language_model.layers``
(``Qwen3VLModel.language_model`` is a ``Qwen3VLTextModel``, confirmed against
the installed transformers==5.14.1 source). Neither built-in discoverer finds
that path, so ``mtq.quantize(..., algorithm={"layerwise": True})`` raises
``ValueError: Could not find transformer layers in model`` for these models
even though they're otherwise ordinary decoder-only-backbone VLMs.

This module registers an additional discoverer for that convention. Verified
against Cosmos-Reason2-2B (``Qwen3VLForConditionalGeneration``).
"""

import functools
from contextlib import contextmanager

import torch.nn as nn

_REGISTERED = False


def _find_language_model_layers(model: nn.Module):
    """Look for ``<model>.language_model.layers`` at the top level or one hop under ``.model``.

    Mirrors ModelOpt's own ``get_homogeneous_hf_decoder_layers`` (which checks
    ``model.model.layers``) but adds the extra ``language_model`` hop recent
    VLM wrappers introduce.
    """
    for candidate in (model, getattr(model, "model", None)):
        if candidate is None:
            continue
        language_model = getattr(candidate, "language_model", None)
        if language_model is None:
            continue
        layers = getattr(language_model, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    return None


def _is_vlm_language_model(model: nn.Module) -> bool:
    return _find_language_model_layers(model) is not None


def register_vlm_layerwise_support() -> None:
    """Register the ``language_model.layers`` discoverer with ModelOpt, once.

    Safe to call multiple times/from multiple entry points (cli.py,
    quantize.py, __init__.py all call it): guarded by a module-level flag
    since re-registering the same predicate/discoverer pair would otherwise
    just be a harmless no-op anyway (ModelOpt de-dupes identical tuples), but
    the flag avoids re-defining and re-appending on repeated imports in
    long-lived processes (e.g. a notebook).
    """
    global _REGISTERED
    if _REGISTERED:
        return
    from modelopt.torch.quantization.utils.layerwise_calib import \
        LayerActivationCollector

    LayerActivationCollector.register_decoder_layer_support(
        _is_vlm_language_model, _find_language_model_layers)
    _REGISTERED = True


# ---------------------------------------------------------------------------
# Calibration-activation CPU offload
# ---------------------------------------------------------------------------
#
# ModelOpt's layerwise collector retains every calibration sample's decoder-
# layer input on whichever device it was captured from -- i.e. the GPU
# (``_LayerCalibState.collected_inputs.append((args, kwargs))``,
# `layerwise_calib.py:235`). That set is held for the whole of one layer's
# calibration, and during the hand-off forward that produces layer *i+1*'s
# inputs both layers' sets are live at once, so peak GPU activation memory is
# roughly ``2 * num_samples`` layer inputs. For Cosmos-Reason2-2B at
# ``--num_samples 512`` (seq 512, hidden 2048, fp16) that is ~1.5 GB per layer
# and ~3 GB at the hand-off -- none of it covered by ``--max_gpu_memory_gb``,
# which only bounds accelerate's *weight* placement (see loader.py).
#
# This patch keeps that set in CPU RAM and pages one entry back to the GPU for
# the duration of the single forward that consumes it, taking peak activation
# residency from ``num_samples`` entries to one. ModelOpt already round-trips
# these exact objects through CPU for layerwise checkpointing
# (``_move_to_device(next_layer_inputs, _cpu)``, `layerwise_calib.py:680`,
# reloaded with ``map_location=layer_device``), so CPU is a device they are
# already known to survive.
#
# Two seams, both on ``LayerActivationCollector``:
#
# 1. ``_set_layer_states`` installs the capture-side container. Overriding the
#    container's ``append`` (rather than the patched forward that calls it) is
#    what avoids the GPU accumulation at the source: the closure that appends
#    is defined inside ``_patch_all_layers`` and is not separately patchable.
# 2. ``_patch_all_layers`` wraps each layer's ``_original_forward``. This is
#    the one point every replay funnels through, which is why hydration lives
#    here instead of at the two call sites: run-mode replay pops from the
#    deque *inside* the patched forward
#    (``info.cached_inputs.popleft()``, `layerwise_calib.py:229`) and so is
#    unreachable from outside it, while calibration replay enters the same
#    patched forward in "original" mode from ``layerwise_calibrate``'s
#    ``_layer_forward_loop`` (`model_calib.py:1752`). Wrapping the inner
#    ``_original_forward`` attribute catches both.
# 3. ``_cleanup_layers`` strips that wrapper again on teardown. This is not
#    optional: ``unpatch_forward_method`` restores by assigning
#    ``module.forward = module._original_forward``, so a wrapper left in that
#    attribute would be promoted to the layer's permanent forward rather than
#    removed.

_CPU = None  # set lazily; torch.device("cpu")


def _make_offload_list(move_to_device):
    """Build the capture-side list subclass, bound to ModelOpt's ``_move_to_device``."""

    class _CpuOffloadedInputs(list):
        """``list`` that copies each captured ``(args, kwargs)`` to CPU on append.

        Only ``append`` is overridden: that is the sole way the collector adds
        to this container, and ``list``'s own ``__init__``/``extend`` are left
        alone so ``list(...)``/``deque(...)`` copies made downstream behave
        like ordinary lists holding the already-CPU entries.
        """

        def append(self, item):
            super().append(move_to_device(item, _CPU))

    return _CpuOffloadedInputs


@contextmanager
def calib_activation_offload():
    """Keep layerwise calibration activations in CPU RAM instead of on the GPU.

    Scoped to one ``quantize_and_export`` call: both patched methods are
    restored in ``finally`` so an exception mid-calibration cannot leave
    ModelOpt's class mutated for the rest of the process.
    """
    global _CPU
    import torch

    from modelopt.torch.quantization.utils import layerwise_calib as _lc
    from modelopt.torch.utils.network import get_module_device

    _CPU = torch.device("cpu")
    move_to_device = _lc._move_to_device
    offload_list_cls = _make_offload_list(move_to_device)

    collector = _lc.LayerActivationCollector
    original_set_layer_states = collector._set_layer_states
    original_patch_all_layers = collector._patch_all_layers
    original_cleanup_layers = collector._cleanup_layers

    @functools.wraps(original_set_layer_states)
    def offloading_set_layer_states(self, layer_idx: int):
        original_set_layer_states(self, layer_idx)
        # The original assigns a plain ``[]`` to the newly-capturing layer
        # (`layerwise_calib.py:323`); swap in the offloading container before
        # any forward reaches it. It is always empty at this point, so nothing
        # needs migrating.
        cur = self._decoder_layers[layer_idx]._layerwise_calib
        cur.collected_inputs = offload_list_cls(cur.collected_inputs)

    @functools.wraps(original_patch_all_layers)
    def hydrating_patch_all_layers(self, decoder_layers=None):
        original_patch_all_layers(self, decoder_layers=decoder_layers)
        for layer in self._decoder_layers:
            layer._original_forward = _hydrating_forward(
                layer, layer._original_forward, move_to_device,
                get_module_device)

    @functools.wraps(original_cleanup_layers)
    def unhydrating_cleanup_layers(self):
        # Strip the wrapper *before* ModelOpt's restore, which copies
        # _original_forward back onto .forward verbatim.
        if self._decoder_layers is not None:
            for layer in self._decoder_layers:
                _strip_hydrating_forward(layer)
        original_cleanup_layers(self)

    collector._set_layer_states = offloading_set_layer_states
    collector._patch_all_layers = hydrating_patch_all_layers
    collector._cleanup_layers = unhydrating_cleanup_layers
    try:
        yield
    finally:
        collector._set_layer_states = original_set_layer_states
        collector._patch_all_layers = original_patch_all_layers
        collector._cleanup_layers = original_cleanup_layers


def _hydrating_forward(layer, inner, move_to_device, get_module_device):
    """Wrap *inner* so CPU-resident replay inputs are paged to *layer*'s device.

    ``get_module_device`` is resolved per call rather than captured once
    because the layer is materialized by ``persistent_materialization`` around
    calibration and may be on ``meta`` outside it; inside the forward its
    parameters are always real. Moving an already-on-device tensor is a no-op,
    so the capture and plain "original" paths (which arrive with live GPU
    tensors) are unaffected apart from one cheap tree walk.
    """

    @functools.wraps(inner)
    def forward(*args, **kwargs):
        device = get_module_device(layer)
        args, kwargs = move_to_device((args, kwargs), device)
        return inner(*args, **kwargs)

    # Marks this wrapper for _strip_hydrating_forward. functools.wraps already
    # exposes the wrapped callable as ``forward.__wrapped__``, but ``wraps``
    # copies attributes from *inner* too, so an explicit, uniquely-named flag
    # is what makes the check unambiguous.
    forward._sq_hydrating = True
    return forward


def _strip_hydrating_forward(layer) -> None:
    """Put the true inner forward back on ``layer._original_forward``.

    Required because ModelOpt's teardown does
    ``module.forward = module._original_forward`` (``unpatch_forward_method``,
    `network.py`), i.e. it restores whatever that attribute currently holds --
    so leaving the wrapper there would rebind it as the layer's *permanent*
    forward after calibration instead of removing it.
    """
    wrapper = getattr(layer, "_original_forward", None)
    if getattr(wrapper, "_sq_hydrating", False):
        layer._original_forward = wrapper.__wrapped__
