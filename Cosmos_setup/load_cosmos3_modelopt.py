#!/usr/bin/env python
"""
Drop-in loader for the weight-only NVFP4 / FP8 (NVIDIA ModelOpt) Cosmos3-Super checkpoint
saved in the ROUND-TRIPPABLE format (modelopt_state.pth present -- see repackage_for_hf.py).

Current diffusers + accelerate + modelopt treat this path as experimental; each shim below
works around a specific, source-verified version-skew gap. None modify the checkpoint:

  1. enable_huggingface_checkpointing()
     Registers ModelOpt's HF handlers so `from_pretrained` restores the quantized module
     structure from modelopt_state.pth before weights load.

  2. set_module_tensor_to_device patch (parameter materialization)
     The modelopt_state restore runs inside diffusers' meta-device init, so each quantized
     weight is a QTensorWrapper whose storage is a META tensor carrying dequant metadata.
     `param.data = real` can't cross meta<->real, and accelerate's rebuild both rejects
     requires_grad and discards metadata. We REPLACE the parameter with a fresh wrapper
     around the loaded bytes + the existing metadata. Two extra duties here:
       - payload dtype restore: diffusers casts floating params to torch_dtype during load
         when no hf_quantizer is present (model_loading_utils.py -- their own TODO flags
         float8). FP8 payloads are floating and arrive cast to bf16; we cast back to the
         wrapper's payload dtype (exact: every e4m3fn value round-trips through bf16).
         NVFP4's uint8 payload is never cast, so this is a no-op there.
       - direct-to-GPU materialization: staging payloads on CPU would put the whole packed
         model in system RAM (FP8: ~64 GB on a 32 GB box -> OOM-killed; NVFP4 only survived
         because uncast tensors stay mmap-backed). Wrappers go straight to `materialize_device`.

  3. Post-restore quantizer re-disable
     modelopt_state replays the QUANT CONFIG, not imperative `.disable()` calls made after
     quantize. The NVFP4 build used NVFP4_DEFAULT_CFG (activation quantization ON in-config)
     with activations disabled imperatively -- so the restored model comes back with ~1806
     dynamic fake-quant activation quantizers active: ~10x slower, fatter, and quantizing
     activations the validated regime never quantized. We re-apply weight-only + spare
     disabling after load. (FP8's config had the disables baked in; this is then a no-op.)

  4. Full bf16 normalization (the validated serve regime)
     Cosmos3OmniTransformer pins time_embedder to fp32 via _keep_in_fp32_modules
     (transformer_cosmos3.py:297); the quantized model runs all-bf16. Cast floating buffers
     and non-wrapper floating params to bf16, retarget wrapper dequant dtype to bf16, and
     register bf16 input-cast pre-hooks on the time embedders.

  5. NVIDIAModelOptQuantizer.create_quantized_param patch
     Only relevant if a checkpoint carries an embedded quantization_config (hf_quantizer
     path): requires_grad only for float/complex tensors. Kept for robustness.

Usage (library):
    from load_cosmos3_modelopt import load_pipe
    pipe = load_pipe("YOUR_HF_USERNAME/Cosmos3-Super-nvfp4")   # or a local dir
    # NOTE: a bare pipe(prompt) call renders the pipeline DEFAULT: a 189-frame 720x1280
    # video (~8 s at 24 fps) -- not a still. For a single-image smoke test, be explicit:
    r = pipe("a red cube on a table", height=1024, width=1024, num_frames=1,
             num_inference_steps=50, guidance_scale=4.0)
    r.video[0].save("out.png")          # .video is the list of PIL frames; [0] is the image

Usage (interactive -- lands in a REPL with `pipe` loaded):
    CUDA_VISIBLE_DEVICES=0 python -i load_cosmos3_modelopt.py ./cosmos3-super-nvfp4-hf
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

SPARE_SUBSTRINGS = [
    "time_embedder", "proj_in", "proj_out", "lm_head", "embed", "norm", "audio_proj",
]


def _qtensor_wrapper_cls():
    try:
        from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper
        return QTensorWrapper
    except Exception:
        return None


def _patch_modelopt_quantizer() -> None:
    """diffusers pre-quantized hf_quantizer branch: only float/complex tensors may require grad."""
    import diffusers.quantizers.modelopt.modelopt_quantizer as moq

    if getattr(moq.NVIDIAModelOptQuantizer, "_cqp_patched", False):
        return
    _orig = moq.NVIDIAModelOptQuantizer.create_quantized_param

    def _cqp(self, model, param_value, param_name, target_device, *args, **kwargs):
        if self.pre_quantized:
            module, tname = moq.get_module_from_name(model, param_name)
            needs_grad = param_value.is_floating_point() or param_value.is_complex()
            module._parameters[tname] = torch.nn.Parameter(
                param_value.to(device=target_device), requires_grad=needs_grad
            )
            return
        return _orig(self, model, param_value, param_name, target_device, *args, **kwargs)

    moq.NVIDIAModelOptQuantizer.create_quantized_param = _cqp
    moq.NVIDIAModelOptQuantizer._cqp_patched = True


def _patch_qtensor_loading() -> None:
    """Materialize restored (meta) QTensorWrapper params: replace the parameter object with
    a fresh wrapper around the loaded bytes, restoring the payload dtype (undoes diffusers'
    float cast on FP8) and landing directly on the target device (keeps payloads out of RAM)."""
    import diffusers.models.model_loading_utils as mlu

    if getattr(mlu, "_qtw_inplace_patched", False):
        return
    QTensorWrapper = _qtensor_wrapper_cls()
    if QTensorWrapper is None:
        return  # different modelopt layout; nothing to patch

    _orig = mlu.set_module_tensor_to_device
    stats = {"materialized": 0, "target_device": None}

    def _patched(model, tensor_name, device, value=None, *args, **kwargs):
        if value is not None:
            module, leaf = model, tensor_name
            if "." in tensor_name:
                mod_path, leaf = tensor_name.rsplit(".", 1)
                try:
                    module = model.get_submodule(mod_path)
                except AttributeError:
                    module = None
            if module is not None:
                cur = getattr(module, "_parameters", {}).get(leaf)
                if isinstance(cur, QTensorWrapper):
                    tgt = stats.get("target_device") or device
                    module._parameters[leaf] = QTensorWrapper(
                        value.to(device=tgt, dtype=cur.data.dtype),  # exact cast-back for fp8
                        metadata=dict(cur.metadata),
                    )
                    stats["materialized"] += 1
                    return None
        return _orig(model, tensor_name, device, value=value, *args, **kwargs)

    mlu.set_module_tensor_to_device = _patched
    mlu._qtw_inplace_patched = True
    mlu._qtw_stats = stats


def _enforce_weight_only(transformer) -> None:
    """Re-apply the validated weight-only + spare regime: the state replay re-enables any
    quantizers that were disabled imperatively after quantize (NVFP4 default cfg has
    activation quantization ON in-config)."""
    n_act = n_spare = 0
    for name, m in transformer.named_modules():
        if not (name.endswith("_quantizer") and hasattr(m, "disable")):
            continue
        if name.endswith("weight_quantizer"):
            if any(s in name for s in SPARE_SUBSTRINGS):
                if getattr(m, "is_enabled", False):
                    n_spare += 1
                m.disable()
        else:
            if getattr(m, "is_enabled", False):
                n_act += 1
            m.disable()
    print(f"[load] re-disabled quantizers the state replay re-enabled: "
          f"{n_act} activation, {n_spare} spare-weight")


def _apply_dtype_nudges(transformer) -> None:
    """Reproduce the validated all-bf16 serve regime (all no-ops where already aligned)."""
    QTensorWrapper = _qtensor_wrapper_cls() or ()

    n_buf = n_par = n_meta = 0
    for m in transformer.modules():
        for bn, buf in list(m._buffers.items()):
            if buf is not None and buf.is_floating_point() and buf.dtype != torch.bfloat16:
                m._buffers[bn] = buf.to(torch.bfloat16)
                n_buf += 1
    for _, p in transformer.named_parameters():
        if isinstance(p, QTensorWrapper):
            d = p.metadata.get("dtype")
            if isinstance(d, torch.dtype) and d.is_floating_point and d != torch.bfloat16:
                p.metadata["dtype"] = torch.bfloat16   # dequant target only; payload untouched
                n_meta += 1
            continue  # packed payloads: never cast
        if p.is_floating_point() and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
            n_par += 1
    print(f"[load] normalized to bf16: {n_par} params, {n_buf} buffers; "
          f"retargeted {n_meta} dequant dtypes")

    def _cast_bf16(_m, args):
        return tuple(
            a.to(torch.bfloat16)
            if torch.is_tensor(a) and a.is_floating_point() and a.dtype != torch.bfloat16
            else a
            for a in args
        )

    for name, m in transformer.named_modules():
        if "time_embedder" in name and hasattr(m, "linear_1"):
            m.register_forward_pre_hook(_cast_bf16)


def load_pipe(
    model_id_or_path: str,
    *,
    torch_dtype=torch.bfloat16,
    enable_safety_checker: bool = False,
    device: str = "cuda",
    materialize_device: str | None = "cuda",   # packed weights stream straight here (RAM stays low)
    **kwargs,
):
    """Load a ModelOpt-quantized Cosmos3-Super pipeline with all load-time fixes applied."""
    from diffusers import Cosmos3OmniPipeline
    from modelopt.torch.opt import enable_huggingface_checkpointing

    enable_huggingface_checkpointing()   # must run before from_pretrained
    _patch_modelopt_quantizer()
    _patch_qtensor_loading()

    import diffusers.models.model_loading_utils as mlu
    if hasattr(mlu, "_qtw_stats"):
        mlu._qtw_stats["materialized"] = 0
        mlu._qtw_stats["target_device"] = materialize_device

    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        enable_safety_checker=enable_safety_checker,
        **kwargs,
    )

    n_mat = getattr(mlu, "_qtw_stats", {}).get("materialized", 0)
    print(f"[load] materialized {n_mat} packed quantized weight tensors")

    transformer = getattr(pipe, "transformer", None)
    if transformer is not None:
        _enforce_weight_only(transformer)
        _apply_dtype_nudges(transformer)

    pipe = pipe.to(device)

    QTensorWrapper = _qtensor_wrapper_cls()
    if QTensorWrapper is not None and transformer is not None:
        n_live = sum(1 for p in transformer.parameters() if isinstance(p, QTensorWrapper))
        print(f"[load] {n_live} quantized weight wrappers active after move to {device}")
        if n_mat and not n_live:
            print("[load] WARNING: wrappers were lost during .to() -- do not render; report this")

    return pipe


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "./cosmos3-super-nvfp4-hf"
    print(f"[load] loading {path} ...")
    pipe = load_pipe(path)
    print("[load] OK -- `pipe` is ready.")
    print("       NOTE: bare pipe(prompt) renders a 189-frame 720x1280 video (pipeline default).")
    print("       single-still smoke test:")
    print("       r = pipe('a red cube on a table', height=1024, width=1024, num_frames=1,")
    print("                num_inference_steps=50, guidance_scale=4.0); r.video[0].save('out.png')")
