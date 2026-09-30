import json
import torch
import gc
from diffusers import Cosmos3OmniPipeline, Cosmos3OmniTransformer
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper
import modelopt.torch.opt as mto

# 1. Compatibility Patch for Diffusers and Accelerate
def patch_modelopt_qtensor_loader():
    import accelerate.utils.modeling as accelerate_modeling
    import diffusers.models.model_loading_utils as diffusers_loading

    original = accelerate_modeling.set_module_tensor_to_device
    if getattr(original, "_cosmos3_modelopt_patch", False):
        return

    def patched(module, tensor_name, device, value=None, dtype=None, fp16_statistics=None,
                tied_params_map=None, non_blocking=False, clear_cache=True):
        leaf_module = module
        leaf_name = tensor_name
        if "." in tensor_name:
            parts = tensor_name.split(".")
            for part in parts[:-1]:
                leaf_module = getattr(leaf_module, part)
            leaf_name = parts[-1]
        old_value = getattr(leaf_module, leaf_name) if hasattr(leaf_module, leaf_name) else None
        if isinstance(old_value, QTensorWrapper) and value is not None:
            leaf_module._parameters[leaf_name] = QTensorWrapper(
                value.to(device, non_blocking=non_blocking),
                metadata=old_value.metadata,
            )
            return
        return original(module, tensor_name, device, value, dtype, fp16_statistics,
                        tied_params_map, non_blocking, clear_cache)

    patched._cosmos3_modelopt_patch = True
    accelerate_modeling.set_module_tensor_to_device = patched
    diffusers_loading.set_module_tensor_to_device = patched

def cast_modelopt_runtime_tensors(model, dtype=torch.bfloat16):
    for module in model.modules():
        for name, param in list(module._parameters.items()):
            if isinstance(param, QTensorWrapper):
                param.metadata["dtype"] = dtype
            elif param is not None and param.is_floating_point():
                module._parameters[name] = torch.nn.Parameter(
                    param.detach().to(dtype),
                    requires_grad=param.requires_grad,
                )
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_floating_point():
                module._buffers[name] = buf.to(dtype)
    return model

# Apply patches before loading
patch_modelopt_qtensor_loader()
mto.enable_huggingface_checkpointing()

# 2. Load the FP8 Transformer into CPU/Swap Space (Bypassing GPU Warmup)
print("Loading FP8 Transformer into CPU & Swap...")
transformer = Cosmos3OmniTransformer.from_pretrained(
    "WaveCut/Cosmos3-Super-Text2Image-ModelOpt-FP8-Transformer",
    subfolder="transformer",
    use_safetensors=False
    # NO device_map="cuda" here
)
transformer = cast_modelopt_runtime_tensors(transformer, torch.bfloat16)

gc.collect()

# 3. Assemble the Full Pipeline in CPU/Swap Space
print("Assembling Pipeline...")
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super-Text2Image",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    enable_safety_checker=False
    # NO device_map="cuda" here either
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

# 4. MANUALLY move the shrunken 61 GB model to the H100
print("Pushing the compressed 61 GB model to the GPU...")
pipe.to("cuda")

# 5. Define the Prompt
json_caption = {
    "subjects": ["A weathered lighthouse on a cliff"],
    "background_setting": "Golden hour lighting, photoreal, 50mm.",
    "comprehensive_t2i_caption": "A weathered lighthouse on a cliff at golden hour, photoreal, 50mm.",
    "resolution": {"H": 1024, "W": 1024},
    "aspect_ratio": "1,1",
}

# 6. Generate Image
print("Generating Image...")
with torch.autocast("cuda", dtype=torch.bfloat16):
    result = pipe(
        prompt=json.dumps(json_caption),
        negative_prompt="",
        num_frames=1,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=4.0,
        generator=torch.Generator(device="cuda").manual_seed(1143),
    )

result.video[0].save("cosmos3_modelopt_fp8.png")
print("Saved cosmos3_modelopt_fp8.png")