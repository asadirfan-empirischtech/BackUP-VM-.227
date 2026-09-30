import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# 1. Pipeline Initialization
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

# 2. Physically-Grounded Prompt (Metaphors Removed)
raw_prompt = (
    "First-person dashcam perspective from the inside of a moving car, looking out through the windshield. "
    "It is daytime but extremely dark and overcast with heavy rain pouring down. The windshield wipers are moving rapidly. "
    "The car is driving on a wet, winding mountain asphalt road lined with dark pine trees. "
    "Suddenly, a massive physical landslide collapses from the steep rocky cliff on the right side of the frame. "
    "Hundreds of large grey boulders, thick brown mud, and several uprooted pine trees crash violently down onto the asphalt road directly in front of the car. "
    "The camera shakes violently as debris hits the road, completely blocking the path."
)

raw_negative_prompt = "static, blurry, low quality, distorted, extra limbs, watermark, text, morphing, unrealistic physics, bright sunlight"

# 3. Generation (Fixed argument passing and parameters)
result = pipe(
    prompt=raw_prompt,                     # Fixed: Passing the raw string directly
    negative_prompt=raw_negative_prompt,   # Fixed: Passing the raw string directly
    num_frames=85,                         # Optimized: Reduced to 85 frames for temporal stability
    height=720,
    width=1280,
    num_inference_steps=45,                # Optimized: Increased for better mud/rain detail
    guidance_scale=5.0,                    # Optimized: Slightly lowered to reduce texture artifacts
    fps=24.0,
)

# 4. Export
export_to_video(result.video, "car_landslide_optimized.mp4", fps=24, macro_block_size=1)
print("Saved: car_landslide_optimized.mp4")