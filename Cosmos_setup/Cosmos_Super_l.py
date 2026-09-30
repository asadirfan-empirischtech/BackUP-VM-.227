import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# Load across all available GPUs (0 and 1)
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Super",
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
if hasattr(pipe, "disable_safety_checker"):
    pipe.disable_safety_checker()
pipe.safety_checker = None
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)

# Generate Text-to-Video
result = pipe(
    prompt="Dashcam recording, heavy daytime landslide on a mountain road, rocks and dust collapsing onto asphalt, realistic physics, 4k",
    negative_prompt="blurry, distorted, low quality, jittery",
    image=None,
    num_frames=81,         # Start with 81 frames to verify VRAM headroom
    height=480,            # Tier: 480p (832x480) or 720p (1280x720)
    width=832,
    fps=24,
    num_inference_steps=35,
    guidance_scale=6.0,
    enable_sound=False,
    add_resolution_template=False,
    add_duration_template=False,
    generator=torch.Generator(device="cuda").manual_seed(2026),
)

export_to_video(result.video, "landslide_dashcam.mp4", fps=24, macro_block_size=1)
