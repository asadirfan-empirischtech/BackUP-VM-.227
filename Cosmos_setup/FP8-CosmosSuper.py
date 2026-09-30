import os
import torch
import gc
from huggingface_hub import login
from diffusers import UniPCMultistepScheduler
from load_cosmos3_modelopt import load_pipe

# Apply PyTorch memory fragmentation fix (based on your OOM error log)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Hugging Face Authentication
hf_token = "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" 
login(token=hf_token)

# 2. Model Initialization (Switched to 4-bit to prevent OOM)
print("Loading Cosmos3-Super NVFP4 model (~43GB VRAM)...")
# Using the smaller footprint 4-bit model
model_id = "prometheusAIR/Cosmos3-Super-nvfp4" 

pipe = load_pipe(model_id)

# Offload text encoders/VAE to CPU when not actively computing to save even more VRAM
pipe.enable_model_cpu_offload()

# 3. Configure Scheduler for Still Images
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, 
    flow_shift=3.0  # NVIDIA's recommended setting for text-to-image
)

# 4. Define the 5 Prompts
prompts = [
    # Image 1: Police Escort with Traffic Cones
    "Front dashcam perspective looking strictly out the windshield of a black Toyota Land Cruiser flanked closely by police escort vehicles with flashing red and blue strobe lights, maneuvering through a narrow chicane lined with bright orange reflective traffic cones on a wet urban asphalt street at night, photoreal, 8k resolution, sharp focus.",

    # Image 2: Massive Landslide
    "First-person dashcam perspective looking out the windshield at a massive mountain landslide crashing across a wet highway. Large grey boulders, thick dark mud, and fallen pine trees completely blocking the asphalt road under heavy rain and dark overcast skies, hyper-realistic, highly detailed.",

    # Image 3: Damaged Highway
    "Eye-level road camera view of passenger cars navigating across a severely fractured section of highway asphalt, deep rain-filled potholes, cracked pavement, loose gravel, and bright orange roadwork warning signs under an overcast sky, photoreal, documentary style.",

    # Image 4: Drone Aerial Landslide
    "High-angle cinematic drone shot looking down at a steep forested mountain ridge with a massive catastrophic debris flow and mudslide sweeping down and burying a narrow two-lane highway below, photoreal, realistic textures.",

    # Image 5: Falling Cargo Hazard
    "Forward-facing dashcam perspective closely behind a three-wheeled green tempo cargo vehicle on a multi-lane highway, capturing the exact moment a large cardboard box falls off the rear bed and hits the asphalt road, realistic road hazard, sharp focus."
]

# 5. Batch Image Generation Loop
print("\n" + "="*50)
print("STARTING BATCH IMAGE GENERATION (1024x1024)")
print("="*50 + "\n")

output_dir = "cosmos_super_images"
os.makedirs(output_dir, exist_ok=True)

for idx, prompt_text in enumerate(prompts, start=1):
    output_path = os.path.join(output_dir, f"generation_{idx}.png")
    print(f"[{idx}/5] Generating: {output_path}...")
    
    # Generate single frame (num_frames=1 is critical for VRAM budget)
    result = pipe(
        prompt=prompt_text,
        height=1024,
        width=1024,
        num_frames=1,
        num_inference_steps=50,
        guidance_scale=4.0
    )
    
    # Save the generated PIL image
    result.video[0].save(output_path)
    print(f"✅ Saved: {output_path}\n")
    
    # Clear CUDA cache between generations
    torch.cuda.empty_cache()
    gc.collect()

print(f"All images successfully generated and saved to ./{output_dir}/")