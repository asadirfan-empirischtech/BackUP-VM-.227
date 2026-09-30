import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
import gc

# 1. Pipeline Initialization (Loaded ONLY ONCE)
print("Loading Cosmos3-Nano model into VRAM...")
pipe = Cosmos3OmniPipeline.from_pretrained(
    "nvidia/Cosmos3-Nano",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    enable_safety_checker=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=10.0, use_karras_sigmas=False
)

# 2. Define the 5 Reprompted Videos (Exterior Dashcam Focus Only)
enhanced_prompts = [
    # Video 1: Police Escort (Interior details removed)
    "Front dashcam perspective looking strictly out the windshield of a moving vehicle. The vehicle is part of a high-speed escort, flanked closely by two police escort vehicles on either side, their strobe lights pulsating in a rhythmic dance of red and blue. The road is a narrow urban avenue, its smooth tarmac glistening under streetlights. The avenue is lined with a double row of bright orange reflective traffic cones guiding the procession through a series of precise, controlled chicane turns.",
    
    # Video 2: Landslide (Interior details removed)
    "First-person dashcam video looking strictly out the windshield of a car navigating a treacherous, winding mountain road during a relentless downpour. The road is slick, glistening under the dim glow of distant headlights. Suddenly, the ground beneath the cliffside to the left begins to tremble. With a deafening roar, a colossal landslide breaks off from the cliff, a wall of churning mud, massive boulders, and uprooted trees cascading down, engulfing the asphalt road ahead in a chaotic wave of brown and gray.",
    
    # Video 3: Damaged Highway (Interior details removed)
    "First-person, eye-level dashcam view looking strictly out the windshield of a passenger car navigating a severely distressed highway. The asphalt is a labyrinth of deep potholes filled with rainwater, and cracked sections with loose gravel. Faded roadwork warning signs with bright orange and black lettering punctuate the scene. A diverse array of passenger cars and SUVs cautiously traverse the damaged road, maneuvering gingerly around the obstacles under a soft, overcast sky.",
    
    # Video 4: Drone Landslide (Simplified for literal rendering)
    "High-angled cinematic drone shot looking down at a rugged, steep mountain ridge cloaked in a lush forest. A winding, narrow highway snakes through the valley below. Suddenly, a catastrophic slope failure occurs. A vast swath of earth, rock, and mud cascades down the mountainside in a powerful wave. The churning mass of dark brown and gray debris sweeps across the road, uprooting trees and completely blocking the highway.",
    
    # Video 5: Falling Parcel (Interior details removed)
    "First-person, forward-facing dashcam sequence looking strictly out the windshield, following closely behind a rustic, three-wheeled tempo cargo vehicle on a busy multi-lane highway. The tempo is laden with cardboard boxes and parcels. Suddenly, the tempo hits a bump. An oversized cardboard box teeters and tumbles off the truck bed, bouncing erratically across the hot asphalt roadway, narrowly missing the surrounding traffic."
]

# Adding universal quality triggers to the negative prompt
raw_negative_prompt = "interior, dashboard, steering wheel, inside car, static, blurry, low quality, distorted, extra limbs, watermark, text, morphing, unrealistic physics, pixelated"

# 3. Batch Generation Loop
print("\n" + "="*50)
print("STARTING BATCH VIDEO GENERATION (NATIVE 720p)")
print("="*50 + "\n")

for idx, prompt in enumerate(enhanced_prompts, start=1):
    output_filename = f"cosmos_generation_{idx}.mp4"
    print(f"[{idx}/5] Generating: {output_filename}...")
    
    # Generate the video (Reverted to Native 720p for visual clarity)
    result = pipe(
        prompt=prompt,                     
        negative_prompt=raw_negative_prompt,   
        num_frames=85,                         # Bumping back to 85 frames since resolution is lower
        height=720,                            # Native resolution height
        width=1280,                            # Native resolution width
        num_inference_steps=45,                
        guidance_scale=6.0,                    # Increased slightly for sharper object edges
        fps=24.0,
    )
    
    # Export and save
    export_to_video(result.video, output_filename, fps=24, macro_block_size=1)
    print(f"✅ Saved successfully: {output_filename}\n")
    
    # Clear CUDA cache between generations to prevent memory fragmentation
    torch.cuda.empty_cache()
    gc.collect()

print("🎉 All 5 videos have been successfully generated!")