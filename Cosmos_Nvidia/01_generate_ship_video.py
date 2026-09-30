import os
import sys
import torch
import cv2
import numpy as np

# ==============================================================================
# 1. DYNAMIC PATH INJECTION (Bypasses missing setup.py / pyproject.toml)
# ==============================================================================
BASE_DIR = "/home/azureuser/Documents/Cosmos_Nvidia"
DRIVE_DREAMS_PATH = os.path.join(BASE_DIR, "Cosmos-Drive-Dreams")
COSMOS_CORE_PATH = os.path.join(BASE_DIR, "Cosmos")

# Add both repository roots to sys.path if not already present
for p in [DRIVE_DREAMS_PATH, COSMOS_CORE_PATH]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)
        print(f"[INFO] Added repository path to sys.path: {p}")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "raw_frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = (
    "High quality dashcam video from an ego-vehicle, "
    "a massive cargo ship sitting stationary across a paved urban road lane, "
    "overcast daylight, photorealistic 4k"
)

print(f"\n[INFO] Starting NVIDIA Cosmos Synthetic Video Generation...")
print(f"[INFO] Prompt: '{PROMPT}'")
print(f"[INFO] Target Output Directory: {OUTPUT_DIR}\n")

# ==============================================================================
# 2. MODEL IMPORT & INFERENCE PIPELINE
# ==============================================================================
checkpoint_path = os.path.join(COSMOS_CORE_PATH, "checkpoints/Cosmos-1.0-Prompt2World-7B-Diffusion")
device = "cuda" if torch.cuda.is_available() else "cpu"

generator = None

# Attempt imports across potential repository subfolder structures
try:
    try:
        from cosmos_drive_dreams.inference import CosmosVideoGenerator
        print("[SUCCESS] Imported CosmosVideoGenerator from `cosmos_drive_dreams.inference`")
    except ModuleNotFoundError:
        from inference import CosmosVideoGenerator
        print("[SUCCESS] Imported CosmosVideoGenerator from root `inference`")

    # Initialize model if checkpoint exists
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        generator = CosmosVideoGenerator(
            checkpoint_path=checkpoint_path,
            device=device
        )
    else:
        print(f"[WARN] Checkpoint directory not found at: {checkpoint_path}")
        print("[WARN] Please ensure Hugging Face weights are downloaded using `huggingface-cli login`")

except Exception as err:
    print(f"[WARN] Could not initialize model pipeline directly: {err}")

# ==============================================================================
# 3. VIDEO GENERATION / SYNTHETIC FRAME EXTRACTION
# ==============================================================================
if generator is not None:
    print("\n[INFO] Generating synthetic frames using NVIDIA Cosmos...")
    frames = generator.generate(
        prompt=PROMPT,
        num_frames=30,      # Generate 30 frames (~2 seconds at 15 FPS)
        fps=15,
        resolution=(720, 1280)
    )

    for idx, frame in enumerate(frames):
        frame_path = os.path.join(OUTPUT_DIR, f"frame_{idx:04d}.png")
        # Convert RGB tensor/array to OpenCV BGR format
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)
        print(f"  -> Saved frame: {frame_path}")

    print(f"\n[SUCCESS] Generated and saved {len(frames)} frames to {OUTPUT_DIR}")

else:
    print("\n[FALLBACK MODE] Model weights or dependencies pending. Running fallback frame simulator...")
    # Creates test canvas frames so downstream pipeline scripts (02 & 03) can be developed & tested immediately
    for idx in range(10):
        # Create a synthetic 720p asphalt road frame with an obstacle canvas
        dummy_frame = np.full((720, 1280, 3), (80, 80, 80), dtype=np.uint8) # Asphalt grey
        cv2.rectangle(dummy_frame, (0, 500), (1280, 720), (50, 50, 50), -1) # Drivable lane
        
        # Draw a placeholder cargo ship hull box on the road
        cv2.rectangle(dummy_frame, (300, 350), (980, 550), (100, 40, 20), -1) 
        cv2.putText(
            dummy_frame, "SHIP ANOMALY ON ROAD", (350, 460), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
        )
        
        frame_path = os.path.join(OUTPUT_DIR, f"frame_{idx:04d}.png")
        cv2.imwrite(frame_path, dummy_frame)
        print(f"  -> [Fallback] Saved test frame: {frame_path}")

    print(f"\n[NOTICE] Generated 10 test frames in '{OUTPUT_DIR}' so you can proceed with segmentation testing.")