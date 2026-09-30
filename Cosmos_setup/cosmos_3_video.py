import time
from pathlib import Path
import requests

URL = "http://localhost:8000/v1/videos/sync"

OUTPUT_DIR = Path("generated_cosmos_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEGATIVE_PROMPT = (
    "dark, dim, shadowy, underexposed, gloomy, moody lighting, low contrast, "
    "blurry, motion blur, out of focus, hazy, grainy, noise, low resolution, "
    "mutated anatomy, deformed, missing limbs, floating objects, bad proportions, CGI, 3D render, "
    "cartoon, security camera grid overlay, impossible physics, jerky motion, flickering, morphing, text, watermark"
)

COSMOS_VIDEO_BASE = (
    "High-quality 3-second continuous video clip. Exact top-down 90-degree overhead bird's-eye view from a ceiling CCTV security camera, looking straight down at the floor. "
    "Extremely bright, clean, modern manufacturing factory environment. "
    "High-key illumination, brilliantly lit with clean cool-white 6000K overhead industrial LED panels, no dark shadows, uniform bright lighting. "
    "Pristine, clean light-grey epoxy floor with high clarity and contrast. "
    "Crisp focus, razor-sharp edge definition, 4k detail, fluid realistic human motion and physics. 24 fps, cinematic."
)

scenarios = [
    {
        "filename": "01_worker_avoiding_pit.mp4",
        "prompt": (
            f"{COSMOS_VIDEO_BASE} In the center, a highly dangerous, deep open maintenance pit with exposed electrical wiring and a sparking transformer, surrounded by bright yellow warning tape on the floor. "
            "A worker in a high-visibility yellow hardhat and clean blue coveralls walks steadily from the top edge towards the bottom. "
            "The worker maintains an appropriately safe distance, walking precisely 3 meters away from the edge of the maintenance pit. "
            "Sharp, natural walking motion, crisp natural shadows directly beneath feet on the bright concrete floor."
        )
    },
    {
        "filename": "02_workers_inspecting_robot.mp4",
        "prompt": (
            f"{COSMOS_VIDEO_BASE} In the center, a large, malfunctioning industrial robotic arm swinging erratically inside a heavy steel mesh safety enclosure. "
            "Two workers in bright neon safety vests stand at a safe distance of exactly 5 meters outside the steel enclosure. "
            "One worker paces slowly while observing the hazard from afar, while the other clearly gestures toward the sparking robot. "
            "Razor-sharp silhouettes, distinct body language, pristine spatial stability and consistent lighting throughout."
        )
    }
]

print(f"Starting batch generation of {len(scenarios)} Cosmos video clips...")
print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

for idx, item in enumerate(scenarios, start=1):
    file_path = OUTPUT_DIR / item["filename"]
    print(f"[{idx}/{len(scenarios)}] Generating Video: {item['filename']}...")

    # Adjusted to 960x540 (qHD) at 73 frames (3s) to survive VAE decoding memory spikes
    payload = {
        "model": "nvidia/Cosmos3-Super",
        "prompt": item["prompt"],
        "negative_prompt": NEGATIVE_PROMPT,
        "width": "960",             
        "height": "540",            
        "num_frames": "73",         
        "fps": "24",
        "num_inference_steps": "25",
        "guidance_scale": "7.5",
        "seed": str(30500 + idx),
        "guardrails": "false"
    }

    # Convert the flat payload into a format that forces requests to send multipart/form-data
    multipart_data = {key: (None, str(value)) for key, value in payload.items()}

    start_time = time.time()

    try:
        # 30-minute timeout explicitly set so requests does not drop the connection
        response = requests.post(
            URL,
            files=multipart_data,
            headers={"Accept": "video/mp4"},
            timeout=1800 
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            file_path.write_bytes(response.content)
            print(f" -> Successfully saved MP4 to {file_path} (Took {elapsed:.1f}s)\n")
        else:
            print(f" -> Failed [HTTP {response.status_code}]: {response.text}\n")

    except Exception as e:
        print(f" -> Error connecting to API: {e}\n")

print("All video requests completed!")
