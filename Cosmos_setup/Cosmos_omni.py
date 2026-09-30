import json
import time
import base64
from pathlib import Path
import requests

# Local vLLM-Omni Image Endpoint
URL = "http://localhost:8000/v1/images/generations"

# Output folder creation
OUTPUT_DIR = Path("generated_pure_hazards_cctv_ultra")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Expanded negative prompt banning low-res artifacts, plastic textures, and depth-of-field blurring
NEGATIVE_PROMPT = (
    "PPE, hardhat, helmet, safety vest, high-vis, safety goggles, face shield, face mask, respirator, "
    "gloves, mutated anatomy, blurred faces, pixelated faces, headless, bad proportions, CGI, 3D render, "
    "cartoon, illustration, drawing, painting, cinematic lighting, black and white, monochrome, edited, "
    "depth of field, blurry, out of focus, bokeh, low resolution, noise, artifacts, plastic skin, video game engine"
)

# 2. Camera-specific base prompt demanding 8k resolution, tack-sharp focus, and raw sensor details
CCTV_BASE = (
    "Ultra-realistic, 8k resolution, raw high-definition security camera footage. High-angle CCTV perspective "
    "of an active heavy industrial factory floor. Shot on a high-end surveillance sensor, tack-sharp focus across the entire frame, "
    "highly detailed physical textures, natural subsurface scattering on human skin, authentic fluorescent factory lighting, "
    "photorealistic, unedited, hyper-detailed."
)

# -----------------------------------------------------------------------------
# 15 Hazard Scenarios (Intense, clear faces, zero PPE)
# -----------------------------------------------------------------------------
scenarios = [
    {
        "filename": "01_moving_vehicle_cctv.png",
        "prompt": f"{CCTV_BASE} A heavy industrial forklift driving fast forward. A factory worker wearing casual street clothes and no safety equipment is walking away with their back turned, completely unaware. The speeding forklift is directly behind the worker, inches away, about to violently crush the person from the back."
    },
    {
        "filename": "02_falling_hazard_cctv.png",
        "prompt": f"{CCTV_BASE} An overhead gantry crane suspending a massive heavy steel pipe in mid-air. A worker wearing a t-shirt and jeans, with a clear recognizable face and no hardhat, is standing directly on the floor space underneath the hanging load."
    },
    {
        "filename": "03_robotic_workspace_cctv.png",
        "prompt": f"{CCTV_BASE} An active, moving robotic welding arm. A worker with no safety gear, in regular clothes and a clear visible face, has stepped into the active workspace sweep radius while the robot is turned on and moving."
    },
    {
        "filename": "04_floor_spill_cctv.png",
        "prompt": f"{CCTV_BASE} A large puddle of leaked machine oil and water spread across the concrete floor. A worker with a clear face is walking briskly and stepping directly onto the slippery wet floor area."
    },
    {
        "filename": "05_conveyor_belt_cctv.png",
        "prompt": f"{CCTV_BASE} A moving industrial conveyor belt with exposed metal rollers. A worker wearing a loose unzipped jacket, with a clearly visible face and no gloves, is reaching their bare hands directly into the pinch points of the moving belt."
    },
    {
        "filename": "06_extreme_fire_cctv.png",
        "prompt": f"{CCTV_BASE} A glowing open hot furnace with molten metal and extreme fire. A worker wearing only a regular t-shirt, with a clear visible face and no thermal protective gear, is standing dangerously close in the immediate space next to the heat source."
    },
    {
        "filename": "07_high_edge_cctv.png",
        "prompt": f"{CCTV_BASE} An elevated metal scaffolding platform 20 feet above the ground. A section of the guardrail is missing. A worker in casual clothes, with a clear face and no safety harness, is standing right at the unguarded edge leaning over."
    },
    {
        "filename": "08_floor_opening_cctv.png",
        "prompt": f"{CCTV_BASE} A deep, open rectangular maintenance trench in the floor with no barricades. A distracted worker with a clear visible face and no PPE is walking dangerously close to the edge of the open hole."
    },
    {
        "filename": "09_crushing_machine_cctv.png",
        "prompt": f"{CCTV_BASE} A heavy metal stamping press machine actively running. The safety door is wide open and a worker with a clear visible face and no PPE is putting their bare hands completely inside the area where the heavy press comes down."
    },
    {
        "filename": "10_electrical_panel_cctv.png",
        "prompt": f"{CCTV_BASE} An open, high-voltage industrial electrical box exposing live wiring. A worker in regular clothes, with a clear visible face and no insulated gloves, is reaching their bare hands directly inside the live electrical panel."
    },
    {
        "filename": "11_flash_fire_chemical_cctv.png",
        "prompt": f"{CCTV_BASE} A spilled drum of flammable solvent on the factory floor with visible vapor haze, and a small ignition flame already flickering at its edge near an unshielded electrical spark source. A worker in regular clothes with a clear visible face is standing just a few feet away, unaware the vapor cloud is about to flash-ignite."
    },
    {
        "filename": "12_machinery_fire_response_cctv.png",
        "prompt": f"{CCTV_BASE} An industrial machine on the factory floor is actively on fire, with visible flames and thick smoke rising from its control panel. A worker with a clear visible face, wearing regular clothes and no fire-resistant gear, is standing very close to the burning machine holding a small handheld fire extinguisher, attempting to fight the fire without any protective barrier."
    },
    {
        "filename": "13_blocked_exit_fire_cctv.png",
        "prompt": f"{CCTV_BASE} Thick black smoke and visible flames spreading along stacked pallets near a factory exit door, which is partially blocked by stored crates. A worker with a clear visible face is running toward the blocked exit as the fire and smoke rapidly close off the escape route."
    },
    {
        "filename": "14_welding_spark_ignition_cctv.png",
        "prompt": f"{CCTV_BASE} A worker with a clear visible face is actively welding without a face shield or fire-resistant apron, with bright welding sparks flying directly onto a nearby open container of flammable liquid and loose cardboard packaging, with a small flame just beginning to catch."
    },
    {
        "filename": "15_electrical_fire_overload_cctv.png",
        "prompt": f"{CCTV_BASE} An overloaded tangle of power cords and an electrical distribution box emitting visible sparks and small flames from overheating. A worker with a clear visible face is standing directly next to the sparking equipment, reaching toward it with bare hands to unplug a cord."
    },
]

# -----------------------------------------------------------------------------
# Execution Loop
# -----------------------------------------------------------------------------
print(f"Starting batch generation of {len(scenarios)} ultra-realistic hazard images...")
print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

for idx, item in enumerate(scenarios, start=1):
    file_path = OUTPUT_DIR / item["filename"]
    print(f"[{idx}/{len(scenarios)}] Generating: {item['filename']}...")

    payload = {
        "prompt": item["prompt"],
        "negative_prompt": NEGATIVE_PROMPT,
        "width": 1920,
        "height": 1080,
        "num_inference_steps": 50, # 3. Increased from 35 to 50 for maximum detail rendering
        "guidance_scale": 5.0,     # 4. Bumped slightly to 5.0 to ensure text prompt adherence and sharp edges
        "seed": 10500 + idx,
        "extra_params": {
            "guardrails": False
        }
    }

    start_time = time.time()

    try:
        response = requests.post(
            URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            if "image" in response.headers.get("Content-Type", "") or response.content[:4] in [b'\x89PNG', b'\xff\xd8\xff\xe0']:
                file_path.write_bytes(response.content)
            else:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    img_data = base64.b64decode(data["data"][0]["b64_json"])
                    file_path.write_bytes(img_data)

            print(f" -> Successfully saved to {file_path} (Took {elapsed:.1f}s)\n")
        else:
            print(f" -> Failed [HTTP {response.status_code}]: {response.text}\n")

    except Exception as e:
        print(f" -> Error connecting to API: {e}\n")

print(f"All {len(scenarios)} image requests completed!")
