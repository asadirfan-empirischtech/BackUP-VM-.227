import json
import time
import base64
from pathlib import Path
import requests

# Local Image Endpoint
URL = "http://localhost:8000/v1/images/generations"

# Output folder creation
OUTPUT_DIR = Path("generated_factory_hazards_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Aggressive negative prompt to fix anatomy, scale, and warped perspective issues
NEGATIVE_PROMPT = (
    "mutated anatomy, headless, distorted faces, bad proportions, miniature people, giant objects, "
    "CGI, 3D render, cartoon, digital art, warped perspective, fisheye, security camera grid, "
    "blue overlay, twisted architecture, Escher, impossible physics, text, watermarks"
)

# Clean, distortion-free photography bases
WIDE_BASE = (
    "Wide-angle architectural photography of a bright, modern industrial manufacturing plant. "
    "Photorealistic, perfect perspective, sharp focus, highly detailed."
)

EYE_LEVEL_BASE = (
    "Eye-level documentary photography inside a heavy machinery factory. "
    "Photorealistic, natural lighting, realistic human proportions, grounded."
)

# -----------------------------------------------------------------------------
# 20 Realistic Scenario Definitions
# -----------------------------------------------------------------------------
scenarios = [
    # --- 1. Vehicles ---
    {
        "filename": "vehicle_01.png",
        "prompt": f"{WIDE_BASE} A yellow forklift carrying a wooden pallet is driving down the main concrete aisle. A worker in a high-vis vest is walking in the same aisle, close to the forklift's path."
    },
    {
        "filename": "vehicle_02.png",
        "prompt": f"{EYE_LEVEL_BASE} A parked industrial tugger cart in a warehouse. A worker is walking backwards while pulling a heavy manual pallet jack nearby."
    },

    # --- 2. Overhead Loads ---
    {
        "filename": "crane_01.png",
        "prompt": f"{WIDE_BASE} An overhead gantry crane is actively lifting a large steel pipe above the factory floor. A factory worker wearing a hardhat is standing on the floor in the general vicinity of the lift."
    },
    {
        "filename": "crane_02.png",
        "prompt": f"{EYE_LEVEL_BASE} A chain hoist hanging from the ceiling is holding a heavy engine block. A mechanic is walking past the suspended load."
    },

    # --- 3. Robotic Workspaces ---
    {
        "filename": "robot_01.png",
        "prompt": f"{WIDE_BASE} A yellow robotic welding arm in a factory cell. The floor is painted with yellow warning lines. A worker is standing inside the yellow lined area while the robot is active."
    },
    {
        "filename": "robot_02.png",
        "prompt": f"{EYE_LEVEL_BASE} An industrial robotic assembly line. A technician is leaning forward toward the robotic arm to inspect a component on the belt."
    },

    # --- 4. Floor Spills & Obstacles ---
    {
        "filename": "spill_01.png",
        "prompt": f"{WIDE_BASE} A distinct puddle of dark hydraulic oil leaked onto a clean grey concrete walkway. A worker is walking toward the puddle."
    },
    {
        "filename": "spill_02.png",
        "prompt": f"{EYE_LEVEL_BASE} A heavy industrial extension cord is stretched tight across a designated walking aisle. A worker carrying a box is approaching the cord."
    },

    # --- 5. Moving Machinery ---
    {
        "filename": "machine_01.png",
        "prompt": f"{WIDE_BASE} A large industrial conveyor belt system moving cardboard boxes. A worker is standing right next to the exposed metal rollers."
    },
    {
        "filename": "machine_02.png",
        "prompt": f"{EYE_LEVEL_BASE} A heavy metal lathe machine spinning a steel rod. A machinist wearing a loose, unzipped jacket is standing at the controls."
    },

    # --- 6. Extreme Heat ---
    {
        "filename": "heat_01.png",
        "prompt": f"{WIDE_BASE} A glowing orange smelting furnace in a foundry. A worker wearing standard cotton work clothes is standing near the open furnace door."
    },
    {
        "filename": "heat_02.png",
        "prompt": f"{EYE_LEVEL_BASE} Freshly forged, red-hot steel glowing on a cooling rack. A worker without safety glasses is standing near the rack inspecting the metal."
    },

    # --- 7. Edges & Elevations ---
    {
        "filename": "edge_01.png",
        "prompt": f"{WIDE_BASE} A steel mezzanine platform in a factory. One section of the yellow safety railing is entirely missing. A worker is standing on the platform near the gap."
    },
    {
        "filename": "edge_02.png",
        "prompt": f"{EYE_LEVEL_BASE} A worker standing on the top step of a tall aluminum A-frame ladder in a warehouse, reaching up to fix a light fixture."
    },

    # --- 8. Floor Openings ---
    {
        "filename": "opening_01.png",
        "prompt": f"{WIDE_BASE} An open rectangular maintenance pit in a garage floor. There are no cones or barricades around it. A mechanic is walking past the edge of the pit."
    },
    {
        "filename": "opening_02.png",
        "prompt": f"{EYE_LEVEL_BASE} A factory floor where a large metal grate has been removed, exposing a deep trench. A worker is walking near the open trench."
    },

    # --- 9. Heavy Presses ---
    {
        "filename": "press_01.png",
        "prompt": f"{WIDE_BASE} A massive green hydraulic stamping press. The front safety door is open. A worker is standing directly at the machine interface."
    },
    {
        "filename": "press_02.png",
        "prompt": f"{EYE_LEVEL_BASE} An industrial sheet metal bending brake. A worker is holding a piece of metal inside the jaws of the machine."
    },

    # --- 10. Electrical & Maintenance ---
    {
        "filename": "electrical_01.png",
        "prompt": f"{WIDE_BASE} A large grey industrial electrical panel on a wall with its door wide open, exposing dense wiring. A worker is standing in front of the open panel."
    },
    {
        "filename": "electrical_02.png",
        "prompt": f"{EYE_LEVEL_BASE} A worker standing in a puddle of water while inspecting a large industrial power generator."
    }
]

# -----------------------------------------------------------------------------
# Batch Generation Execution Loop
# -----------------------------------------------------------------------------
print(f"Starting batch generation of {len(scenarios)} realistic factory hazard images...")
print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

for idx, item in enumerate(scenarios, start=1):
    file_path = OUTPUT_DIR / item["filename"]
    print(f"[{idx}/{len(scenarios)}] Generating: {item['filename']}...")

    payload = {
        "prompt": item["prompt"],
        "negative_prompt": NEGATIVE_PROMPT,
        "width": 1920, 
        "height": 1080,
        "num_inference_steps": 30,
        "guidance_scale": 7.0,
        "seed": 9000 + idx
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
