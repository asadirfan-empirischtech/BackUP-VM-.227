import json
import time
import base64
from pathlib import Path
import requests

# Local vLLM-Omni Image/Video Endpoint
URL = "http://localhost:8000/v1/images/generations"

# Output folder creation
OUTPUT_DIR = Path("generated_cosmos_distance_semantics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Aggressive negative prompt to enforce high realism and physical consistency
NEGATIVE_PROMPT = (
    "mutated anatomy, deformed, missing limbs, floating objects, bad proportions, CGI, 3D render, "
    "cartoon, fisheye, security camera grid overlay, impossible physics, impossible shadows, blurred faces, pixelated faces, text, watermark"
)

# Enhanced base prompt specifically tuned for NVIDIA Cosmos physics and lighting semantics
COSMOS_BASE = (
    "Exact top-down 90-degree overhead bird's-eye view from a ceiling CCTV security camera, looking perfectly straight down at the floor. "
    "Photorealistic modern manufacturing factory environment. "
    "Physically accurate global illumination with bright cool white 5000K industrial LED panel lighting. "
    "Light sources cast physically consistent, soft, diffused ambient occlusion shadows directly beneath objects, anchoring them to the floor. "
    "Smooth light-grey epoxy-coated concrete floor showing realistic micro-texture and subtle specular highlights. "
    "Cinematic depth of field, sharp focus, perfect spatial consistency, volumetric lighting."
)

# -----------------------------------------------------------------------------
# Distance Scenarios (1m, 2m, 3m, 5m) for Cosmos
# Emphasizing physical spacing and proper spatial anchoring
# -----------------------------------------------------------------------------
scenarios = [
    # --- HAZARD 1: TRAFFIC CONES ---
    {
        "filename": "01_cones_1_meter.png",
        "prompt": (
            f"{COSMOS_BASE} In the center, a rectangular hazard zone marked by bright orange PVC traffic cones with black rubber bases. "
            "A worker in a yellow hardhat and blue coveralls is standing exactly 1 meter away from the perimeter of the cones. "
            "The physical 1-meter spatial gap between the worker's steel-toed boots and the nearest cone base is clearly visible on the concrete floor. "
            "Both the worker and the cones cast consistent, proportionate drop shadows."
        )
    },
    {
        "filename": "02_cones_2_meters.png",
        "prompt": (
            f"{COSMOS_BASE} In the center, a rectangular hazard zone marked by bright orange PVC traffic cones. "
            "A worker in a white hardhat and high-visibility vest is standing exactly 2 meters away from the hazard perimeter. "
            "There is a distinct, physically accurate 2-meter gap of empty grey concrete separating the worker from the cones. "
            "Proper physics semantics: the scale of the worker relative to the cones correctly reflects the overhead perspective."
        )
    },
    {
        "filename": "03_cones_3_meters.png",
        "prompt": (
            f"{COSMOS_BASE} In the center, a rectangular hazard zone marked by bright orange PVC traffic cones. "
            "Two workers are standing together exactly 3 meters away from the hazard boundary. "
            "A wide, 3-meter physical expanse of empty factory floor lies between the workers and the safety cones, illustrating spatial depth in a 2D top-down plane. "
            "Realistic light bouncing off the concrete floor provides natural fill light to the objects."
        )
    },
    {
        "filename": "04_cones_5_meters.png",
        "prompt": (
            f"{COSMOS_BASE} In the top-left quadrant, a rectangular hazard zone marked by bright orange PVC traffic cones. "
            "A worker in a yellow hardhat is standing in the bottom-right quadrant, exactly 5 meters away from the hazard zone. "
            "A massive 5-meter physical distance separates the worker from the cones, occupying a large portion of the frame. "
            "The spatial layout strictly adheres to real-world physics and scale."
        )
    },

    # --- HAZARD 2: METAL BARRICADES ---
    {
        "filename": "05_barricades_1_meter.png",
        "prompt": (
            f"{COSMOS_BASE} In the center, an enclosed hazard zone surrounded by heavy interlocking galvanized steel crowd-control barrier panels. "
            "A worker in blue coveralls is standing exactly 1 meter away from the steel barricades. "
            "The 1-meter physical clearance between the worker and the metal fence is clearly depicted, with accurate shadows anchoring both elements to the floor."
        )
    },
    {
        "filename": "06_barricades_2_meters.png",
        "prompt": (
            f"{COSMOS_BASE} In the center, an enclosed hazard zone surrounded by heavy galvanized steel crowd-control barriers. "
            "A worker in a safety vest is standing exactly 2 meters away from the metal barricades. "
            "A clear, 2-meter physical distance of empty floor separates the worker from the barricades. "
            "Volumetric lighting emphasizes the 3D structure of the barriers and the worker from the top-down perspective."
        )
    },
    {
        "filename": "07_barricades_3_meters.png",
        "prompt": (
            f"{COSMOS_BASE} In the center, an enclosed hazard zone surrounded by galvanized steel barriers. "
            "A worker operating a yellow manual pallet jack is standing exactly 3 meters away from the barricade perimeter. "
            "A substantial 3-meter physical gap of concrete is visible between the pallet jack and the steel fence. "
            "Accurate ambient occlusion shadows ground the heavy equipment to the floor."
        )
    },
    {
        "filename": "08_barricades_5_meters.png",
        "prompt": (
            f"{COSMOS_BASE} In the top half of the frame, an enclosed hazard zone surrounded by galvanized steel barriers. "
            "A worker in a hardhat is standing in the bottom half of the frame, exactly 5 meters away from the barricades. "
            "The vast 5-meter physical separation is accurately scaled, demonstrating proper top-down physical perspective and camera lens geometry."
        )
    },

    # --- HAZARD 3: PAINTED FLOOR MARKINGS ---
    {
        "filename": "09_painted_1_meter.png",
        "prompt": (
            f"{COSMOS_BASE} On the factory floor, a permanent restricted zone is painted with bright vivid yellow 10cm-wide lines forming a rectangle with diagonal hatching. "
            "A worker in a white hardhat is standing exactly 1 meter away from the outer edge of the yellow painted boundary. "
            "The 1-meter spatial gap of plain grey concrete between the worker's boots and the paint is precisely rendered."
        )
    },
    {
        "filename": "10_painted_2_meters.png",
        "prompt": (
            f"{COSMOS_BASE} A hazard zone marked by bright yellow painted floor lines. "
            "A worker in a yellow safety vest is standing exactly 2 meters away from the painted boundary. "
            "There is a clear 2-meter physical distance of unmarked concrete separating the worker from the hazard marking. "
            "The lighting perfectly captures the matte finish of the floor paint versus the glossy epoxy floor."
        )
    },
    {
        "filename": "11_painted_3_meters.png",
        "prompt": (
            f"{COSMOS_BASE} A hazard zone defined by a bright yellow painted rectangular border on the floor. "
            "Two workers are standing side-by-side exactly 3 meters away from the nearest painted line. "
            "A wide 3-meter expanse of empty floor lies between them and the hazard zone, demonstrating excellent physical spatial relationship."
        )
    },
    {
        "filename": "12_painted_5_meters.png",
        "prompt": (
            f"{COSMOS_BASE} In the center of the frame, a relatively small hazard zone marked by bright yellow floor paint. "
            "A worker in blue coveralls is standing far away, exactly 5 meters from the painted zone. "
            "The 5-meter physical distance is visually immense in the overhead view, showcasing accurate spatial scaling and top-down geometry."
        )
    }
]

# -----------------------------------------------------------------------------
# Execution Loop
# -----------------------------------------------------------------------------
print(f"Starting batch generation of {len(scenarios)} Cosmos-optimized distance semantic images...")
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
        "seed": 20500 + idx,
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
