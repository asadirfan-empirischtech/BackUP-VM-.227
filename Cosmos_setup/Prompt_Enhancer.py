import os
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------------------------
# 1. Authentication
# NEVER hardcode a token in a script. Set it in your shell instead:
#   export HF_TOKEN="your_token_here"        (Linux/macOS)
#   setx HF_TOKEN "your_token_here"          (Windows)
# If a token was ever pasted into a script, chat, or committed to a
# repo, treat it as compromised and rotate/revoke it on huggingface.co
# under Settings -> Access Tokens.
# -------------------------------------------------------------------
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not set. Proceeding without auth "
          "(fine for public repos, required for gated ones).")

# 2. Configuration
model_id = "mradermacher/Cosmos-1.0-Prompt-Upsampler-12B-Text2World-hf-GGUF"
filename = "Cosmos-1.0-Prompt-Upsampler-12B-Text2World-hf.Q4_K_M.gguf"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename, token=hf_token)

print("Loading Model onto GPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    gguf_file=filename,
    device_map="auto",
    token=hf_token,
)

# 3. Base prompts: physics-grounded, natural-movement, real-world-scale scenes
prompts = [
    # --- Landslides / rockfall, dashcam only, no mud (3) ---
    "Forward-facing dashcam POV, dry mountain switchback road, late afternoon sun: a cliffside above the right lane suddenly sheds a dense cluster of dry boulders and loose scree, no mud or wet debris, individual rocks detaching at slightly different moments and tumbling with visible mass-dependent momentum, some skipping and shattering into fragments on the asphalt, dust puffing up on impact, the dashcam view shaking as the driver brakes hard and swerves left to avoid the largest boulder rolling to a stop mid-lane.",
    "Forward-facing dashcam POV descending a forested ridge road, overcast daylight: a rockfall releases from a dry cut-slope above the highway, a cascade of bare rock and stone fragments bouncing down through sparse trees and clattering onto the road surface in a scattered line, no soil or mud, each rock's bounce height and roll distance varying with its size and the slope angle, the dashcam's windshield-wiper rhythm continuing as the car brakes and stops short of the debris field, other cars visible ahead doing the same.",
    "Forward-facing dashcam POV on a highway shoulder, midday, dry clear conditions: a sudden tremor dislodges a cluster of boulders from a road-cut cliff face directly ahead, rocks detaching asynchronously, tumbling end-over-end and shattering into smaller fragments on the first impact, fragments ricocheting unpredictably onto the roadway and rolling to a stop with visibly decaying momentum, the dashcam capturing the car swerving into the opposite lane to avoid the debris, dust hanging in the still air afterward.",

    # --- Animals on a highway inside a tunnel, dashcam (2) ---
    "Forward-facing dashcam POV inside a well-lit two-lane tunnel, moderate traffic: a herd of wild goats emerges from a maintenance alcove and crosses diagonally ahead, hooves striking wet concrete, animals moving at uneven paces, some trotting, one pausing mid-lane before bolting, the dashcam recording the car braking hard as headlights flash and horns sound, tunnel lights strobing off wet fur as vehicles pass beneath overhead lamps.",
    "Forward-facing dashcam POV entering then moving through a sodium-lit tunnel at night: a stray dog and then a small group of deer dart across the lanes ahead at different tunnel sections seconds apart, motion blur consistent with the dog's faster sprint versus the deer's bounding gait, dashcam headlight beams catching eyeshine, the car's own shadow lengthening and swinging across the tunnel wall as the driver swerves to avoid them.",

    # --- Animals on a highway, normal day, dashcam (2) ---
    "Forward-facing dashcam POV on an open highway, clear day: a cow ambles onto the road from a roadside field directly in the car's path, moving at an unhurried, weight-shifting walk with a natural head-bob, traffic ahead slowing in a ripple effect visible through the windshield, the cow's tail swishing and ears flicking at passing engine noise before it steps back onto the verge as the dashcam car passes slowly.",
    "Forward-facing dashcam POV on a rural divided highway, bright afternoon: a troop of monkeys crosses via an overhead cable ahead and drops onto the median, several bounding across the lanes in front of the car with quick springy leaps, one carrying a younger monkey clinging to its back, the dashcam catching the driver braking as a motorbike alongside also slows, one monkey pausing on the centerline to glance at the oncoming dashcam car before continuing.",

    # --- Unusual objects on a highway, dashcam-POV still images (3) ---
    "Photorealistic still image, forward-facing dashcam POV through a bug-flecked windshield, dawn light: a mid-size cargo ship sitting statically on a six-lane highway ahead, its full hull resting flush on the asphalt as if simply placed there, no impact damage, no debris trail, no skid marks, hull plates showing ordinary weathering and rust streaks, a clean natural shadow falling from the hull onto the road, the car's hood and side mirrors visible in frame for scale, a few cars parked at a realistic distance in the foreground lanes, morning haze softening the background.",
    "Photorealistic still image, forward-facing dashcam POV, low dashboard-height angle: a commercial airliner sitting statically on the centerline of a straight highway ahead, fuselage and landing gear resting calmly on the road surface as if parked there, no scorch marks, no debris field, no skid trail, wings level, road markings undisturbed beneath it, the windshield's lower edge and wiper blades visible in frame, correct proportional scale against the highway's lane markings, soft even daylight.",
    "Photorealistic still image, forward-facing dashcam POV, eye-level driver's view through the windshield: a large tree sitting statically across two lanes of a forested highway ahead, resting fully on the road surface with its trunk and branches spread naturally across the asphalt, bark and leaves intact and undisturbed, no scattered debris or drag marks, a calm overcast light, road surface dry and undisturbed elsewhere, a sliver of dashboard visible at the bottom of frame.",
]

# 4. Batch Generation Loop & File Saving
print("\n" + "=" * 50)
print("STARTING PROMPT ENHANCEMENT")
print("=" * 50 + "\n")

output_filename = "enhanced_prompts.txt"
total = len(prompts)

with open(output_filename, "w", encoding="utf-8") as file:
    for idx, original_prompt in enumerate(prompts, start=1):
        messages = [
            {"role": "user", "content": f"Enhance this prompt for a text-to-world generation model: {original_prompt}"}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        upsampled_prompt = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        print(f"[{idx}/{total}] Completed.")
        print(f"--- Original {idx} ---\n{original_prompt}\n")
        print(f"--- Enhanced {idx} ---\n{upsampled_prompt}\n")
        print("-" * 50 + "\n")

        file.write(f"Prompt [{idx}/{total}]\n")
        file.write(f"Original: {original_prompt}\n")
        file.write(f"Enhanced: {upsampled_prompt}\n")
        file.write("-" * 50 + "\n\n")

print(f"All enhanced prompts have been successfully saved to {output_filename}")