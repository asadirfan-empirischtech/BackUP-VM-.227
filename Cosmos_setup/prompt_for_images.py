import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Configuration
model_id = "mradermacher/Cosmos-1.0-Prompt-Upsampler-12B-Text2World-hf-GGUF"
filename = "Cosmos-1.0-Prompt-Upsampler-12B-Text2World-hf.Q4_K_M.gguf"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)

print("Loading Model onto GPU...")
# Note: Native GGUF loading in transformers requires the `gguf` python package
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    gguf_file=filename,
    device_map="auto"
)

# 2. Define the 5 Base Prompts
prompts = [
    "Front dashcam perspective of a black Toyota Land Cruiser flanked closely by police escort vehicles with flashing red and blue strobe lights, maneuvering smoothly through a narrow chicane lined with bright orange reflective traffic cones on an urban avenue.",
    "First-person dashcam footage from a car traveling on a wet mountain road during heavy rainfall, when a massive landslide suddenly breaks off a steep rocky cliffside, sending massive boulders, churning mud, and fallen trees crashing across the highway.",
    "Eye-level road camera view of multiple passenger cars and SUVs slowing down and navigating across a severely fractured section of highway asphalt, featuring deep potholes, cracked pavement, loose gravel, and roadwork warning signs.",
    "High-angle cinematic drone shot looking down at a steep forested mountain ridge as a catastrophic slope failure triggers a massive debris flow of earth, rock, and mud sweeping across a winding highway below.",
    "Forward-facing dashcam perspective following closely behind a three-wheeled cargo tempo on a busy multi-lane highway, capturing the exact moment an unsecured cardboard parcel falls off the rear bed, bouncing across the roadway."
]

# 3. Batch Generation Loop & File Saving
print("\n" + "="*50)
print("STARTING PROMPT ENHANCEMENT FOR IMAGES")
print("="*50 + "\n")

enhanced_results = []
output_filename = "enhanced_image_prompts.txt"

with open(output_filename, "w", encoding="utf-8") as file:
    for idx, original_prompt in enumerate(prompts, start=1):
        
        # UPGRADE: Instruct the model to focus strictly on still-image attributes
        system_instruction = (
            "You are an expert prompt upsampler for a state-of-the-art text-to-image AI. "
            "Enhance the following base prompt into a highly detailed, descriptive paragraph. "
            "Focus on photographic elements: lighting, textures, camera angle, depth of field, and physical atmosphere. "
            "CRITICAL: This is for a still image. Do not include any motion, movement, or video-related descriptions. "
            "Output ONLY the enhanced prompt."
        )
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": original_prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,  # Reduced from 512 as still image prompts don't need to be as long as video prompts
                do_sample=True,
                temperature=0.6,     # Slightly lower temperature for more focused, less hallucinated physical details
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        upsampled_prompt = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        enhanced_results.append({
            "id": idx,
            "original": original_prompt,
            "enhanced": upsampled_prompt
        })

        # Console Output
        print(f"[{idx}/5] Completed.")
        print(f"--- Original {idx} ---\n{original_prompt}\n")
        print(f"--- Enhanced {idx} ---\n{upsampled_prompt}\n")
        print("-" * 50 + "\n")
        
        # File Output
        file.write(f"Prompt [{idx}/5]\n")
        file.write(f"Original: {original_prompt}\n")
        file.write(f"Enhanced: {upsampled_prompt}\n")
        file.write("-" * 50 + "\n\n")

print(f"All enhanced image prompts have been successfully saved to {output_filename}")