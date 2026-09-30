# hf_token = "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" 


# -------------


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Configuration
model_id = "mradermacher/Cosmos-1.0-Prompt-Upsampler-12B-Text2World-hf-GGUF"
filename = "Cosmos-1.0-Prompt-Upsampler-12B-Text2World-hf.Q4_K_M.gguf"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)

print("Loading Model onto GPU...")
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
print("STARTING PROMPT ENHANCEMENT")
print("="*50 + "\n")

enhanced_results = []
output_filename = "enhanced_prompts.txt"

# Open the text file so we can write to it as we loop
with open(output_filename, "w", encoding="utf-8") as file:
    for idx, original_prompt in enumerate(prompts, start=1):
        messages = [
            {"role": "user", "content": f"Enhance this prompt for a text-to-world generation model: {original_prompt}"}
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
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
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

        # Print out to the console
        print(f"[{idx}/5] Completed.")
        print(f"--- Original {idx} ---\n{original_prompt}\n")
        print(f"--- Enhanced {idx} ---\n{upsampled_prompt}\n")
        print("-" * 50 + "\n")
        
        # Write the results to the text file
        file.write(f"Prompt [{idx}/5]\n")
        file.write(f"Original: {original_prompt}\n")
        file.write(f"Enhanced: {upsampled_prompt}\n")
        file.write("-" * 50 + "\n\n")

print(f"All enhanced prompts have been successfully saved to {output_filename}")