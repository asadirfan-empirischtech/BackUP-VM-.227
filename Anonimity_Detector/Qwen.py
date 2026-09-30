import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
import time
import uuid

app = FastAPI(title="HuggingFace Qwen Server (OpenAI Compatible)")

# 1. Load the model and processor globally
print("Loading Qwen model into H100 VRAM...")
MODEL_ID = "Qwen/Qwen3.8-27B"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    device_map="auto",
    torch_dtype=torch.bfloat16, # Use bfloat16 for H100
    max_memory={0: "70GiB"}     # Strictly limit GPU usage to 70GB
).eval()

# Optional: To actually speed up native HuggingFace generation on an H100, 
# you can try compiling the generation pass (uncomment below if supported)
# print("Compiling model for speed...")
# model = torch.compile(model)

print("Model loaded and ready on port 8000!")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    openai_messages = body.get("messages", [])
    max_new_tokens = body.get("max_tokens", 1024)
    temperature = body.get("temperature", 0.1)
    
    # 2. Convert OpenAI message format to Qwen Processor format
    qwen_messages = []
    for msg in openai_messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        
        qwen_content = []
        if isinstance(content, str):
            qwen_content.append({"type": "text", "text": content})
        else:
            for block in content:
                if block.get("type") == "text":
                    qwen_content.append({"type": "text", "text": block.get("text")})
                elif block.get("type") == "image_url":
                    # Extract the base64 string or URL
                    img_data = block["image_url"]["url"]
                    # Qwen's apply_chat_template expects the key "image" instead of "image_url"
                    qwen_content.append({"type": "image", "image": img_data})
                    
        qwen_messages.append({"role": role, "content": qwen_content})

    # 3. Apply chat template and tokenize
    start_time = time.time()
    
    inputs = processor.apply_chat_template(
        qwen_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    # 4. Generate Output
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        
    # 5. Decode output
    # Slice the output to only get the newly generated tokens
    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_len:]
    generated_text = processor.decode(generated_tokens, skip_special_tokens=True)
    
    end_time = time.time()
    
    # Calculate throughput for logging
    num_tokens = len(generated_tokens)
    duration = end_time - start_time
    print(f"Generated {num_tokens} tokens in {duration:.2f}s ({num_tokens/duration:.2f} tokens/s)")

    # 6. Format the response exactly like OpenAI so your pipeline doesn't break
    response_data = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": num_tokens,
            "total_tokens": input_len + num_tokens
        }
    }
    
    return JSONResponse(content=response_data)

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

