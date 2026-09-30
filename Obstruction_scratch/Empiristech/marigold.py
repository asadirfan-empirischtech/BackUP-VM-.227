from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import subprocess
import os
import uuid
import uvicorn

app = FastAPI()

@app.post("/predict")
async def predict_depth(file: UploadFile = File(...)):
    # Create unique directories for this request to prevent conflicts
    req_id = str(uuid.uuid4())
    in_dir = f"temp_in_{req_id}"
    out_dir = f"temp_out_{req_id}"
    os.makedirs(in_dir, exist_ok=True)
    
    # Save the uploaded image
    input_path = os.path.join(in_dir, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())
        
    # Run Marigold V2 inference
    subprocess.run([
        "python", "scripts/infer.py",
        "--image_dir", in_dir,
        "--output_dir", out_dir
    ])
    
    # Locate the output visualization
    vis_dir = os.path.join(out_dir, "visualizations")
    if os.path.exists(vis_dir):
        output_files = os.listdir(vis_dir)
        if output_files:
            output_image_path = os.path.join(vis_dir, output_files[0])
            return FileResponse(output_image_path)
            
    return {"error": "Inference failed"}

if __name__ == "__main__":
    # Runs the server on port 8002 when executing `python marigold.py`
    uvicorn.run(app, host="0.0.0.0", port=8002)
