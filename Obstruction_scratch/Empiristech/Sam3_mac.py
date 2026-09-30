import os
import tempfile
import cv2
import numpy as np
import uvicorn
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import torch
from ultralytics import SAM

app = FastAPI(title="Dynamic SAM 3 Segmenter API")
model = None

def select_device():
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

@app.on_event("startup")
def load_model():
    global model
    print("Initializing SAM 3 API Server (High-Level Wrapper)...")
    model = SAM("sam3.pt")

@app.post("/v1/segment")
async def segment_dynamic(
    file: UploadFile = File(...),
    labels: str = Form(None),
    points: str = Form(None),
    point_labels: str = Form(None)
):
    global model
    import json
    
    target_phrases = []
    if labels and not points:
        target_phrases = [label.strip() for label in labels.split(",") if label.strip()]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name
        
    try:
        image = cv2.imread(input_path)
        detections = []
        
        if points and point_labels:
            pts = json.loads(points)
            plabels = json.loads(point_labels)
            
            # Using the high-level SAM object routes automatically to the right predictor
            # without triggering language_features KeyErrors.
            try:
                results = model(
                    image, 
                    points=pts, 
                    labels=plabels, 
                    conf=0.50, 
                    device=select_device(), 
                    save=False, 
                    verbose=False
                )
            except torch.cuda.OutOfMemoryError:
                print("GPU OOM detected! Falling back to CPU for SAM3 inference...")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                results = model(
                    image, 
                    points=pts, 
                    labels=plabels, 
                    conf=0.50, 
                    device="cpu", 
                    save=False, 
                    verbose=False
                )
            result = results[0]
            
            if result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()
                boxes_data = result.boxes.data.cpu().numpy() if result.boxes is not None else None
                names = getattr(result, "names", {})

                for j in range(masks_data.shape[0]):
                    if boxes_data is not None and len(boxes_data) > j:
                        cls_id = int(boxes_data[j, 5])
                        score = float(boxes_data[j, 4])
                        bbox = boxes_data[j, :4].tolist()
                    else:
                        cls_id = 0
                        score = 1.0
                        bbox = [0,0,0,0]
                        
                    label_str = names.get(cls_id, f"hazard_zone_{j}") if isinstance(names, dict) else (names[cls_id] if names else f"hazard_zone_{j}")

                    mask_uint8 = (masks_data[j] * 255).astype(np.uint8)
                    _, buffer = cv2.imencode('.png', mask_uint8)
                    mask_b64 = base64.b64encode(buffer).decode('utf-8')

                    detections.append({
                        "label": label_str,
                        "score": score,
                        "bbox": bbox,
                        "mask_b64": mask_b64
                    })
                    
        elif target_phrases:
            from ultralytics.models.sam.predict import SAM3SemanticPredictor
            semantic_predictor = SAM3SemanticPredictor(overrides={
                "conf": 0.50,
                "task": "segment",
                "mode": "predict",
                "model": "sam3.pt",
                "device": select_device(),
                "save": False,
                "verbose": False,
            })
            
            try:
                semantic_predictor.set_image(image)
                results = semantic_predictor(text=target_phrases)
            except torch.cuda.OutOfMemoryError:
                print("GPU OOM detected! Falling back to CPU for SAM3 Semantic inference...")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                semantic_predictor = SAM3SemanticPredictor(overrides={
                    "conf": 0.50,
                    "task": "segment",
                    "mode": "predict",
                    "model": "sam3.pt",
                    "device": "cpu",
                    "save": False,
                    "verbose": False,
                })
                semantic_predictor.set_image(image)
                results = semantic_predictor(text=target_phrases)
            finally:
                semantic_predictor.reset_image()
                semantic_predictor.reset_prompts()
                
            result = results[0]
            
            if result.masks is not None and result.boxes is not None:
                masks_data = result.masks.data.cpu().numpy()
                boxes_data = result.boxes.data.cpu().numpy()
                names = getattr(result, "names", {})

                for j in range(masks_data.shape[0]):
                    cls_id = int(boxes_data[j, 5])
                    label = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else (names[cls_id] if names else f"class_{cls_id}")
                    score = float(boxes_data[j, 4])
                    bbox = boxes_data[j, :4].tolist()

                    mask_uint8 = (masks_data[j] * 255).astype(np.uint8)
                    _, buffer = cv2.imencode('.png', mask_uint8)
                    mask_b64 = base64.b64encode(buffer).decode('utf-8')

                    detections.append({
                        "label": label,
                        "score": score,
                        "bbox": bbox,
                        "mask_b64": mask_b64
                    })

        return JSONResponse(content={"status": "success", "detections": detections})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
