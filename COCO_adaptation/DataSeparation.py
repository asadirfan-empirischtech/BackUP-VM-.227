import os
import json
import glob
from ultralytics import YOLO
from tqdm import tqdm

# --- 1. Configuration ---
model_path = "last.pt"
dataset_folders = ["train2017", "val2017", "test2017"]
output_json_path = "confident_cone_images_85.json"
confidence_threshold = 0.85  # Strictly set to 85%
inference_device = 0  
batch_size = 64  # Optimal for A100 VRAM

def main():
    # --- 2. Gather All Image Paths ---
    all_image_paths = []
    for folder in dataset_folders:
        if os.path.exists(folder):
            folder_images = glob.glob(os.path.join(folder, "*.jpg"))
            all_image_paths.extend(folder_images)
            print(f"Located {len(folder_images)} images in '{folder}'")
        else:
            print(f"Warning: Folder '{folder}' not found. Skipping.")

    total_images = len(all_image_paths)
    if total_images == 0:
        print("No images found. Exiting.")
        return

    print(f"\nStarting YOLO11 inference on {total_images} total images utilizing A100 GPU...")

    # --- 3. Load Model ---
    model = YOLO(model_path)
    
    # --- 4. Run Inference (Batched for A100) ---
    confident_images = []
    
    # Process the massive list in chunks to bypass the YOLO CPU bottleneck
    for i in tqdm(range(0, total_images, batch_size), desc="A100 Batch Processing"):
        
        # Slice out a batch of 64 paths
        batch_paths = all_image_paths[i : i + batch_size]
        
        # Pass the small batch to the model
        results = model.predict(
            source=batch_paths, 
            conf=confidence_threshold, 
            device=inference_device, 
            stream=True, 
            verbose=False 
        )

        # ---------------------------------------------------------
        # THE FIX: Triple-Filter Manual Validation
        # ---------------------------------------------------------
        for input_path, result in zip(batch_paths, results):
            valid_cone_found = False
            
            # Manually iterate through every single detected box in the image
            for box in result.boxes:
                conf = box.conf[0].item()         # Extract exact confidence
                cls_id = int(box.cls[0].item())   # Extract exact class ID
                
                # Calculate bounding box area (width * height)
                # box.xywh returns [x_center, y_center, width, height]
                width = box.xywh[0][2].item()
                height = box.xywh[0][3].item()
                area = width * height
                
                # TRIPLE FILTER:
                # 1. Must be >= 85% confidence
                # 2. Must be Class 0 (Assuming 'cone' is your 0th index class)
                # 3. Must be larger than 400 pixels squared (e.g., a 20x20 box)
                if conf >= 0.85 and cls_id == 0 and area > 400:
                    valid_cone_found = True
                    break  # We only need 1 valid cone to keep the image, stop checking boxes
                    
            if valid_cone_found:
                confident_images.append(input_path)

    # --- 5. Export to JSON ---
    with open(output_json_path, 'w') as json_file:
        json.dump(confident_images, json_file, indent=4)

    print(f"\n--- Process Complete ---")
    print(f"Found {len(confident_images)} images containing traffic cones (>85% confidence).")
    print(f"Paths successfully saved to: {output_json_path}")

if __name__ == "__main__":
    main()