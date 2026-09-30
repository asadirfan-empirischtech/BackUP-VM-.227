import os
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- 1. Define Paths ---
# Adjust 'train-3' if your latest run saved to a different folder
run_dir = "/home/azureuser/Documents/COCO_adaptation/" 
best_weights = os.path.join(run_dir, "best.pt")
last_weights = os.path.join(run_dir, "last.pt")

# Hardcode the exact image path here
target_image_path = "/home/azureuser/Documents/COCO_adaptation/train2017/000000112250.jpg"

# --- 2. Inference & Display Function ---
def run_and_display(weight_path, image_path, title_label):
    """Loads model, runs inference, and plots the result inline."""
    if not os.path.exists(weight_path):
        print(f"❌ Error: Weights not found at {weight_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return
        
    # Load the specific weights
    model = YOLO(weight_path)
    
    # Run prediction (conf=0.25 filters out extremely weak predictions)
    results = model.predict(image_path, conf=0.25, verbose=False)
    
    # Extract the annotated image array (converting BGR to RGB for matplotlib)
    annotated_img = results[0].plot()[..., ::-1]
    
    # Render in the notebook
    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_img)
    plt.title(f"Inference using: {title_label} ({os.path.basename(weight_path)})")
    plt.axis('off')
    plt.show()

# --- 3. Execute Inference ---
print(f"Target Image: {os.path.basename(target_image_path)}")

print("\nLoading BEST weights...")
run_and_display(best_weights, target_image_path, "BEST Weights")

print("\nLoading LAST weights...")
run_and_display(last_weights, target_image_path, "LAST Weights")