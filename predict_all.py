import os
import sys
import torch
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Mapping for classification levels
DR_LEVELS = {
    0: 'No Diabetic Retinopathy',
    1: 'Mild Diabetic Retinopathy',
    2: 'Moderate Diabetic Retinopathy',
    3: 'Severe Diabetic Retinopathy',
    4: 'Proliferative Diabetic Retinopathy'
}

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "test_results")
    create_dir(results_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Classification ---
    print("\n--- Running Classification ---")
    clf_model_path = os.path.join(base_dir, "Classification", "model", "model.h5")
    clf_model = load_model(clf_model_path, compile=False)
    
    img = cv2.imread(image_path)
    img_clf = cv2.resize(img, (64, 64))
    img_clf = np.reshape(img_clf, [1, 64, 64, 3])
    
    prediction = clf_model.predict(img_clf, verbose=0)
    r = prediction[0][0]
    # Logic from app.py
    class_idx = int(np.round(r)) - 2
    class_idx = max(0, min(4, class_idx))
    print(f"Diagnosis: {DR_LEVELS[class_idx]}")

    # --- 2. Segmentation ---
    print("\n--- Running Segmentation Modules ---")
    segmentation_modules = [
        "Blood Vessel Segmentation",
        "Haemorage Segmentation",
        "Hard Exudate Segmentation",
        "Microanuerism Segmentation",
        "Optical Disc Segmentation",
        "Soft Exudate Segmentation"
    ]

    # Process original image for segmentation (512x512 is common in your modules)
    img_seg_ori = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_seg = cv2.resize(img_seg_ori, (512, 512))
    x = np.transpose(img_seg, (2, 0, 1))
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x.astype(np.float32)).to(device)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes[0, 0].imshow(cv2.cvtColor(img_seg_ori, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    for i, module_name in enumerate(segmentation_modules):
        # We'll use a dynamic import for build_unet to avoid conflicts
        module_path = os.path.join(base_dir, module_name)
        sys.path.insert(0, module_path)
        from model import build_unet
        
        seg_model = build_unet()
        seg_model = seg_model.to(device)
        
        checkpoint_path = os.path.join(module_path, "files", "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            seg_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            seg_model.eval()
            
            with torch.no_grad():
                pred_y = seg_model(x)
                pred_y = torch.sigmoid(pred_y)
                pred_y = pred_y[0][0].cpu().numpy()
                pred_y = (pred_y > 0.5).astype(np.uint8)
            
            # Plotting
            row = (i + 1) // 4
            col = (i + 1) % 4
            axes[row, col].imshow(pred_y, cmap='gray')
            axes[row, col].set_title(module_name.split(" ")[0])
            axes[row, col].axis('off')
            
            # Save individual mask
            cv2.imwrite(os.path.join(results_dir, f"mask_{module_name.replace(' ', '_')}.png"), pred_y * 255)
            print(f"Completed: {module_name}")
        else:
            print(f"Warning: Checkpoint not found for {module_name}")

        sys.path.pop(0)
        # Clear module cache
        for m in ['model', 'utils', 'data', 'loss']:
            if m in sys.modules: del sys.modules[m]

    # Save summary plot
    plt.suptitle(f"Diagnostic Report: {DR_LEVELS[class_idx]}", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diagnostic_summary.png"))
    plt.close()
    
    print(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Default to a sample image if none provided
        sample = os.path.join("Classification", "sample data", "10003_left.jpeg")
        main(sample)
