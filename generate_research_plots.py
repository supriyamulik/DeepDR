import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import pandas as pd

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_metrics(y_true, y_pred):
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8).reshape(-1)
    
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8).reshape(-1)
    
    return {
        "jaccard": jaccard_score(y_true, y_pred, zero_division=1),
        "f1": f1_score(y_true, y_pred, zero_division=1),
        "recall": recall_score(y_true, y_pred, zero_division=1),
        "precision": precision_score(y_true, y_pred, zero_division=1),
        "acc": accuracy_score(y_true, y_pred)
    }, confusion_matrix(y_true, y_pred, labels=[0, 1])

def plot_confusion_matrix(cm, name, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Background', 'Feature'], yticklabels=['Background', 'Feature'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {name}')
    plt.savefig(os.path.join(save_path, f'cm_{name.replace(" ", "_")}.png'))
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    graph_dir = os.path.join(base_dir, "graphs")
    create_dir(graph_dir)
    
    modules = [
        "Blood Vessel Segmentation",
        "Haemorage Segmentation",
        "Hard Exudate Segmentation",
        "Microanuerism Segmentation",
        "Optical Disc Segmentation",
        "Soft Exudate Segmentation"
    ]
    
    all_metrics = []
    
    for module_name in modules:
        print(f"\n--- Processing {module_name} ---")
        module_path = os.path.join(base_dir, module_name)
        
        # Add module path to sys.path to import model and utils
        sys.path.insert(0, module_path)
        from model import build_unet
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_unet()
        model = model.to(device)
        
        checkpoint_path = os.path.join(module_path, "files", "checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {module_name}: Checkpoint not found at {checkpoint_path}")
            sys.path.pop(0)
            continue
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        test_x = sorted(glob(os.path.join(module_path, "new_data", "test", "image", "*")))
        test_y = sorted(glob(os.path.join(module_path, "new_data", "test", "mask", "*")))
        
        if not test_x:
            print(f"Skipping {module_name}: No test data found.")
            sys.path.pop(0)
            continue
            
        module_total_cm = np.zeros((2, 2), dtype=int)
        module_total_metrics = {"jaccard": 0, "f1": 0, "recall": 0, "precision": 0, "acc": 0}
        
        for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_x)):
            image = cv2.imread(x_path, cv2.IMREAD_COLOR)
            x = np.transpose(image, (2, 0, 1))
            x = x/255.0
            x = np.expand_dims(x, axis=0)
            x = torch.from_numpy(x.astype(np.float32)).to(device)
            
            mask = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
            y = mask/255.0
            
            with torch.no_grad():
                pred_y = model(x)
                pred_y = torch.sigmoid(pred_y)
                pred_y = pred_y[0][0].cpu().numpy()
            
            metrics, cm = calculate_metrics(y, pred_y)
            module_total_cm += cm
            for k in module_total_metrics:
                module_total_metrics[k] += metrics[k]
                
        # Average metrics
        avg_metrics = {k: v/len(test_x) for k, v in module_total_metrics.items()}
        avg_metrics["module"] = module_name
        all_metrics.append(avg_metrics)
        
        # Plot Confusion Matrix
        plot_confusion_matrix(module_total_cm, module_name, graph_dir)
        
        # Remove module path from sys.path
        sys.path.pop(0)
        # Clear modules from cache to avoid conflicts between different build_unet versions
        for m in list(sys.modules.keys()):
            if m in ['model', 'utils', 'data', 'loss']:
                del sys.modules[m]

    # Save summary metrics to CSV
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(graph_dir, "segmentation_metrics.csv"), index=False)
    
    # Plot Bar Chart for Metrics
    plt.figure(figsize=(12, 8))
    df_melted = df.melt(id_vars="module", var_name="Metric", value_name="Score")
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="module")
    plt.title("Segmentation Performance Comparison")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "segmentation_performance_comparison.png"))
    plt.close()
    
    print(f"\nAll graphs and metrics have been saved to the 'graphs' folder.")

if __name__ == "__main__":
    main()